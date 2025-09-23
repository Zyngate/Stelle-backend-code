import os
import re
import ast
import uuid
import json
import asyncio
import datetime
import random
import logging
import pytz
import numpy as np
import httpx
import base64


from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    Query,
    BackgroundTasks,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
    Response,
)
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from urllib.parse import unquote, urlparse, parse_qs
from dotenv import load_dotenv
from pywebpush import webpush, WebPushException

import docx2txt
import fitz
from io import BytesIO
from loguru import logger
from ratelimit import limits, sleep_and_retry
from sentence_transformers import SentenceTransformer
import faiss

from groq import Groq, AsyncGroq
import smtplib
from email.message import EmailMessage
import secrets
from typing import Dict, Union, List, Tuple, Optional
from gen import *
from pydantic import BaseModel
from fastapi import HTTPException
from html2image import Html2Image
from enum import Enum

# Environment and Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
load_dotenv()

VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY")
VAPID_CLAIMS = {"sub": "mailto:info@stelle.world"}

# FastAPI and CORS Initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error, please try again later."},
    )


# Database & FAISS Setup
def get_database():
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
    return client["stelle_db"]


db = get_database()
chats_collection = db["chats"]
memory_collection = db["long_term_memory"]
uploads_collection = db["uploads"]
goals_collection = db["goals"]
users_collection = db["WebPush"]
notifications_collection = db["notifications"]
otp_collection = db["user_otps"]
otp_collection.create_index("created_at", expireAfterSeconds=300)
weekly_plans_collection = db["weekly_plans"]


SMTP_CONFIG = {
    "server": "smtpout.secureserver.net",
    "port": 465,
    "username": os.getenv(
        "SMTP_USERNAME",
    ),
    "password": os.getenv(
        "SMTP_PASSWORD",
    ),
    "from_email": os.getenv(
        "SMTP_USERNAME",
    ),
}

doc_index = faiss.IndexFlatL2(768)
code_index = faiss.IndexFlatL2(768)

user_memory_map = {}
file_doc_memory_map = {}
code_memory_map = {}

local_embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

load_dotenv()

# Step 1: Get all keys that start with 'GROQ_API_KEY_'
available_keys = [
    value for key, value in os.environ.items() if key.startswith("GROQ_API_KEY_")
]

# Step 2: Choose a random key for each client
internet_client = Groq(api_key=random.choice(available_keys))
deepsearch_client = Groq(api_key=random.choice(available_keys))
client = AsyncGroq(api_key=random.choice(available_keys))
# Rate Limiting
CALLS_PER_MINUTE = 50
PERIOD = 60  # seconds


@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
async def query_internet_via_groq(
    query: str, return_sources: bool = False
) -> Union[str, Tuple[str, List[dict]]]:
    """
    Send `query` to Groq; always returns the LLM's content.
    If return_sources=True, also returns a list of {"title","url"} from any search tool.
    """
    try:
        # fire off the chat completion in a thread
        completion = await asyncio.to_thread(
            internet_client.chat.completions.create,
            messages=[{"role": "user", "content": query}],
            model="compound-beta",
        )
        content = completion.choices[0].message.content

        if not return_sources:
            return content

        # collect search‐tool sources, guarding object vs dict shapes
        sources = []
        executed_tools = (
            getattr(completion.choices[0].message, "executed_tools", []) or []
        )
        for tool in executed_tools:
            if (
                tool.type == "search"
                and hasattr(tool, "search_results")
                and tool.search_results
            ):
                raw = tool.search_results
                # raw may be an object with .results or a dict with ["results"]
                hits = getattr(raw, "results", None) or raw.get("results", [])
                for result in hits:
                    if isinstance(result, dict):
                        title = result.get("title")
                        url = result.get("url")
                    else:
                        title = getattr(result, "title", None)
                        url = getattr(result, "url", None)

                    if title and url:
                        sources.append({"title": title, "url": url})

        return content, sources

    except Exception as e:
        logging.error(f"Error querying Groq API: {e}", exc_info=True)
        if return_sources:
            return "Error accessing internet information.", []
        return "Error accessing internet information."


# Utility Functions
def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%B %d, %Y, %I:%M %p")


def generate_otp():
    return str(secrets.randbelow(900000) + 100000)


def send_email(to_email, otp):
    msg = EmailMessage()
    msg.set_content(f"Your OTP is {otp}. It is valid for 5 minutes.")
    msg["Subject"] = "Your OTP for Password Reset"
    msg["From"] = SMTP_CONFIG["from_email"]
    msg["To"] = to_email
    try:
        with smtplib.SMTP_SSL(SMTP_CONFIG["server"], SMTP_CONFIG["port"]) as server:
            server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
            server.send_message(msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")


def filter_think_messages(messages: list) -> list:
    filtered = []
    for msg in messages:
        content = msg.get("content") or ""
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if cleaned:
            new_msg = msg.copy()
            new_msg["content"] = cleaned
            filtered.append(new_msg)
    return filtered


def convert_object_ids(document: dict) -> dict:
    for key, value in document.items():
        if key == "_id":
            document[key] = str(value)
        elif isinstance(value, dict):
            document[key] = convert_object_ids(value)
        elif isinstance(value, list):
            document[key] = [
                convert_object_ids(item) if isinstance(item, dict) else item
                for item in value
            ]
    return document


def extract_code_segments(code: str) -> list:
    segments = []
    try:
        tree = ast.parse(code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.body[-1].lineno if node.body else node.lineno
                code_lines = code.splitlines()[start:end]
                segment_code = "\n".join(code_lines)
                segments.append({"name": node.name, "code": segment_code})
    except Exception as e:
        logging.warning(f"AST parse failed: {e}. Falling back to manual chunking.")
    if not segments:
        lines = code.splitlines()
        chunk_size = 300
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i : i + chunk_size])
            segments.append({"name": f"chunk_{i//chunk_size+1}", "code": chunk})
    return segments


def split_text_into_chunks(
    text: str, chunk_size: int = 500, overlap: int = 100
) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


# In main16.py, add these functions, for example, after the "Utility Functions" section


async def generate_seed_keywords(query: str) -> list:
    """Generate 3 seed keywords from the user's query."""
    prompt = f"Generate 8 seed keywords based on the following query: {query}. Separate the keywords with commas. Must output only kewords nothing else"
    # Use the existing 'client' from main16.py
    completion = await asyncio.to_thread(
        client.chat.completions.create,
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_completion_tokens=1024,
        reasoning_format="hidden",
    )
    response = completion.choices[0].message.content
    seed_keywords = response.split(", ")
    if len(seed_keywords) != 3:
        # It's better to handle this gracefully than to raise a ValueError in a production app
        # For instance, you could try to split by comma, or return the first 3 words.
        # For this integration, we'll keep the logic but you may want to refine it.
        logger.warning(
            f"Expected 3 seed keywords, but got {len(seed_keywords)}. Using what was returned."
        )
    return seed_keywords


async def fetch_trending_hashtags(seed_keywords: list) -> list:
    """Fetch up to 5 trending hashtags per seed keyword."""
    hashtags = []
    for keyword in seed_keywords:
        prompt = f"Browse intagram ,X and reddit and fetch the hastags related to {keyword} make sure those keywords trending and have potential to boost SEO. Only provide the thirty different hashtags separated by spaces. Must output only hastags nothing else"
        # Use the existing 'internet_client' for Browse tasks
        completion = await asyncio.to_thread(
            internet_client.chat.completions.create,
            model="OpenAI-Agent",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=3000,
            stream=False,
        )
        response = completion.choices[0].message.content
        keyword_hashtags = response.split(" ")
        hashtags.extend(keyword_hashtags)
    return list(set(hashtags))[:30]  # Limit to 30 unique hashtags


async def fetch_seo_keywords(seed_keywords: list) -> list:
    """Fetch up to 5 SEO keywords per seed keyword."""
    seo_keywords = []
    for keyword in seed_keywords:
        prompt = f"Go to internet and see the tob blogs and posts related to {keyword}. Only provide the keywords separated by commas. Must output only kewords nothing else"
        # Use the existing 'internet_client' for Browse tasks
        completion = await asyncio.to_thread(
            internet_client.chat.completions.create,
            model="compound-beta-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
        )
        response = completion.choices[0].message.content
        keyword_seo = response.split(", ")[:5]
        seo_keywords.extend(keyword_seo)
    return list(set(seo_keywords))[:15]  # Limit to 15 unique keywords


async def generate_caption(
    query: str, seed_keywords: list, trending_hashtags: list, seo_keywords: list
) -> str:
    """Generate a caption with a hook using query, keywords, hashtags, and SEO terms."""
    prompt = f"Write a good caption with a starting hook for social media content about {query}. Use the following seed keywords: {', '.join(seed_keywords)}. Incorporate some of these trending hashtags: {', '.join(trending_hashtags)}. Also, consider these SEO keywords: {', '.join(seo_keywords)}. make sure caption is almost 50 words long and does not include m-dash(—)"
    # Use the existing 'client' for generation
    completion = await asyncio.to_thread(
        client.chat.completions.create,
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_completion_tokens=1024,
        reasoning_format="raw",
    )
    return completion.choices[0].message.content


# Pydantic Models
class GenerateRequest(BaseModel):
    user_id: str
    session_id: str
    prompt: str
    filenames: list[str] = []


class GenerateResponse(BaseModel):
    response: str


class BrowseRequest(BaseModel):
    query: str


class BrowseResponse(BaseModel):
    result: str


class Subscription(BaseModel):
    user_id: str
    subscription: dict
    time_zone: str = "UTC"


class OTPRequest(BaseModel):
    email: EmailStr


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str


class Prompt(BaseModel):
    text: str


class DeepSearchRequest(BaseModel):
    user_id: str
    session_id: str
    prompt: str
    filenames: list[str] = []


class NLPRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    filenames: list[str] = []


class ResearchQuery(BaseModel):
    text: str


class RegenerateRequest(BaseModel):
    user_id: str
    session_id: str
    filenames: list[str] = []


class PlanWeekRequest(BaseModel):
    user_id: str


class InstaPostRequest(BaseModel):
    prompt: str


class UserInput(BaseModel):
    query: str


class PostGenOptions(str, Enum):
    Text = "text"
    Photo = "photo"
    Video = "video"

# New pydantic models
# class Task(BaseModel):

#     task_id: str
#     title: str
#     status: str
#     deadline: Optional[str] = None


# class Goal(BaseModel):
#     goal_id: str
#     title: str
#     description: str
#     status: str
#     tasks: List[Task]


# class PlanWeekRequest(BaseModel):
#     user_id: str
#     planning_horizon_weeks: Optional[int] = 4  # Default to 4 weeks


# class SubTask(BaseModel):
#     title: str
#     description: str


# class PlannedTask(BaseModel):
#     task_id: str
#     title: str
#     status: str
#     description: str
#     sub_tasks: List[SubTask]


# class DayPlan(BaseModel):
#     date: str
#     day_of_week: str
#     daily_focus: str
#     tasks: List[PlannedTask]


# class WeeklyPlan(BaseModel):
#     week_number: int
#     week_start_date: str
#     week_end_date: str
#     strategic_focus: str
#     days: List[DayPlan]


# class MultiWeekPlanResponse(BaseModel):
#     overall_strategic_approach: str
#     planning_horizon_weeks: int
#     weekly_plans: List[WeeklyPlan]

# # new code - Mohit
# def calculate_planning_horizon(goals_data: List[Dict]) -> int:
#     """Calculate the planning horizon based on goal deadlines"""
#     max_weeks = 26  # Maximum 12 weeks planning horizon
#     today = datetime.datetime.now().date()

#     latest_deadline = today
#     for goal in goals_data:
#         for task in goal.get("tasks", []):
#             deadline_str = task.get("deadline")
#             if deadline_str:
#                 try:
#                     # Extract date part only (in case there's time included)
#                     deadline_date_str = deadline_str.split(" ")[0]
#                     deadline_dt = datetime.datetime.strptime(
#                         deadline_date_str, "%Y-%m-%d"
#                     ).date()
#                     if deadline_dt > latest_deadline:
#                         latest_deadline = deadline_dt
#                 except (ValueError, TypeError):
#                     continue

#     # Calculate weeks between today and latest deadline
#     if latest_deadline > today:
#         days_diff = (latest_deadline - today).days
#         weeks_needed = min(
#             max_weeks, (days_diff // 7) + 1
#         )  # Add 1 to include partial weeks
#         return max(1, weeks_needed)  # At least 1 week

#     return 4  # Default to 4 weeks if no future deadlines

# # new code - Mohit
# def generate_week_dates(start_date: datetime.date, week_number: int) -> tuple:
#     """Generate start and end dates for a specific week"""
#     week_start = start_date + datetime.timedelta(weeks=week_number)
#     week_end = week_start + datetime.timedelta(days=6)
#     return week_start, week_end


# Document Extraction Functions
async def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        logging.info(f"PDF text extracted (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logging.error(f"PDF extraction error: {e}", exc_info=True)
        return ""


async def extract_text_from_docx(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        contents = await file.read()
        text = docx2txt.process(BytesIO(contents))
        logging.info(f"DOCX text extracted (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logging.error(f"DOCX extraction error: {e}", exc_info=True)
        return ""


async def extract_text_from_txt(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        contents = await file.read()
        text = contents.decode("utf-8")
        logging.info(f"TXT text extracted (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logging.error(f"TXT extraction error: {e}", exc_info=True)
        return ""


# Embedding Generation (Local)
async def generate_text_embedding(text: str | None) -> list:
    if not text:
        logging.warning(
            "generate_text_embedding called with empty or None text; returning empty embedding."
        )
        return []
    try:
        embedding = await asyncio.to_thread(
            local_embedding_model.encode, text, convert_to_numpy=True
        )
        embedding_list = embedding.tolist()
        if len(embedding_list) != 768:
            logging.error(f"Embedding has unexpected length: {len(embedding_list)}")
            return []
        return embedding_list
    except Exception as e:
        logging.error(f"Local embedding generation error: {e}", exc_info=True)
        return []


# Groq API Integration Functions
async def content_for_website(content: str) -> str:
    prompt = (
        f"Summarize the following content concisely:\n\n{content}\n\n"
        "List key themes and provide a brief final summary."
    )
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY_CONTENT"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are a content analysis expert."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=700,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Content summary error: {e}", exc_info=True)
        return "Error generating content summary."


async def detailed_explanation(content: str) -> str:
    prompt = (
        "Provide a detailed explanation by listing key themes and challenges, "
        "and then generate a comprehensive summary of the content below:\n\n" + content
    )
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY_EXPLANATION"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are an expert analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=700,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Detailed explanation error: {e}", exc_info=True)
        return "Error generating detailed explanation."


async def classify_prompt(prompt: str) -> str:
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY_CLASSIFY"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {
                    "role": "system",
                    "content": "Determine if this query requires real time research. Respond with 'research' or 'no research'.",
                },
                {"role": "user", "content": prompt},
            ],
            model="llama-3.1-8b-instant",
        )
        reply = response.choices[0].message.content.strip().lower()
        logging.info(f"Classify prompt response: {reply}")
        return reply
    except Exception as e:
        logging.error(f"Classify prompt error: {e}", exc_info=True)
        return "no research"


async def browse_and_generate(user_query: str) -> str:
    current_date = get_current_datetime()
    query_with_date = f"Provide information on the following query: {user_query.strip()}. Today’s date/time is: {current_date}"
    logging.info(f"Browse query: {query_with_date}")
    try:
        response = await query_internet_via_groq(query_with_date)
        logging.info(f"Groq response: {response[:300]}...")
        return response
    except Exception as e:
        logging.error(f"Browse and generate error: {e}", exc_info=True)
        return "Error processing browse and generate request."


# Multi-Modal Retrieval Integration
async def retrieve_multimodal_context(
    query: str, session_id: str, filenames: list[str] = None
) -> tuple[str, set]:
    try:
        embedding = await generate_text_embedding(query)
        contexts = []
        used_filenames = set()
        if embedding and doc_index.ntotal > 0:
            query_vector = np.array(embedding, dtype="float32").reshape(1, -1)
            k = 10
            distances, indices = doc_index.search(query_vector, k)
            file_chunks = {}
            for idx in indices[0]:
                meta = file_doc_memory_map.get(idx)
                if meta and (
                    meta.get("session_id") == session_id
                    or meta.get("session_id") != session_id
                ):
                    if filenames and meta.get("filename") not in filenames:
                        continue
                    filename = meta["filename"]
                    if filename not in file_chunks:
                        file_chunks[filename] = []
                    file_chunks[filename].append(
                        (idx, meta, distances[0][indices[0].tolist().index(idx)])
                    )
            for filename, chunks in file_chunks.items():
                top_chunks = sorted(chunks, key=lambda x: x[8])[:8]
                for idx, meta, _ in top_chunks:
                    chunk = await uploads_collection.find_one(
                        {
                            "user_id": meta["user_id"],
                            "session_id": session_id,
                            "filename": filename,
                            "chunk_index": meta.get("chunk_index"),
                        }
                    )
                    if chunk and chunk.get("query_count", 0) < 15:
                        snippet = f"From {filename} (Chunk {meta.get('chunk_index', 'N/A')}):\n{meta['text_snippet']}"
                        contexts.append(snippet)
                        used_filenames.add(filename)
        if embedding and code_index.ntotal > 0:
            query_vector = np.array(embedding, dtype="float32").reshape(1, -1)
            k = 8
            distances, indices = code_index.search(query_vector, k)
            for idx in indices[0]:
                meta = code_memory_map.get(idx)
                if meta and meta.get("session_id") == session_id:
                    if filenames and meta.get("filename") not in filenames:
                        continue
                    chunk = await uploads_collection.find_one(
                        {
                            "user_id": meta["user_id"],
                            "session_id": session_id,
                            "filename": meta["filename"],
                            "segment_name": meta.get("segment_name"),
                        }
                    )
                    if chunk and chunk.get("query_count", 0) < 15:
                        snippet = f"From {meta['filename']} (Code Segment: {meta.get('segment_name', 'N/A')}):\n{meta['text_snippet']}"
                        contexts.append(snippet)
                        used_filenames.add(meta["filename"])
        return "\n\n".join(contexts), used_filenames
    except Exception as e:
        logging.error(f"Error during multimodal retrieval: {e}", exc_info=True)
        return "", set()


# Long-Term Memory Functions
async def efficient_summarize(
    previous_summary: str,
    new_messages: list,
    user_id: str,
    max_summary_length: int = 500,
) -> str:
    user_queries = "\n".join(
        [msg["content"] for msg in new_messages if msg["role"] == "user"]
    )
    context_text = f"User ID: {user_id}\n"
    if previous_summary:
        context_text += f"Previous Summary:\n{previous_summary}\n\n"
    context_text += f"New User Queries:\n{user_queries}\n\n"
    active_goals = await goals_collection.find(
        {"user_id": user_id, "status": "active"}
    ).to_list(None)
    goals_context = ""
    if active_goals:
        goals_context = "User's current goals and tasks:\n"
        for goal in active_goals:
            goals_context += f"- Goal: {goal['title']} ({goal['status']})\n"
            for task in goal["tasks"]:
                goals_context += f"  - Task: {task['title']} ({task['status']})\n"
    context_text += goals_context
    summary_prompt = (
        f"Based on the following context, generate a concise summary (max {max_summary_length} characters) "
        f"that captures the user's interests, style, and ongoing goals:\n\n{context_text}"
    )
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY_MEMORY_SUMMARY"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI that creates personalized conversation summaries.",
                },
                {"role": "user", "content": summary_prompt},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Long-term memory summarization error: {e}", exc_info=True)
        return previous_summary if previous_summary else "Summary unavailable."


async def store_long_term_memory(user_id: str, session_id: str, new_messages: list):
    try:
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        previous_summary = mem_entry.get("summary", "") if mem_entry else ""
        new_summary = await efficient_summarize(previous_summary, new_messages, user_id)
        new_vector = await generate_text_embedding(new_summary)
        new_vector_np = np.array(new_vector, dtype="float32").reshape(1, -1)
        if mem_entry:
            await memory_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "summary": new_summary,
                        "session_id": session_id,
                        "vector": new_vector,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    }
                },
            )
            if user_id in user_memory_map:
                doc_index.remove_ids(
                    np.array([user_memory_map[user_id]], dtype="int64")
                )
            idx = doc_index.ntotal
            doc_index.add(new_vector_np)
            user_memory_map[user_id] = idx
        else:
            await memory_collection.insert_one(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "summary": new_summary,
                    "vector": new_vector,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                }
            )
            idx = doc_index.ntotal
            doc_index.add(new_vector_np)
            user_memory_map[user_id] = idx
        logging.info(f"Long-term memory updated for user {user_id}")
    except Exception as e:
        logging.error(f"Error storing long-term memory: {e}", exc_info=True)


# Push Notification System Implementation
@app.post("/subscribe")
async def subscribe(subscription: Subscription):
    try:
        user_filter = {"user_id": subscription.user_id}
        update_data = {
            "$set": {
                "push_subscription": subscription.subscription,
                "time_zone": subscription.time_zone,
            }
        }
        await users_collection.update_one(user_filter, update_data, upsert=True)
        return {"success": True, "message": "Subscription stored successfully."}
    except Exception as e:
        logging.error(f"Error in subscription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store subscription.")


async def schedule_notification(
    user_id: str,
    message: str,
    scheduled_time: datetime.datetime,
    notif_type: str = "general",
):
    notif = {
        "user_id": user_id,
        "message": message,
        "scheduled_time": scheduled_time,
        "type": notif_type,
        "sent": False,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
    }
    await notifications_collection.insert_one(notif)
    logging.info(
        f"Notification scheduled for user {user_id} at {scheduled_time} with type '{notif_type}'."
    )


async def notification_checker():
    while True:
        now = datetime.datetime.now(datetime.timezone.utc)
        cursor = notifications_collection.find(
            {"scheduled_time": {"$lte": now}, "sent": False}
        )
        async for notif in cursor:
            user_id = notif["user_id"]
            message = notif["message"]
            user = await users_collection.find_one({"user_id": user_id})
            if not user or "push_subscription" not in user:
                logging.warning(
                    f"No subscription found for user {user_id}. Marking notification as skipped."
                )
                await notifications_collection.update_one(
                    {"_id": notif["_id"]},
                    {
                        "$set": {
                            "sent": True,
                            "sent_at": datetime.datetime.now(datetime.timezone.utc),
                            "status": "skipped_no_subscription",
                        }
                    },
                )
                continue
            subscription_info = user["push_subscription"]
            payload = json.dumps(
                {
                    "title": "Stelle Team",
                    "body": message,
                    "icon": "https://media-hosting.imagekit.io/2b232c0c6a354b82/2.png?Expires=1839491017&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=0aDSb0TotX2vejnzaFZMFXJwrUHnW2bBoedYufEzfJXip-gsz6jztBphzMCAJT800MYhdLeWVuV2Vf0vlSKLAE~38kxJno7d8QwNqOVq5~SH~2tZ5Pi8~4L16RZsfvxs2QdXMp~3Md9GPAWhJLZX5wsXdKpXEgi4BYT9qUOyJj7mzzqjoV9O7m6lFGKO8RhuoVyumDWv3dn6FmOc69UegOU5qmpfgTBdSrBQle5YRQBAfGYWqpTRWiFN-cQFa42ORLDLZbctNkqmZqXhoxM0ZqGyQYhUeKdogx9r32M9ssuuebQ4GyG8cRQaMDyE6dUmSvFluPFxUL1TCmOzHIK9dQ__",
                    "data": {"url": "https://stelle.chat"},
                }
            )
            try:
                webpush(
                    subscription_info,
                    data=payload,
                    vapid_private_key=VAPID_PRIVATE_KEY,
                    vapid_claims=VAPID_CLAIMS,
                )
                await notifications_collection.update_one(
                    {"_id": notif["_id"]},
                    {
                        "$set": {
                            "sent": True,
                            "sent_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    },
                )
                logging.info(f"Push notification sent to user {user_id}")
            except WebPushException as ex:
                await asyncio.sleep(60)


async def daily_checkin_scheduler():
    while True:
        async for user in users_collection.find():
            user_id = user["user_id"]
            tz_name = user.get("time_zone", "UTC")
            try:
                user_tz = pytz.timezone(tz_name)
            except Exception:
                user_tz = pytz.UTC
            now_local = datetime.datetime.now(user_tz)
            checkin_time_local = now_local.replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            if 8 * 60 + 55 <= (now_local.hour * 60 + now_local.minute) <= 9 * 60 + 5:
                checkin_time_utc = user_tz.localize(checkin_time_local).astimezone(
                    pytz.UTC
                )
                existing = await notifications_collection.find_one(
                    {
                        "user_id": user_id,
                        "type": "daily_checkin",
                        "scheduled_time": checkin_time_utc,
                    }
                )
                if existing:
                    continue
                goals_cursor = goals_collection.find(
                    {"user_id": user_id, "status": {"$in": ["active", "in progress"]}}
                )
                active_goals = []
                async for g in goals_cursor:
                    active_goals.append(g["title"])
                if active_goals:
                    message = f"Good morning! Your goals for today: {', '.join(active_goals)}. Keep it up!"
                else:
                    message = "Good morning! Set some goals today to stay on track!"
                await schedule_notification(
                    user_id, message, checkin_time_utc, notif_type="daily_checkin"
                )
                logging.info(
                    f"Scheduling daily check-in for user {user_id} at {checkin_time_utc}"
                )
        await asyncio.sleep(60)


async def proactive_checkin_scheduler():
    checkin_times = [
        {"hour": 9, "minute": 0, "type": "proactive_morning"},
        {"hour": 14, "minute": 0, "type": "proactive_afternoon"},
        {"hour": 19, "minute": 0, "type": "proactive_evening"},
    ]
    while True:
        async for user in users_collection.find():
            user_id = user["user_id"]
            tz_name = user.get("time_zone", "UTC")
            try:
                user_tz = pytz.timezone(tz_name)
            except Exception:
                user_tz = pytz.UTC
            now_local = datetime.datetime.now(user_tz)
            for checkin in checkin_times:
                target = now_local.replace(
                    hour=checkin["hour"],
                    minute=checkin["minute"],
                    second=0,
                    microsecond=0,
                )
                target_naive = target.replace(tzinfo=None)
                if abs((now_local - target).total_seconds()) <= 15 * 60:
                    target_utc = user_tz.localize(target_naive).astimezone(pytz.UTC)
                    existing = await notifications_collection.find_one(
                        {
                            "user_id": user_id,
                            "type": checkin["type"],
                            "scheduled_time": target_utc,
                        }
                    )
                    if existing:
                        continue
                    goals_cursor = goals_collection.find(
                        {
                            "user_id": user_id,
                            "status": {"$in": ["active", "in progress"]},
                        }
                    )
                    active_goals = []
                    async for g in goals_cursor:
                        active_goals.append(g["title"])
                    if active_goals:
                        message = f"Hi there! How are you progressing on your goals ({', '.join(active_goals)})? We're here to support you!"
                    else:
                        message = "Hi there! How are you doing today? Let us know if you need any help."
                    await schedule_notification(
                        user_id, message, target_utc, notif_type=checkin["type"]
                    )
                    logging.info(
                        f"Scheduling proactive check-in for user {user_id} at {target_utc} with type '{checkin['type']}'"
                    )
        await asyncio.sleep(60)


async def schedule_immediate_reminder(user_id: str, reminder_text: str):
    user_info = await users_collection.find_one({"user_id": user_id})
    tz_name = user_info.get("time_zone", "UTC") if user_info else "UTC"
    try:
        user_tz = pytz.timezone(tz_name)
    except Exception:
        user_tz = pytz.UTC
    now_utc = datetime.datetime.now(pytz.UTC)
    now_local = now_utc.astimezone(user_tz)
    scheduled_local = now_local + datetime.timedelta(minutes=1)
    scheduled_time_utc = scheduled_local.astimezone(pytz.UTC)
    await schedule_notification(
        user_id, f"Reminder: {reminder_text}", scheduled_time_utc, notif_type="reminder"
    )
    logging.info(f"Immediate reminder scheduled for user {user_id}: {reminder_text}")


# Rate limiting for Groq calls
@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
async def rate_limited_groq_call(*args, **kwargs):
    return await asyncio.to_thread(
        deepsearch_client.chat.completions.create, *args, **kwargs
    )


async def clarify_query(query: str) -> str:
    prompt = (
        "You are an expert prompt engineer. Refine and optimize the user’s original query "
        "into the most effective, concise search prompt possible, considering the user's context.\n"
        f"Original query: {query}\n"
        "Return ONLY the optimized query—no explanations or extra text."
    )
    try:
        response = await rate_limited_groq_call(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error clarifying query: {e}")
        return query


async def clarify_response(query: str) -> str:
    prompt = (
        "You are an expert person who unerstands the user and return that what you understand by user querry and what is your next steps to adress the answers to querry. also make sure your response is in proper slack markdown "
        "into the most effective,  considering the user's context.\n"
        f"Original query: {query}\n"
    )
    try:
        response = await rate_limited_groq_call(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=900,
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error clarifying query: {e}")
        return clarify_response


async def generate_keywords(clarified_query: str) -> List[str]:
    current_date = get_current_datetime()
    prompt = (
        "Todays date is "
        + current_date
        + ". Act as a expert web Browse agent and give 1 search queries from the following query as a JSON array of strings, "
        'e.g. ["term1",]:\n\n'
        f"{clarified_query}"
    )
    try:
        response = await rate_limited_groq_call(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
        )
        keywords_text = response.choices[0].message.content.strip()
        try:
            return json.loads(keywords_text)
        except json.JSONDecodeError:
            parts = re.split(r"[\n,]+", keywords_text)
            return [kw.strip().strip('"') for kw in parts if kw.strip()]
    except Exception as e:
        logging.error(f"Error generating keywords: {e}")
        return [clarified_query]


async def understanding_query(clarify_response: str) -> List[str]:
    current_date = get_current_datetime()
    prompt = (
        "You are proceeding agent You understand the querry that come to you and act like you are doing some research on the querry, in last end the response like you proceeding to generate final answer also make sure your response is in proper slack markdown"
        f"{clarify_response}"
    )
    try:
        response = await rate_limited_groq_call(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=900,
            temperature=0.6,
        )
        keywords_text = response.choices[0].message.content.strip()
        try:
            return json.loads(keywords_text)
        except json.JSONDecodeError:
            parts = re.split(r"[\n,]+", keywords_text)
            return [kw.strip().strip('"') for kw in parts if kw.strip()]
    except Exception as e:
        logging.error(f"Error generating keywords: {e}")
        return [understanding_query]


async def deep_search(query_data: Dict[str, str], websocket: WebSocket):
    user_id = query_data["user_id"]
    session_id = query_data["session_id"]
    main_query = query_data["prompt"]
    filenames = query_data["filenames"]

    try:
        await websocket.send_json(
            {"step": "start", "message": "Starting deep search..."}
        )

        # Fetch memory summary
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        memory_summary = mem_entry.get("summary", "") if mem_entry else ""

        # Clarify query
        clarified_query = await clarify_query(main_query)
        clarifyd_response = await clarify_response(main_query)
        await websocket.send_json(
            {
                "step": "clarified_query",
                "message": f"Clarified response: {clarifyd_response}",
            }
        )

        # Generate keywords
        keywords = await generate_keywords(clarified_query)
        understanding = await understanding_query(clarified_query)
        await websocket.send_json(
            {
                "step": "keywords_generated",
                "message": f"Processing request: {understanding}",
            }
        )

        all_responses = []
        all_sources = []
        for keyword in keywords:
            response, sources = await query_internet_via_groq(
                f"Provide information on {keyword}", return_sources=True
            )
            if response and response != "Error accessing internet information.":
                all_responses.append(response)
                all_sources.extend(sources)
                await websocket.send_json(
                    {
                        "step": "response_received",
                        "keyword": keyword,
                        "response": (
                            response[:200] + "..." if len(response) > 200 else response
                        ),
                        "sources": [
                            source["url"] for source in sources
                        ],  # Modified to send only URLs
                    }
                )

        if not all_responses:
            await websocket.send_json(
                {"step": "no_content", "message": "No information retrieved."}
            )
            return

        # Synthesize final answer
        combined_responses = "\n\n".join(all_responses)
        prompt = (
            f"Based on the following information, provide a detailed and insightful answer to the query:\n"
            f"'{main_query}'\n\nInformation:\n{combined_responses}"
        )
        try:
            completion = await rate_limited_groq_call(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1800,
                temperature=0.7,
            )
            final_answer = completion.choices[0].message.content.strip()
            # Remove duplicate sources based on URL
            unique_sources = [
                source["url"]
                for source in {source["url"]: source for source in all_sources}.values()
            ]
            await websocket.send_json(
                {
                    "step": "final_answer",
                    "message": final_answer,
                    "sources": unique_sources,  # Modified to send only URLs
                }
            )
        except Exception as e:
            logging.error(f"Error synthesizing final answer: {e}")
            await websocket.send_json(
                {"step": "error", "message": "Error generating final answer."}
            )

        await websocket.send_json({"step": "end", "message": "Deep search complete!"})
        return final_answer  # Return final_answer for use in WebSocket endpoint
    except Exception as e:
        logging.error(f"Deep search error: {e}")
        await websocket.send_json(
            {"step": "error", "message": "An unexpected error occurred."}
        )
        return "Error during deep search."


# Endpoints
@app.post("/upload")
async def upload_file(
    user_id: str = Form(...),
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
):
    allowed_text_types = [
        "text/plain",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    allowed_code_extensions = [".py", ".js", ".java", ".cpp", ".c", ".ts"]
    responses = []
    for file in files:
        try:
            modality = "document"
            filename_lower = file.filename.lower()
            ext = os.path.splitext(filename_lower)[1]
            if (
                file.content_type in allowed_text_types
                or ext in allowed_code_extensions
            ):
                if ext in allowed_code_extensions:
                    modality = "code"
                    extracted_text = await extract_text_from_txt(file)
                    code_segments = extract_code_segments(extracted_text)
                    if not code_segments:
                        responses.append(
                            {
                                "filename": file.filename,
                                "success": False,
                                "message": "Code segmentation failed",
                            }
                        )
                        continue
                    for segment in code_segments:
                        segment_text = segment["code"]
                        embedding_vector = await generate_text_embedding(segment_text)
                        if not embedding_vector:
                            continue
                        new_vector = np.array(
                            embedding_vector, dtype="float32"
                        ).reshape(1, -1)
                        new_id = code_index.ntotal
                        code_index.add(new_vector)
                        snippet = segment_text[:300]
                        code_memory_map[new_id] = {
                            "user_id": user_id,
                            "session_id": session_id,
                            "filename": file.filename,
                            "modality": modality,
                            "segment_name": segment["name"],
                            "text_snippet": snippet,
                            "usage_count": 0,
                        }
                        await uploads_collection.insert_one(
                            {
                                "user_id": user_id,
                                "session_id": session_id,
                                "filename": file.filename,
                                "modality": modality,
                                "segment_name": segment["name"],
                                "text_snippet": snippet,
                                "embedding": embedding_vector,
                                "timestamp": datetime.datetime.now(
                                    datetime.timezone.utc
                                ),
                                "query_count": 0,
                            }
                        )
                    responses.append(
                        {
                            "filename": file.filename,
                            "success": True,
                            "segments": len(code_segments),
                        }
                    )
                    logging.info(
                        f"Code file '{file.filename}' segmented into {len(code_segments)} segments for user {user_id} in session {session_id}"
                    )
                else:
                    if file.content_type == "application/pdf":
                        extracted_text = await extract_text_from_pdf(file)
                    elif file.content_type in [
                        "application/msword",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ]:
                        extracted_text = await extract_text_from_docx(file)
                    elif file.content_type == "text/plain":
                        extracted_text = await extract_text_from_txt(file)
                    if not extracted_text:
                        responses.append(
                            {
                                "filename": file.filename,
                                "success": False,
                                "message": "Parsing Failed",
                            }
                        )
                        continue
                    chunks = split_text_into_chunks(extracted_text)
                    for i, chunk in enumerate(chunks):
                        embedding_vector = await generate_text_embedding(chunk)
                        if not embedding_vector:
                            continue
                        new_vector = np.array(
                            embedding_vector, dtype="float32"
                        ).reshape(1, -1)
                        new_id = doc_index.ntotal
                        doc_index.add(new_vector)
                        snippet = chunk[:300]
                        file_doc_memory_map[new_id] = {
                            "user_id": user_id,
                            "session_id": session_id,
                            "filename": file.filename,
                            "modality": modality,
                            "chunk_index": i,
                            "text_snippet": snippet,
                            "usage_count": 0,
                        }
                        await uploads_collection.insert_one(
                            {
                                "user_id": user_id,
                                "session_id": session_id,
                                "filename": file.filename,
                                "modality": modality,
                                "chunk_index": i,
                                "text_snippet": snippet,
                                "embedding": embedding_vector,
                                "timestamp": datetime.datetime.now(
                                    datetime.timezone.utc
                                ),
                                "query_count": 0,
                            }
                        )
                    responses.append(
                        {
                            "filename": file.filename,
                            "success": True,
                            "chunks": len(chunks),
                        }
                    )
                    logging.info(
                        f"Document file '{file.filename}' uploaded with {len(chunks)} chunks for user {user_id} in session {session_id}"
                    )
            else:
                responses.append(
                    {
                        "filename": file.filename,
                        "success": False,
                        "message": "Format not allowed",
                    }
                )
        except Exception as e:
            logging.error(
                f"Error processing file '{file.filename}': {e}", exc_info=True
            )
            responses.append(
                {
                    "filename": file.filename,
                    "success": False,
                    "message": "File Processing Failed",
                }
            )
    return {"results": responses}


@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: Request, background_tasks: BackgroundTasks):
    try:
        current_date = get_current_datetime()
        data = await request.json()
        req = GenerateRequest(**data)
        user_id, session_id, user_message, filenames = (
            req.user_id,
            req.session_id,
            req.prompt,
            req.filenames,
        )
        if not user_id or not session_id or not user_message:
            raise HTTPException(status_code=400, detail="Invalid request parameters.")

        active_goals = await goals_collection.find(
            {"user_id": user_id, "status": {"$in": ["active", "in progress"]}}
        ).to_list(None)
        goals_context = ""
        if active_goals:
            goals_context = "User's current goals and tasks:\n"
            for goal in active_goals:
                goals_context += f"- Goal: {goal['title']} ({goal['status']}) [ID: {goal.get('goal_id','N/A')}]\n"
                for task in goal["tasks"]:
                    goals_context += f"  - Task: {task['title']} ({task['status']}) [ID: {task.get('task_id','N/A')}]\n"

        uploaded_files = await uploads_collection.distinct(
            "filename", {"session_id": session_id}
        )
        mentioned_filenames = [
            fn for fn in uploaded_files if fn.lower() in user_message.lower()
        ]
        hooked_filenames = filenames if filenames else mentioned_filenames
        logger.info(f"Hooked filenames: {hooked_filenames}")

        external_content = ""
        url_match = re.search(r"https?://[^\s]+", user_message)
        if url_match:
            url = url_match.group(0)
            logging.info(f"Detected URL in prompt: {url}")
            if "youtube.com" in url or "youtu.be" in url:
                external_content = await query_internet_via_groq(
                    f"Summarize the content of the YouTube video at {url}"
                )
                external_content = await detailed_explanation(external_content)
            else:
                external_content = await query_internet_via_groq(
                    f"Summarize the content of the webpage at {url}"
                )
                external_content = await content_for_website(external_content)

        multimodal_context, used_filenames = await retrieve_multimodal_context(
            user_message, session_id, hooked_filenames
        )
        unified_prompt = f"User Query: {user_message}\n"
        if external_content:
            unified_prompt += f"\n[External Content]:\n{external_content}\n"
        if multimodal_context:
            unified_prompt += (
                f"\n[Retrieved File & Code Context]:\n{multimodal_context}\n"
            )
        unified_prompt += f"\nCurrent Date/Time: {current_date}\n\nProvide a detailed and context-aware response."

        research_needed = await classify_prompt(user_message)
        if research_needed == "research" and not multimodal_context:
            research_results = await browse_and_generate(user_message)
            if research_results:
                unified_prompt += f"\n\n[Additional Research]:\n{research_results}"

        chat_entry = await chats_collection.find_one(
            {"user_id": user_id, "session_id": session_id}
        )
        if chat_entry:
            past_messages = chat_entry.get("messages", [])
            if past_messages:
                # Filter valid embeddings (must be lists of 768 floats)
                past_embeddings = [
                    msg["embedding"]
                    for msg in past_messages
                    if "embedding" in msg
                    and isinstance(msg["embedding"], list)
                    and len(msg["embedding"]) == 768
                ]
                if past_embeddings:
                    logging.info(
                        f"Found {len(past_embeddings)} valid embeddings in chat history."
                    )
                    past_embeddings = np.array(past_embeddings)
                    current_embedding = await generate_text_embedding(user_message)
                    if current_embedding and len(current_embedding) == 768:
                        distances = np.linalg.norm(
                            past_embeddings - np.array(current_embedding), axis=1
                        )
                        n = len(past_messages)
                        ages = np.array([n - 1 - i for i in range(n)])
                        lambda_val = 0.05
                        modified_distances = distances + lambda_val * ages
                        k = 7
                        top_k_indices = np.argsort(modified_distances)[:k]
                        m = 5
                        last_m_indices = (
                            list(range(n - m, n)) if n >= m else list(range(n))
                        )
                        combined_indices = list(
                            set(top_k_indices.tolist() + last_m_indices)
                        )
                        sorted_indices = sorted(combined_indices)
                        chat_history = [past_messages[i] for i in sorted_indices]
                    else:
                        logging.warning(
                            "Current message embedding is invalid; using recent messages only."
                        )
                        chat_history = (
                            filter_think_messages(past_messages[-2:])
                            if len(past_messages) >= 2
                            else filter_think_messages(past_messages)
                        )
                else:
                    logging.info("No valid embeddings found; using recent messages.")
                    chat_history = (
                        filter_think_messages(past_messages[-2:])
                        if len(past_messages) >= 2
                        else filter_think_messages(past_messages)
                    )
            else:
                chat_history = []
        else:
            chat_history = []

        long_term_memory = ""
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        if mem_entry and "summary" in mem_entry:
            long_term_memory = mem_entry["summary"]

        system_prompt = (
            "You are Stelle, a strategic, empathetic AI assistant with autonomous goal/task management. "
            "If you have to add tasks to a goal, beforehand make the task id then add it to the goal. "
            "When the user sets a new goal, use '[GOAL_SET: <goal_title>]' Must use '[TASK: <task_desc>]' lines. for adding tasks. "
            "To delete a goal: '[GOAL_DELETE: <goal_id>]'. To delete a task: '[TASK_DELETE: <task_id>]'. "
            "To add a new task: '[TASK_ADD: <goal_id>: <task_description>]'. "
            "To modify a task's title: '[TASK_MODIFY: <task_id>: <new_title_or_description>]'. "
            "To start a goal: '[GOAL_START: <goal_id>]'. To start a task: '[TASK_START: <task_id>]'. "
            "To complete a goal: '[GOAL_COMPLETE: <goal_id>]'. To complete a task: '[TASK_COMPLETE: <task_id>]'. "
            "Must ask user for deadlines using '[TASK_DEADLINE: <task_id>: <YYYY-MM-DD HH:MM>]' and log progress using '[TASK_PROGRESS: <task_id>: <progress_description>]'.\n"
            f"Current date/time: {current_date}\n"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if long_term_memory:
            messages.append(
                {"role": "system", "content": f"Long-term memory: {long_term_memory}"}
            )
        if goals_context:
            messages.append({"role": "system", "content": goals_context})
        for msg in chat_history:
            cleaned = re.sub(
                r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL
            ).strip()
            if cleaned:
                if len(cleaned) > 800:
                    cleaned = cleaned[:800] + "…"
                if cleaned:
                    messages.append({"role": msg["role"], "content": cleaned})
        messages.append({"role": "user", "content": unified_prompt})
        logging.info(f"LLM prompt messages: {messages}")

        generate_api_keys = [
            os.getenv("GROQ_API_KEY_GENERATE_1"),
            os.getenv("GROQ_API_KEY_GENERATE_2"),
            os.getenv("GROQ_API_KEY_GENERATE_3"),
        ]
        generate_api_keys = [k for k in generate_api_keys if k]
        if not generate_api_keys:
            raise HTTPException(
                status_code=500,
                detail="No valid GROQ_API_KEY_GENERATE environment variables found.",
            )
        selected_key = random.choice(generate_api_keys)
        client_generate = AsyncGroq(api_key=selected_key)

        stream = await client_generate.chat.completions.create(
            messages=messages,
            model="deepseek-r1-distill-llama-70b",
            max_completion_tokens=4000,
            temperature=0.7,
            stream=True,
        )

        async def generate_stream():
            full_reply = ""
            async for chunk in stream:
                delta = (
                    chunk.choices[0].delta.content
                    if chunk.choices[0].delta.content
                    else ""
                )
                full_reply += delta
                yield delta
            reply_content = full_reply.strip()
            new_goals_map = {}
            goal_set_matches = re.findall(r"\[GOAL_SET: (.*?)\]", reply_content)
            for goal_phrase in goal_set_matches:
                goal_id = str(uuid.uuid4())
                new_goals_map[goal_phrase] = goal_id
                existing_goal = await goals_collection.find_one(
                    {
                        "user_id": user_id,
                        "title": goal_phrase,
                        "status": {"$in": ["active", "in progress"]},
                    }
                )
                if existing_goal:
                    logging.info(
                        f"Skipping creation of duplicate goal '{goal_phrase}' for user {user_id}."
                    )
                    continue
                new_goal = {
                    "user_id": user_id,
                    "goal_id": goal_id,
                    "session_id": session_id,
                    "title": goal_phrase,
                    "description": "",
                    "status": "active",
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                    "tasks": [],
                }
                task_matches = re.findall(r"\[TASK: (.*?)\]", reply_content)
                for task_desc in task_matches:
                    task_id = str(uuid.uuid4())
                    new_task = {
                        "task_id": task_id,
                        "title": task_desc,
                        "description": "",
                        "status": "not started",
                        "created_at": datetime.datetime.now(datetime.timezone.utc),
                        "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        "deadline": None,
                        "progress": [],
                    }
                    new_goal["tasks"].append(new_task)
                await goals_collection.insert_one(new_goal)
                logging.info(
                    f"Goal set: '{goal_phrase}' with {len(task_matches)} tasks for user {user_id}"
                )

            goal_delete_matches = re.findall(r"\[GOAL_DELETE: (.*?)\]", reply_content)
            for gid in goal_delete_matches:
                real_goal_id = new_goals_map.get(gid, gid)
                result = await goals_collection.delete_one(
                    {"user_id": user_id, "goal_id": real_goal_id}
                )
                if result.deleted_count > 0:
                    logging.info(f"Goal {real_goal_id} deleted for user {user_id}")
                else:
                    logging.warning(
                        f"Goal {real_goal_id} not found or could not be deleted."
                    )

            task_delete_matches = re.findall(r"\[TASK_DELETE: (.*?)\]", reply_content)
            for tid in task_delete_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {"$pull": {"tasks": {"task_id": tid}}},
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} deleted for user {user_id}")
                else:
                    logging.warning(f"Task {tid} not found or could not be deleted.")

            task_add_matches = re.findall(
                r"\[TASK_ADD:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for goal_id_str, task_desc in task_add_matches:
                real_goal_id = new_goals_map.get(goal_id_str, goal_id_str)
                task_id = str(uuid.uuid4())
                new_task = {
                    "task_id": task_id,
                    "title": task_desc,
                    "description": "",
                    "status": "not started",
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                    "deadline": None,
                    "progress": [],
                }
                result = await goals_collection.update_one(
                    {"user_id": user_id, "goal_id": real_goal_id},
                    {
                        "$push": {"tasks": new_task},
                        "$set": {
                            "updated_at": datetime.datetime.now(datetime.timezone.utc)
                        },
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Added task '{task_desc}' to goal {real_goal_id}")
                else:
                    logging.warning(
                        f"Could not add task to goal {real_goal_id} (not found?)."
                    )

            task_modify_matches = re.findall(
                r"\[TASK_MODIFY:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for tid, new_desc in task_modify_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.title": new_desc,
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} modified to '{new_desc}'")
                else:
                    logging.warning(f"Task {tid} not found for modification.")

            goal_start_matches = re.findall(r"\[GOAL_START: (.*?)\]", reply_content)
            for gid in goal_start_matches:
                real_goal_id = new_goals_map.get(gid, gid)
                result = await goals_collection.update_one(
                    {"user_id": user_id, "goal_id": real_goal_id},
                    {
                        "$set": {
                            "status": "in progress",
                            "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Goal {real_goal_id} started (in progress).")
                else:
                    logging.warning(f"Goal {real_goal_id} not found for GOAL_START.")

            task_start_matches = re.findall(r"\[TASK_START: (.*?)\]", reply_content)
            for tid in task_start_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.status": "in progress",
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} started (in progress).")
                else:
                    logging.warning(f"Task {tid} not found for TASK_START.")

            goal_complete_matches = re.findall(
                r"\[GOAL_COMPLETE: (.*?)\]", reply_content
            )
            for gid in goal_complete_matches:
                real_goal_id = new_goals_map.get(gid, gid)
                result = await goals_collection.update_one(
                    {"user_id": user_id, "goal_id": real_goal_id},
                    {
                        "$set": {
                            "status": "completed",
                            "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Goal {real_goal_id} marked as completed.")
                else:
                    logging.warning(f"Goal {real_goal_id} not found for GOAL_COMPLETE.")

            task_complete_matches = re.findall(
                r"\[TASK_COMPLETE: (.*?)\]", reply_content
            )
            for tid in task_complete_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.status": "completed",
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} marked as completed.")
                else:
                    logging.warning(f"Task {tid} not found for completion.")

            task_deadline_matches = re.findall(
                r"\[TASK_DEADLINE:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for tid, deadline_str in task_deadline_matches:
                try:
                    deadline_dt = datetime.datetime.strptime(
                        deadline_str, "%Y-%m-%d %H:%M"
                    )
                except Exception as ex:
                    logging.error(
                        f"Invalid deadline format for task {tid}: {deadline_str}"
                    )
                    continue
                user_info = await users_collection.find_one({"user_id": user_id})
                tz_name = user_info.get("time_zone", "UTC") if user_info else "UTC"
                try:
                    user_tz = pytz.timezone(tz_name)
                except Exception:
                    user_tz = pytz.UTC
                localized_deadline = user_tz.localize(deadline_dt)
                reminder_time = localized_deadline - datetime.timedelta(days=1)
                reminder_time_utc = reminder_time.astimezone(pytz.UTC)
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.deadline": deadline_str,
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Set deadline for Task {tid} to {deadline_str}")
                    await schedule_notification(
                        user_id,
                        f"Reminder: Task {tid} is due on {deadline_str}",
                        reminder_time_utc,
                        notif_type="deadline_reminder",
                    )
                else:
                    logging.warning(f"Task {tid} not found for deadline update.")

            task_progress_matches = re.findall(
                r"\[TASK_PROGRESS:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for tid, progress_desc in task_progress_matches:
                progress_entry = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "description": progress_desc,
                }
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$push": {"tasks.$.progress": progress_entry},
                        "$set": {
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            )
                        },
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Added progress entry to Task {tid}: {progress_desc}")
                else:
                    logging.warning(f"Task {tid} not found for progress update.")

            remind_match = re.search(r"remind me (.+)", user_message, re.IGNORECASE)
            if remind_match:
                reminder_text = remind_match.group(1).strip()
                await schedule_immediate_reminder(user_id, reminder_text)

            lines = reply_content.split("\n")
            clean_lines = [
                line for line in lines if not re.match(r"\[.*?: .*?\]", line.strip())
            ]
            reply_content_clean = "\n".join(clean_lines).strip()

            # Generate embeddings and ensure they are valid
            user_embedding = await generate_text_embedding(user_message)
            if user_embedding and len(user_embedding) != 768:
                logging.warning("Invalid user embedding; not storing.")
                user_embedding = None

            assistant_embedding = await generate_text_embedding(reply_content_clean)
            if assistant_embedding and len(assistant_embedding) != 768:
                logging.warning("Invalid assistant embedding; not storing.")
                assistant_embedding = None

            new_messages = [
                {
                    "role": "user",
                    "content": user_message,
                    **({"embedding": user_embedding} if user_embedding else {}),
                },
                {
                    "role": "assistant",
                    "content": reply_content_clean,
                    **(
                        {"embedding": assistant_embedding}
                        if assistant_embedding
                        else {}
                    ),
                },
            ]

            if chat_entry:
                await chats_collection.update_one(
                    {"user_id": user_id, "session_id": session_id},
                    {
                        "$push": {"messages": {"$each": new_messages}},
                        "$set": {
                            "last_updated": datetime.datetime.now(datetime.timezone.utc)
                        },
                    },
                )
            else:
                await chats_collection.insert_one(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "messages": new_messages,
                        "last_updated": datetime.datetime.now(datetime.timezone.utc),
                    }
                )

            if chat_entry and len(chat_entry.get("messages", [])) >= 10:
                background_tasks.add_task(
                    store_long_term_memory,
                    user_id,
                    session_id,
                    chat_entry["messages"][-10:],
                )

            for filename in used_filenames:
                await uploads_collection.update_many(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "filename": filename,
                    },
                    {"$inc": {"query_count": 1}},
                )

            cursor = uploads_collection.find(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "query_count": {"$gte": 15},
                }
            )
            documents_to_remove = set()
            async for chunk in cursor:
                documents_to_remove.add(chunk["filename"])
            for filename in documents_to_remove:
                await uploads_collection.delete_many(
                    {"user_id": user_id, "session_id": session_id, "filename": filename}
                )
                indices_to_remove = [
                    idx
                    for idx, m in file_doc_memory_map.items()
                    if m["filename"] == filename and m["session_id"] == session_id
                ]
                if indices_to_remove:
                    doc_index.remove_ids(np.array(indices_to_remove, dtype="int64"))
                    for idx in indices_to_remove:
                        del file_doc_memory_map[idx]
                code_indices_to_remove = [
                    idx
                    for idx, m in code_memory_map.items()
                    if m["filename"] == filename and m["session_id"] == session_id
                ]
                if code_indices_to_remove:
                    code_index.remove_ids(
                        np.array(code_indices_to_remove, dtype="int64")
                    )
                    for idx in code_indices_to_remove:
                        del code_memory_map[idx]

        return StreamingResponse(generate_stream(), media_type="text/plain")
    except Exception as e:
        logging.error(f"Error in /generate endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal error processing your request."
        )


@app.get("/chat-history")
async def get_chat_history(user_id: str = Query(...), session_id: str = Query(...)):
    try:
        chat_entry = await chats_collection.find_one(
            {"user_id": user_id, "session_id": session_id}, {"messages": 1}
        )
        if chat_entry and "messages" in chat_entry:
            return {"messages": filter_think_messages(chat_entry["messages"])}
        return {"messages": []}
    except Exception as e:
        logging.error(f"Chat history retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving chat history.")


@app.get("/get-goals")
async def get_goals(user_id: str = Query(...)):
    try:
        goals = await goals_collection.find({"user_id": user_id}).to_list(None)
        if not goals:
            return {"goals": []}
        for goal in goals:
            goal = convert_object_ids(goal)
            goal["created_at"] = goal["created_at"].isoformat()
            goal["updated_at"] = goal["updated_at"].isoformat()
            for task in goal["tasks"]:
                if "_id" in task:
                    task["_id"] = str(task["_id"])
                task["created_at"] = task["created_at"].isoformat()
                task["updated_at"] = task["updated_at"].isoformat()
                if task.get("deadline"):
                    if isinstance(task["deadline"], datetime.datetime):
                        task["deadline"] = task["deadline"].isoformat()
                    else:
                        task["deadline"] = str(task["deadline"])
                for progress in task.get("progress", []):
                    progress["timestamp"] = progress["timestamp"].isoformat()
        return {"goals": goals}
    except Exception as e:
        logging.error(f"Error retrieving goals for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving goals.")


@app.get("/history")
async def get_history(user_id: str = Query(...)):
    try:
        sessions = await chats_collection.find({"user_id": user_id}).to_list(None)
        history = []
        for session in sessions:
            session_id = session.get("session_id")
            messages = session.get("messages", [])
            time = session.get("last_updated")
            filtered_messages = filter_think_messages(messages)
            first_message = filtered_messages[0]["content"] if filtered_messages else ""
            history.append(
                {"session_id": session_id, "first_message": first_message, "time": time}
            )
        return {"history": history}
    except Exception as e:
        logging.error(f"Error retrieving session history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving session history.")


@app.get("/get-quote")
async def get_quote(user_id: str = Query(...)):
    try:
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        if not mem_entry or "summary" not in mem_entry:
            raise HTTPException(status_code=404, detail="User summary not found.")
        summary = mem_entry["summary"]
        prompt = (
            f"Based on the following user summary, generate a single-line quote that captures "
            f"the essence of the user's interests or personality and today's focus based on user goal. The quote should be concise. "
            f"Must return quote and todays focus noting else. Summary: {summary}"
        )
        client = Groq(api_key=os.getenv("GROQ_API_KEY_BROWSE_ENDPOINT"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=50,
            temperature=0.6,
        )
        quote = response.choices[0].message.content.strip()
        logging.info(f"Generated quote for user {user_id}: {quote}")
        return {"quote": quote}
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error generating quote for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating quote.")


@app.get("/recommended-content")
async def recommended_content_endpoint(
    user_id: str = Query(..., description="Unique identifier for the user")
):
    try:
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        if not mem_entry or "summary" not in mem_entry:
            raise HTTPException(status_code=404, detail="User summary not found.")
        summary = mem_entry["summary"]
        logger.info(f"Retrieved summary for user {user_id}: {summary[:100]}...")
        subqueries = await generate_subqueries(summary, num_subqueries=3)
        if not subqueries:
            logger.warning(f"No subqueries generated for user {user_id}.")
            return {"recommended_content": []}
        logger.info(f"Generated subqueries for user {user_id}: {subqueries}")
        recommended_content = []
        for subquery in subqueries:
            response = await query_internet_via_groq(
                f"Provide a brief summary of {subquery}"
            )
            if response and response != "Error accessing internet information.":
                description = (
                    response[:150] + "..." if len(response) > 150 else response
                )
                recommended_content.append(
                    {
                        "title": subquery,
                        "type": "summary",
                        "description": description,
                        "link": None,
                    }
                )
        logger.info(
            f"Returning {len(recommended_content)} content items for user {user_id}"
        )
        return {"recommended_content": recommended_content}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(
            f"Error in /recommended-content for user {user_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error retrieving recommended content."
        )


@app.post("/send-otp")
async def send_otp(request: OTPRequest):
    email = request.email
    otp = generate_otp()
    await otp_collection.insert_one(
        {
            "email": email,
            "otp": otp,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
        }
    )
    send_email(email, otp)
    return {"message": "OTP sent", "success": True}


@app.post("/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    email = request.email
    otp = request.otp
    stored_otp = await otp_collection.find_one({"email": email})
    if stored_otp is None:
        raise HTTPException(
            status_code=400, detail="No OTP found for this email or it has expired"
        )
    if stored_otp["otp"] != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    await otp_collection.delete_one({"email": email})
    return {"message": "OTP verified", "success": True}


# Research and WebSocket Functions
async def generate_subqueries(main_query: str, num_subqueries: int = 3) -> list[str]:
    current_dt = get_current_datetime()
    prompt = (
        f"todays_date {current_dt} Provide exactly {num_subqueries} distinct search queries related to the following topic:\n"
        f"'{main_query}'\n"
        "Write each query on a separate line, with no numbering or additional text."
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=150,
            temperature=0.7,
        )
        response = chat_completion.choices[0].message.content.strip()
        raw_queries = [line.strip() for line in response.splitlines() if line.strip()]
        clean_queries = []
        for q in raw_queries:
            q = re.sub(r"^['\"]|['\"]$", "", q).strip()
            if q:
                clean_queries.append(q)
        unique_subqueries = list(dict.fromkeys(clean_queries))[:num_subqueries]
        logger.info(f"Generated subqueries for '{main_query}': {unique_subqueries}")
        return unique_subqueries
    except Exception as e:
        logger.error(f"Subquery generation error for '{main_query}': {e}")
        return []


async def visual_generate_subqueries(main_query: str, num_subqueries: int = 3) -> list:
    current_dt = get_current_datetime()
    prompt = (
        f"As of {current_dt} Provide exactly {num_subqueries} distinct search queries related to the following visualization topic:\n"
        f"'{main_query}'\n"
        "Write each query on a separate line, with no numbering or additional text."
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=150,
            temperature=0.7,
        )
        response = chat_completion.choices[0].message.content.strip()
        raw_queries = [line.strip() for line in response.splitlines() if line.strip()]
        clean_queries = []
        for q in raw_queries:
            q = re.sub(r"^['\"]|['\"]$", "", q).strip()
            if q:
                clean_queries.append(q)
        unique_subqueries = list(dict.fromkeys(clean_queries))[:num_subqueries]
        logger.info(f"Generated subqueries for '{main_query}': {unique_subqueries}")
        return unique_subqueries
    except Exception as e:
        logger.error(f"Subquery generation error for '{main_query}': {e}")
        return []


async def visual_synthesize_result(
    main_query: str, contents: list, max_context: int = 4000
) -> str:
    trimmed_contents = [c[:1000] for c in contents if c]
    combined_content = " ".join(trimmed_contents)[:max_context]
    prompt = (
        f"Based on the following content, provide a concise, accurate, and well-structured answer to the visualization prompt:\n"
        f"'{main_query}'\n\nContent:\n{combined_content}"
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            temperature=0.7,
        )
        result = chat_completion.choices[0].message.content.strip()
        logger.info(f"Synthesized result for '{main_query}'.")
        return result
    except Exception as e:
        logger.error(f"Error during synthesis for '{main_query}': {e}")
        return "Error generating the result."


async def generate_html_visualization(content: str) -> str:
    prompt = f"Generate a complete HTML document code that presents the following research result in a professional and modern layout. The document should have a dark theme with a background color of #000000 and use the brand color #6ee2f5 for accents and interactive elements. Include interactive visualizations such as pie charts, bar graphs, or other diagrams to represent the data mentioned in the result . Use modern CSS styling to create a clean and sleek design, and ensure the layout is structured similar to an A4 sheet, with text explanations and visualizations properly aligned and if you use maintainAspectRatio make sure its true. Incorporate interactive features like hover effects, clickable elements, or animations to enhance user engagement. Output only the HTML code, with all styles and scripts included in the file. Result: '{content}'"
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",
            max_tokens=8000,
            temperature=0.8,
            reasoning_format="hidden",
        )
        html_code = chat_completion.choices[0].message.content.strip()
        logger.info("Generated HTML visualization code.")
        return html_code
    except Exception as e:
        logger.error(f"Error generating HTML visualization: {e}")
        return "<html><body><h1>Error generating HTML visualization.</h1></body></html>"


async def visualization_process(main_query: str, websocket: WebSocket):
    try:
        await websocket.send_json(
            {"step": "start", "message": "Starting visualization research..."}
        )

        await websocket.send_json(
            {"step": "generating_subqueries", "message": "Generating subqueries..."}
        )
        subqueries = await visual_generate_subqueries(main_query)
        if not subqueries:
            await websocket.send_json(
                {"step": "error", "message": "Failed to generate subqueries."}
            )
            return
        await websocket.send_json(
            {"step": "subqueries_generated", "subqueries": subqueries}
        )

        all_contents = []
        for subquery in subqueries:
            await websocket.send_json(
                {
                    "step": "searching",
                    "subquery": subquery,
                    "message": f"Searching for '{subquery}'...",
                }
            )
            content = await query_internet_via_groq(
                f"Provide information on {subquery}"
            )
            if content and content != "Error accessing internet information.":
                all_contents.append(content)
                await websocket.send_json(
                    {
                        "step": "content_received",
                        "subquery": subquery,
                        "content": (
                            content[:200] + "..." if len(content) > 200 else content
                        ),
                    }
                )

        if not all_contents:
            await websocket.send_json(
                {
                    "step": "no_content",
                    "message": "No content retrieved from any subquery.",
                }
            )
            return

        await websocket.send_json(
            {"step": "synthesizing", "message": "Synthesizing research result..."}
        )
        synthesized = await visual_synthesize_result(main_query, all_contents)
        await websocket.send_json({"step": "synthesized", "result": synthesized})

        await websocket.send_json(
            {"step": "generating_html", "message": "Generating HTML visualization..."}
        )
        html_code = await generate_html_visualization(synthesized)
        await websocket.send_json({"step": "html_generated", "html": html_code})

        await websocket.send_json({"step": "end", "message": "Visualization complete."})
    except WebSocketDisconnect:
        logger.info("Client disconnected during visualization process.")
    except Exception as e:
        logger.error(f"Visualization process error: {e}")
        await websocket.send_json(
            {"step": "error", "message": f"Unexpected error: {str(e)}"}
        )


async def synthesize_result(
    main_query: str, contents: list[str], max_context: int = 4000
) -> str:
    trimmed_contents = [c[:1000] for c in contents if c]
    combined_content = " ".join(trimmed_contents)[:max_context]
    prompt = (
        f"Based on the following collected information, provide a concise, accurate, and well-structured answer to the query:\n"
        f"'{main_query}'\n\nInformation:\n{combined_content}"
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            temperature=0.7,
        )
        result = chat_completion.choices[0].message.content.strip()
        logger.info(f"Synthesized result for '{main_query}'.")
        return result
    except Exception as e:
        logger.error(f"Error during synthesis for '{main_query}': {e}")
        return "Error generating the final result."


async def research_process(main_query: str, websocket: WebSocket):
    try:
        await websocket.send_json({"step": "start", "message": "Starting research..."})
        await websocket.send_json(
            {"step": "generating_subqueries", "message": "Generating subqueries..."}
        )
        subqueries = await generate_subqueries(main_query)
        if not subqueries:
            await websocket.send_json(
                {"step": "error", "message": "Failed to generate subqueries."}
            )
            return
        await websocket.send_json(
            {"step": "subqueries_generated", "subqueries": subqueries}
        )
        all_contents = []
        for subquery in subqueries:
            await websocket.send_json(
                {
                    "step": "querying",
                    "subquery": subquery,
                    "message": f"Querying for '{subquery}'...",
                }
            )
            content = await query_internet_via_groq(subquery)
            if content and content != "Error accessing internet information.":
                all_contents.append(content)
                await websocket.send_json(
                    {
                        "step": "response_received",
                        "subquery": subquery,
                        "response": (
                            content[:200] + "..." if len(content) > 200 else content
                        ),
                    }
                )
        if not all_contents:
            await websocket.send_json(
                {
                    "step": "no_content",
                    "message": "No content retrieved from any subquery.",
                }
            )
            return
        await websocket.send_json(
            {"step": "synthesizing", "message": "Synthesizing final result..."}
        )
        final_result = await synthesize_result(main_query, all_contents)
        await websocket.send_json({"step": "final_result", "result": final_result})
        await websocket.send_json({"step": "end", "message": "Research complete."})
    except WebSocketDisconnect:
        logger.info("Client disconnected during research process.")
    except Exception as e:
        logger.error(f"Research process error: {e}")
        await websocket.send_json(
            {"step": "error", "message": f"Unexpected error: {str(e)}"}
        )


@app.post("/start_visualization")
async def start_visualization(prompt: Prompt):
    visual_query_id = str(uuid.uuid4())
    queries[visual_query_id] = prompt.text
    logger.info(f"Received visualization prompt: {prompt.text} (ID: {visual_query_id})")
    return {"query_id": visual_query_id}


@app.websocket("/ws/visualization/{query_id}")
async def websocket_endpoint_visualization(websocket: WebSocket, query_id: str):
    await websocket.accept()
    try:
        main_query = queries.get(query_id)
        if not main_query:
            await websocket.send_json({"step": "error", "message": "Invalid query ID."})
            return
        await visualization_process(main_query, websocket)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for query ID: {query_id}")
    finally:
        await websocket.close()


@app.post("/start_research")
async def start_research(query: ResearchQuery):
    query_id = str(uuid.uuid4())
    queries[query_id] = query.text
    logger.info(f"Received research request: {query.text} (ID: {query_id})")
    return {"query_id": query_id}


queries = {}
deepsearch_queries: Dict[str, Dict[str, str]] = {}


@app.websocket("/ws/{query_id}")
async def websocket_endpoint_research(websocket: WebSocket, query_id: str):
    await websocket.accept()
    try:
        main_query = queries.get(query_id)
        if not main_query:
            await websocket.send_json({"step": "error", "message": "Invalid query ID."})
            return
        await research_process(main_query, websocket)
    except WebSocketDisconnect:
        logger.info(f"WebSocket closed for query ID: {query_id}")
    finally:
        await websocket.close()


@app.post("/start_deepsearch")
async def start_deepsearch(request: DeepSearchRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    query_id = str(uuid.uuid4())
    deepsearch_queries[query_id] = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "prompt": request.prompt,
        "filenames": request.filenames,
    }
    logger.info(f"Deep search initiated: {query_id}, prompt: {request.prompt}")
    return {"query_id": query_id}


@app.websocket("/ws/deepsearch/{query_id}")
async def deepsearch_websocket_endpoint(websocket: WebSocket, query_id: str):
    await websocket.accept()
    try:
        query_data = deepsearch_queries.get(query_id)
        if not query_data:
            await websocket.send_json({"step": "error", "message": "Invalid query ID"})
            return

        user_id = query_data["user_id"]
        session_id = query_data["session_id"]
        user_message = query_data["prompt"]
        filenames = query_data["filenames"]

        multimodal_context, _ = await retrieve_multimodal_context(
            user_message, session_id, filenames
        )
        research_needed = "research"

        if research_needed == "research" and not multimodal_context:
            final_answer = await deep_search(query_data, websocket)
            user_embedding = await generate_text_embedding(user_message)
            assistant_embedding = await generate_text_embedding(final_answer)
            new_messages = [
                {"role": "user", "content": user_message, "embedding": user_embedding},
                {
                    "role": "assistant",
                    "content": final_answer,
                    "embedding": assistant_embedding,
                },
            ]
            chat_entry = await chats_collection.find_one(
                {"user_id": user_id, "session_id": session_id}
            )
            if chat_entry:
                await chats_collection.update_one(
                    {"user_id": user_id, "session_id": session_id},
                    {
                        "$push": {"messages": {"$each": new_messages}},
                        "$set": {
                            "last_updated": datetime.datetime.now(datetime.timezone.utc)
                        },
                    },
                )
            else:
                await chats_collection.insert_one(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "messages": new_messages,
                        "last_updated": datetime.datetime.now(datetime.timezone.utc),
                    }
                )
            if chat_entry and len(chat_entry.get("messages", [])) >= 10:
                await store_long_term_memory(
                    user_id, session_id, chat_entry["messages"][-10:]
                )
        else:
            await websocket.send_json(
                {"step": "standard_response", "message": "Using standard generation..."}
            )
            await websocket.send_json({"step": "end", "message": "Response complete"})
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {query_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"step": "error", "message": "Server error occurred"})
    finally:
        deepsearch_queries.pop(query_id, None)
        await websocket.close()


# NLP Endpoint for Real-Time Voice Conversation
@app.websocket("/nlp")
async def nlp_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            nlp_data = json.loads(data)
            req = NLPRequest(**nlp_data)
            user_id, session_id, user_message, filenames = (
                req.user_id,
                req.session_id,
                req.message,
                req.filenames,
            )

            if not user_id or not session_id or not user_message:
                await websocket.send_json({"error": "Invalid request parameters"})
                continue

            current_date = get_current_datetime()

            active_goals = await goals_collection.find(
                {"user_id": user_id, "status": {"$in": ["active", "in progress"]}}
            ).to_list(None)
            goals_context = ""
            if active_goals:
                goals_context = "User's current goals and tasks:\n"
                for goal in active_goals:
                    goals_context += f"- Goal: {goal['title']} ({goal['status']}) [ID: {goal.get('goal_id','N/A')}]\n"
                    for task in goal["tasks"]:
                        goals_context += f"  - Task: {task['title']} ({task['status']}) [ID: {task.get('task_id','N/A')}]\n"

            uploaded_files = await uploads_collection.distinct(
                "filename", {"session_id": session_id}
            )
            mentioned_filenames = [
                fn for fn in uploaded_files if fn.lower() in user_message.lower()
            ]
            hooked_filenames = filenames if filenames else mentioned_filenames

            external_content = ""
            url_match = re.search(r"https?://[^\s]+", user_message)
            if url_match:
                url = url_match.group(0)
                logging.info(f"Detected URL in prompt: {url}")
                if "youtube.com" in url or "youtu.be" in url:
                    external_content = await query_internet_via_groq(
                        f"Summarize the content of the YouTube video at {url}"
                    )
                    external_content = await detailed_explanation(external_content)
                else:
                    external_content = await query_internet_via_groq(
                        f"Summarize the content of the webpage at {url}"
                    )
                    external_content = await content_for_website(external_content)

            multimodal_context, used_filenames = await retrieve_multimodal_context(
                user_message, session_id, hooked_filenames
            )
            unified_prompt = f"User Query: {user_message}\n"
            if external_content:
                unified_prompt += f"\n[External Content]:\n{external_content}\n"
            if multimodal_context:
                unified_prompt += (
                    f"\n[Retrieved File & Code Context]:\n{multimodal_context}\n"
                )
            unified_prompt += f"\nCurrent Date/Time: {current_date}\n\nProvide a conversational, friendly response as if speaking directly to the user."

            research_needed = await classify_prompt(user_message)
            if research_needed == "research" and not multimodal_context:
                await websocket.send_json(
                    {"status": "researching", "message": "Researching the topic..."}
                )
                research_results = await browse_and_generate(user_message)
                if research_results:
                    unified_prompt += f"\n\n[Additional Research]:\n{research_results}"

            chat_entry = await chats_collection.find_one(
                {"user_id": user_id, "session_id": session_id}
            )
            chat_history = []
            if chat_entry:
                past_messages = chat_entry.get("messages", [])
                if past_messages:
                    past_embeddings = np.array(
                        [
                            msg["embedding"]
                            for msg in past_messages
                            if "embedding" in msg
                        ]
                    )
                    if past_embeddings.size > 0:
                        current_embedding = await generate_text_embedding(user_message)
                        distances = np.linalg.norm(
                            past_embeddings - current_embedding, axis=1
                        )
                        n = len(past_messages)
                        ages = np.array([n - 1 - i for i in range(n)])
                        lambda_val = 0.05
                        modified_distances = distances + lambda_val * ages
                        k = 3
                        top_k_indices = np.argsort(modified_distances)[:k]
                        m = 2
                        last_m_indices = (
                            list(range(n - m, n)) if n >= m else list(range(n))
                        )
                        combined_indices = list(
                            set(top_k_indices.tolist() + last_m_indices)
                        )
                        sorted_indices = sorted(combined_indices)
                        chat_history = [past_messages[i] for i in sorted_indices]
                    else:
                        chat_history = (
                            filter_think_messages(past_messages[-2:])
                            if len(past_messages) >= 2
                            else filter_think_messages(past_messages)
                        )

            long_term_memory = ""
            mem_entry = await memory_collection.find_one({"user_id": user_id})
            if mem_entry and "summary" in mem_entry:
                long_term_memory = mem_entry["summary"]

            system_prompt = (
                "You are Stelle, a friendly, empathetic AI companion designed for natural, one-on-one voice conversations. "
                "Respond in a casual, engaging manner as if speaking directly to the user, avoiding formal text-like responses.,dont give response in markdowns make it like you and user doing conversation"
                "Use a warm tone, show interest in the user's input, and adapt to their context. "
                "If you have to add tasks to a goal, beforehand make the task id then add it to the goal. "
                "When the user sets a new goal, use '[GOAL_SET: <goal_title>]' Must use '[TASK: <task_desc>]' lines. for adding tasks. "
                "To delete a goal: '[GOAL_DELETE: <goal_id>]'. To delete a task: '[TASK_DELETE: <task_id>]'. "
                "To add a new task: '[TASK_ADD: <goal_id>: <task_description>]'. "
                "To modify a task's title: '[TASK_MODIFY: <task_id>: <new_title_or_description>]'. "
                "To start a goal: '[GOAL_START: <goal_id>]'. To start a task: '[TASK_START: <task_id>]'. "
                "To complete a goal: '[GOAL_COMPLETE: <goal_id>]'. To complete a task: '[TASK_COMPLETE: <task_id>]'. "
                "Must ask user for deadlines using '[TASK_DEADLINE: <task_id>: <YYYY-MM-DD HH:MM>]' and log progress using '[TASK_PROGRESS: <task_id>: <progress_description>]'.\n"
                f"Current date/time: {current_date}\n"
            )

            messages = [{"role": "system", "content": system_prompt}]
            if long_term_memory:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Long-term memory: {long_term_memory}",
                    }
                )
            if goals_context:
                messages.append({"role": "system", "content": goals_context})
            for msg in chat_history:
                cleaned = re.sub(
                    r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL
                ).strip()
                if cleaned:
                    if len(cleaned) > 800:
                        cleaned = cleaned[:800] + "…"
                    if cleaned:
                        messages.append({"role": msg["role"], "content": cleaned})

            messages.append({"role": "user", "content": unified_prompt})

            generate_api_keys = [
                os.getenv("GROQ_API_KEY_GENERATE_1"),
                os.getenv("GROQ_API_KEY_GENERATE_2"),
                os.getenv("GROQ_API_KEY_GENERATE_3"),
            ]
            generate_api_keys = [k for k in generate_api_keys if k]
            if not generate_api_keys:
                await websocket.send_json(
                    {"error": "No valid GROQ_API_KEY_GENERATE found"}
                )
                continue
            selected_key = random.choice(generate_api_keys)
            client_generate = AsyncGroq(api_key=selected_key)

            stream = await client_generate.chat.completions.create(
                messages=messages,
                model="deepseek-r1-distill-llama-70b",
                max_completion_tokens=4000,
                temperature=0.7,
                stream=True,
                reasoning_format="hidden",
            )

            full_reply = ""
            async for chunk in stream:
                delta = (
                    chunk.choices[0].delta.content
                    if chunk.choices[0].delta.content
                    else ""
                )
                full_reply += delta
                await websocket.send_json({"status": "streaming", "message": delta})

            reply_content = full_reply.strip()
            new_goals_map = {}
            goal_set_matches = re.findall(r"\[GOAL_SET: (.*?)\]", reply_content)
            for goal_phrase in goal_set_matches:
                goal_id = str(uuid.uuid4())
                new_goals_map[goal_phrase] = goal_id
                existing_goal = await goals_collection.find_one(
                    {
                        "user_id": user_id,
                        "title": goal_phrase,
                        "status": {"$in": ["active", "in progress"]},
                    }
                )
                if existing_goal:
                    logging.info(
                        f"Skipping creation of duplicate goal '{goal_phrase}' for user {user_id}."
                    )
                    continue
                new_goal = {
                    "user_id": user_id,
                    "goal_id": goal_id,
                    "title": goal_phrase,
                    "session_id": session_id,
                    "description": "",
                    "status": "active",
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                    "tasks": [],
                }
                task_matches = re.findall(r"\[TASK: (.*?)\]", reply_content)
                for task_desc in task_matches:
                    task_id = str(uuid.uuid4())
                    new_task = {
                        "task_id": task_id,
                        "title": task_desc,
                        "description": "",
                        "status": "not started",
                        "created_at": datetime.datetime.now(datetime.timezone.utc),
                        "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        "deadline": None,
                        "progress": [],
                    }
                    new_goal["tasks"].append(new_task)
                await goals_collection.insert_one(new_goal)
                logging.info(
                    f"Goal set: '{goal_phrase}' with {len(task_matches)} tasks for user {user_id}"
                )

            goal_delete_matches = re.findall(r"\[GOAL_DELETE: (.*?)\]", reply_content)
            for gid in goal_delete_matches:
                real_goal_id = new_goals_map.get(gid, gid)
                result = await goals_collection.delete_one(
                    {"user_id": user_id, "goal_id": real_goal_id}
                )
                if result.deleted_count > 0:
                    logging.info(f"Goal {real_goal_id} deleted for user {user_id}")
                else:
                    logging.warning(
                        f"Goal {real_goal_id} not found or could not be deleted."
                    )

            task_delete_matches = re.findall(r"\[TASK_DELETE: (.*?)\]", reply_content)
            for tid in task_delete_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {"$pull": {"tasks": {"task_id": tid}}},
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} deleted for user {user_id}")
                else:
                    logging.warning(f"Task {tid} not found or could not be deleted.")

            task_add_matches = re.findall(
                r"\[TASK_ADD:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for goal_id_str, task_desc in task_add_matches:
                real_goal_id = new_goals_map.get(goal_id_str, goal_id_str)
                task_id = str(uuid.uuid4())
                new_task = {
                    "task_id": task_id,
                    "title": task_desc,
                    "description": "",
                    "status": "not started",
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                    "deadline": None,
                    "progress": [],
                }
                result = await goals_collection.update_one(
                    {"user_id": user_id, "goal_id": real_goal_id},
                    {
                        "$push": {"tasks": new_task},
                        "$set": {
                            "updated_at": datetime.datetime.now(datetime.timezone.utc)
                        },
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Added task '{task_desc}' to goal {real_goal_id}")
                else:
                    logging.warning(
                        f"Could not add task to goal {real_goal_id} (not found?)."
                    )

            task_modify_matches = re.findall(
                r"\[TASK_MODIFY:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for tid, new_desc in task_modify_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.title": new_desc,
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} modified to '{new_desc}'")
                else:
                    logging.warning(f"Task {tid} not found for modification.")

            goal_start_matches = re.findall(r"\[GOAL_START: (.*?)\]", reply_content)
            for gid in goal_start_matches:
                real_goal_id = new_goals_map.get(gid, gid)
                result = await goals_collection.update_one(
                    {"user_id": user_id, "goal_id": real_goal_id},
                    {
                        "$set": {
                            "status": "in progress",
                            "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Goal {real_goal_id} started (in progress).")
                else:
                    logging.warning(f"Goal {real_goal_id} not found for GOAL_START.")

            task_start_matches = re.findall(r"\[TASK_START: (.*?)\]", reply_content)
            for tid in task_start_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.status": "in progress",
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} started (in progress).")
                else:
                    logging.warning(f"Task {tid} not found for TASK_START.")

            goal_complete_matches = re.findall(
                r"\[GOAL_COMPLETE: (.*?)\]", reply_content
            )
            for gid in goal_complete_matches:
                real_goal_id = new_goals_map.get(gid, gid)
                result = await goals_collection.update_one(
                    {"user_id": user_id, "goal_id": real_goal_id},
                    {
                        "$set": {
                            "status": "completed",
                            "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Goal {real_goal_id} marked as completed.")
                else:
                    logging.warning(f"Goal {real_goal_id} not found for GOAL_COMPLETE.")

            task_complete_matches = re.findall(
                r"\[TASK_COMPLETE: (.*?)\]", reply_content
            )
            for tid in task_complete_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": "tid"},
                    {
                        "$set": {
                            "tasks.$.status": "completed",
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} marked as completed.")
                else:
                    logging.warning(f"Task {tid} not found for completion.")

            task_deadline_matches = re.findall(
                r"\[TASK_DEADLINE:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for tid, deadline_str in task_deadline_matches:
                try:
                    deadline_dt = datetime.datetime.strptime(
                        deadline_str, "%Y-%m-%d %H:%M"
                    )
                except Exception as ex:
                    logging.error(
                        f"Invalid deadline format for task {tid}: {deadline_str}"
                    )
                    continue
                user_info = await users_collection.find_one({"user_id": user_id})
                tz_name = user_info.get("time_zone", "UTC") if user_info else "UTC"
                try:
                    user_tz = pytz.timezone(tz_name)
                except Exception:
                    user_tz = pytz.UTC
                localized_deadline = user_tz.localize(deadline_dt)
                reminder_time = localized_deadline - datetime.timedelta(days=1)
                reminder_time_utc = reminder_time.astimezone(pytz.UTC)
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.deadline": deadline_str,
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Set deadline for Task {tid} to {deadline_str}")
                    await schedule_notification(
                        user_id,
                        f"Reminder: Task {tid} is due on {deadline_str}",
                        reminder_time_utc,
                        notif_type="deadline_reminder",
                    )
                else:
                    logging.warning(f"Task {tid} not found for deadline update.")

            task_progress_matches = re.findall(
                r"\[TASK_PROGRESS:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for tid, progress_desc in task_progress_matches:
                progress_entry = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "description": progress_desc,
                }
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$push": {"tasks.$.progress": progress_entry},
                        "$set": {
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            )
                        },
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Added progress entry to Task {tid}: {progress_desc}")
                else:
                    logging.warning(f"Task {tid} not found for progress update.")

            remind_match = re.search(r"remind me (.+)", user_message, re.IGNORECASE)
            if remind_match:
                reminder_text = remind_match.group(1).strip()
                await schedule_immediate_reminder(user_id, reminder_text)

            lines = reply_content.split("\n")
            clean_lines = [
                line for line in lines if not re.match(r"\[.*?: .*?\]", line.strip())
            ]
            reply_content_clean = "\n".join(clean_lines).strip()

            user_embedding = await generate_text_embedding(user_message)
            assistant_embedding = await generate_text_embedding(reply_content_clean)
            new_messages = [
                {"role": "user", "content": user_message, "embedding": user_embedding},
                {
                    "role": "assistant",
                    "content": reply_content_clean,
                    "embedding": assistant_embedding,
                },
            ]
            if chat_entry:
                await chats_collection.update_one(
                    {"user_id": user_id, "session_id": session_id},
                    {
                        "$push": {"messages": {"$each": new_messages}},
                        "$set": {
                            "last_updated": datetime.datetime.now(datetime.timezone.utc)
                        },
                    },
                )
            else:
                await chats_collection.insert_one(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "messages": new_messages,
                        "last_updated": datetime.datetime.now(datetime.timezone.utc),
                    }
                )
            if chat_entry and len(chat_entry.get("messages", [])) >= 10:
                await store_long_term_memory(
                    user_id, session_id, chat_entry["messages"][-10:]
                )

            await websocket.send_json(
                {"status": "complete", "message": reply_content_clean}
            )

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from /nlp")
    except Exception as e:
        logger.error(f"Error in /nlp endpoint: {e}")
        await websocket.send_json({"error": "Internal error processing your request"})


@app.post("/regenerate", response_model=GenerateResponse)
async def regenerate_response(
    request: RegenerateRequest, background_tasks: BackgroundTasks
):
    try:
        user_id = request.user_id
        session_id = request.session_id
        filenames = request.filenames

        # Retrieve chat entry
        chat_entry = await chats_collection.find_one(
            {"user_id": user_id, "session_id": session_id}
        )
        if not chat_entry or not chat_entry.get("messages"):
            raise HTTPException(
                status_code=400, detail="No chat history found for regeneration."
            )

        # Find the last user message
        messages = chat_entry["messages"]
        last_user_message = next(
            (msg for msg in reversed(messages) if msg["role"] == "user"), None
        )
        if not last_user_message:
            raise HTTPException(
                status_code=400,
                detail="No user message found to regenerate response for.",
            )

        prompt = last_user_message["content"]

        # Proceed with response generation similar to /generate
        current_date = get_current_datetime()

        active_goals = await goals_collection.find(
            {"user_id": user_id, "status": {"$in": ["active", "in progress"]}}
        ).to_list(None)
        goals_context = ""
        if active_goals:
            goals_context = "User's current goals and tasks:\n"
            for goal in active_goals:
                goals_context += f"- Goal: {goal['title']} ({goal['status']}) [ID: {goal.get('goal_id','N/A')}]\n"
                for task in goal["tasks"]:
                    goals_context += f"  - Task: {task['title']} ({task['status']}) [ID: {task.get('task_id','N/A')}]\n"

        uploaded_files = await uploads_collection.distinct(
            "filename", {"session_id": session_id}
        )
        mentioned_filenames = [
            fn for fn in uploaded_files if fn.lower() in prompt.lower()
        ]
        hooked_filenames = filenames if filenames else mentioned_filenames
        logger.info(f"Hooked filenames for regenerate: {hooked_filenames}")

        external_content = ""
        url_match = re.search(r"https?://[^\s]+", prompt)
        if url_match:
            url = url_match.group(0)
            logging.info(f"Detected URL in prompt for regenerate: {url}")
            if "youtube.com" in url or "youtu.be" in url:
                external_content = await query_internet_via_groq(
                    f"Summarize the content of the YouTube video at {url}"
                )
                external_content = await detailed_explanation(external_content)
            else:
                external_content = await query_internet_via_groq(
                    f"Summarize the content of the webpage at {url}"
                )
                external_content = await content_for_website(external_content)

        multimodal_context, used_filenames = await retrieve_multimodal_context(
            prompt, session_id, hooked_filenames
        )
        unified_prompt = f"User Query: {prompt}\n"
        if external_content:
            unified_prompt += f"\n[External Content]:\n{external_content}\n"
        if multimodal_context:
            unified_prompt += (
                f"\n[Retrieved File & Code Context]:\n{multimodal_context}\n"
            )
        unified_prompt += f"\nCurrent Date/Time: {current_date}\n\nProvide a detailed and context-aware response."

        research_needed = await classify_prompt(prompt)
        if research_needed == "research" and not multimodal_context:
            research_results = await browse_and_generate(prompt)
            if research_results:
                unified_prompt += f"\n\n[Additional Research]:\n{research_results}"

        # Select chat history
        past_messages = chat_entry.get("messages", [])
        if past_messages:
            past_embeddings = np.array(
                [msg["embedding"] for msg in past_messages if "embedding" in msg]
            )
            if past_embeddings.size > 0:
                current_embedding = await generate_text_embedding(prompt)
                distances = np.linalg.norm(past_embeddings - current_embedding, axis=1)
                n = len(past_messages)
                ages = np.array([n - 1 - i for i in range(n)])
                lambda_val = 0.05
                modified_distances = distances + lambda_val * ages
                k = 3
                top_k_indices = np.argsort(modified_distances)[:k]
                m = 2
                last_m_indices = list(range(n - m, n)) if n >= m else list(range(n))
                combined_indices = list(set(top_k_indices.tolist() + last_m_indices))
                sorted_indices = sorted(combined_indices)
                chat_history = [past_messages[i] for i in sorted_indices]
            else:
                chat_history = (
                    filter_think_messages(past_messages[-2:])
                    if len(past_messages) >= 2
                    else filter_think_messages(past_messages)
                )
        else:
            chat_history = []

        long_term_memory = ""
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        if mem_entry and "summary" in mem_entry:
            long_term_memory = mem_entry["summary"]

        system_prompt = (
            "You are Stelle, a strategic, empathetic AI assistant with autonomous goal/task management. "
            "If you have to add tasks to a goal, beforehand make the task id then add it to the goal. "
            "When the user sets a new goal, use '[GOAL_SET: <goal_title>]' Must use '[TASK: <task_desc>]' lines. for adding tasks. "
            "To delete a goal: '[GOAL_DELETE: <goal_id>]'. To delete a task: '[TASK_DELETE: <task_id>]'. "
            "To add a new task: '[TASK_ADD: <goal_id>: <task_description>]'. "
            "To modify a task's title: '[TASK_MODIFY: <task_id>: <new_title_or_description>]'. "
            "To start a goal: '[GOAL_START: <goal_id>]'. To start a task: '[TASK_START: <task_id>]'. "
            "To complete a goal: '[GOAL_COMPLETE: <goal_id>]'. To complete a task: '[TASK_COMPLETE: <task_id>]'. "
            "Must ask user for deadlines using '[TASK_DEADLINE: <task_id>: <YYYY-MM-DD HH:MM>]' and log progress using '[TASK_PROGRESS: <task_id>: <progress_description>]'.\n"
            f"Current date/time: {current_date}\n"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if long_term_memory:
            messages.append(
                {"role": "system", "content": f"Long-term memory: {long_term_memory}"}
            )
        if goals_context:
            messages.append({"role": "system", "content": goals_context})
        for msg in chat_history:
            cleaned = re.sub(
                r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL
            ).strip()
            if cleaned:
                if len(cleaned) > 800:
                    cleaned = cleaned[:800] + "…"
                if cleaned:
                    messages.append({"role": msg["role"], "content": cleaned})
        messages.append({"role": "user", "content": unified_prompt})

        generate_api_keys = [
            os.getenv("GROQ_API_KEY_GENERATE_1"),
            os.getenv("GROQ_API_KEY_GENERATE_2"),
            os.getenv("GROQ_API_KEY_GENERATE_3"),
        ]
        generate_api_keys = [k for k in generate_api_keys if k]
        if not generate_api_keys:
            raise HTTPException(
                status_code=500,
                detail="No valid GROQ_API_KEY_GENERATE environment variables found.",
            )
        selected_key = random.choice(generate_api_keys)
        client_generate = AsyncGroq(api_key=selected_key)

        stream = await client_generate.chat.completions.create(
            messages=messages,
            model="deepseek-r1-distill-llama-70b",
            max_completion_tokens=4000,
            temperature=0.7,
            stream=True,
        )

        async def generate_stream():
            full_reply = ""
            async for chunk in stream:
                delta = (
                    chunk.choices[0].delta.content
                    if chunk.choices[0].delta.content
                    else ""
                )
                full_reply += delta
                yield delta
            reply_content = full_reply.strip()

            # Process goals, tasks, etc.
            new_goals_map = {}
            goal_set_matches = re.findall(r"\[GOAL_SET: (.*?)\]", reply_content)
            for goal_phrase in goal_set_matches:
                goal_id = str(uuid.uuid4())
                new_goals_map[goal_phrase] = goal_id
                existing_goal = await goals_collection.find_one(
                    {
                        "user_id": user_id,
                        "title": goal_phrase,
                        "status": {"$in": ["active", "in progress"]},
                    }
                )
                if existing_goal:
                    logging.info(
                        f"Skipping creation of duplicate goal '{goal_phrase}' for user {user_id}."
                    )
                    continue
                new_goal = {
                    "user_id": user_id,
                    "goal_id": goal_id,
                    "session_id": session_id,
                    "title": goal_phrase,
                    "description": "",
                    "status": "active",
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                    "tasks": [],
                }
                task_matches = re.findall(r"\[TASK: (.*?)\]", reply_content)
                for task_desc in task_matches:
                    task_id = str(uuid.uuid4())
                    new_task = {
                        "task_id": task_id,
                        "title": task_desc,
                        "description": "",
                        "status": "not started",
                        "created_at": datetime.datetime.now(datetime.timezone.utc),
                        "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        "deadline": None,
                        "progress": [],
                    }
                    new_goal["tasks"].append(new_task)
                await goals_collection.insert_one(new_goal)
                logging.info(
                    f"Goal set: '{goal_phrase}' with {len(task_matches)} tasks for user {user_id}"
                )

            goal_delete_matches = re.findall(r"\[GOAL_DELETE: (.*?)\]", reply_content)
            for gid in goal_delete_matches:
                real_goal_id = new_goals_map.get(gid, gid)
                result = await goals_collection.delete_one(
                    {"user_id": user_id, "goal_id": real_goal_id}
                )
                if result.deleted_count > 0:
                    logging.info(f"Goal {real_goal_id} deleted for user {user_id}")
                else:
                    logging.warning(
                        f"Goal {real_goal_id} not found or could not be deleted."
                    )

            task_delete_matches = re.findall(r"\[TASK_DELETE: (.*?)\]", reply_content)
            for tid in task_delete_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {"$pull": {"tasks": {"task_id": tid}}},
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} deleted for user {user_id}")
                else:
                    logging.warning(f"Task {tid} not found or could not be deleted.")

            task_add_matches = re.findall(
                r"\[TASK_ADD:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for goal_id_str, task_desc in task_add_matches:
                real_goal_id = new_goals_map.get(goal_id_str, goal_id_str)
                task_id = str(uuid.uuid4())
                new_task = {
                    "task_id": task_id,
                    "title": task_desc,
                    "description": "",
                    "status": "not started",
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                    "deadline": None,
                    "progress": [],
                }
                result = await goals_collection.update_one(
                    {"user_id": user_id, "goal_id": real_goal_id},
                    {
                        "$push": {"tasks": new_task},
                        "$set": {
                            "updated_at": datetime.datetime.now(datetime.timezone.utc)
                        },
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Added task '{task_desc}' to goal {real_goal_id}")
                else:
                    logging.warning(
                        f"Could not add task to goal {real_goal_id} (not found?)."
                    )

            task_modify_matches = re.findall(
                r"\[TASK_MODIFY:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for tid, new_desc in task_modify_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.title": new_desc,
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} modified to '{new_desc}'")
                else:
                    logging.warning(f"Task {tid} not found for modification.")

            goal_start_matches = re.findall(r"\[GOAL_START: (.*?)\]", reply_content)
            for gid in goal_start_matches:
                real_goal_id = new_goals_map.get(gid, gid)
                result = await goals_collection.update_one(
                    {"user_id": user_id, "goal_id": real_goal_id},
                    {
                        "$set": {
                            "status": "in progress",
                            "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Goal {real_goal_id} started (in progress).")
                else:
                    logging.warning(f"Goal {real_goal_id} not found for GOAL_START.")

            task_start_matches = re.findall(r"\[TASK_START: (.*?)\]", reply_content)
            for tid in task_start_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.status": "in progress",
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} started (in progress).")
                else:
                    logging.warning(f"Task {tid} not found for TASK_START.")

            goal_complete_matches = re.findall(
                r"\[GOAL_COMPLETE: (.*?)\]", reply_content
            )
            for gid in goal_complete_matches:
                real_goal_id = new_goals_map.get(gid, gid)
                result = await goals_collection.update_one(
                    {"user_id": user_id, "goal_id": real_goal_id},
                    {
                        "$set": {
                            "status": "completed",
                            "updated_at": datetime.datetime.now(datetime.timezone.utc),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Goal {real_goal_id} marked as completed.")
                else:
                    logging.warning(f"Goal {real_goal_id} not found for GOAL_COMPLETE.")

            task_complete_matches = re.findall(
                r"\[TASK_COMPLETE: (.*?)\]", reply_content
            )
            for tid in task_complete_matches:
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.status": "completed",
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Task {tid} marked as completed.")
                else:
                    logging.warning(f"Task {tid} not found for completion.")

            task_deadline_matches = re.findall(
                r"\[TASK_DEADLINE:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for tid, deadline_str in task_deadline_matches:
                try:
                    deadline_dt = datetime.datetime.strptime(
                        deadline_str, "%Y-%m-%d %H:%M"
                    )
                except Exception as ex:
                    logging.error(
                        f"Invalid deadline format for task {tid}: {deadline_str}"
                    )
                    continue
                user_info = await users_collection.find_one({"user_id": user_id})
                tz_name = user_info.get("time_zone", "UTC") if user_info else "UTC"
                try:
                    user_tz = pytz.timezone(tz_name)
                except Exception:
                    user_tz = pytz.UTC
                localized_deadline = user_tz.localize(deadline_dt)
                reminder_time = localized_deadline - datetime.timedelta(days=1)
                reminder_time_utc = reminder_time.astimezone(pytz.UTC)
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$set": {
                            "tasks.$.deadline": deadline_str,
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            ),
                        }
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Set deadline for Task {tid} to {deadline_str}")
                    await schedule_notification(
                        user_id,
                        f"Reminder: Task {tid} is due on {deadline_str}",
                        reminder_time_utc,
                        notif_type="deadline_reminder",
                    )
                else:
                    logging.warning(f"Task {tid} not found for deadline update.")

            task_progress_matches = re.findall(
                r"\[TASK_PROGRESS:\s*(.*?):\s*(.*?)\]", reply_content
            )
            for tid, progress_desc in task_progress_matches:
                progress_entry = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                    "description": progress_desc,
                }
                result = await goals_collection.update_one(
                    {"user_id": user_id, "tasks.task_id": tid},
                    {
                        "$push": {"tasks.$.progress": progress_entry},
                        "$set": {
                            "tasks.$.updated_at": datetime.datetime.now(
                                datetime.timezone.utc
                            )
                        },
                    },
                )
                if result.modified_count > 0:
                    logging.info(f"Added progress entry to Task {tid}: {progress_desc}")
                else:
                    logging.warning(f"Task {tid} not found for progress update.")

            # Clean the response
            lines = reply_content.split("\n")
            clean_lines = [
                line for line in lines if not re.match(r"\[.*?: .*?\]", line.strip())
            ]
            reply_content_clean = "\n".join(clean_lines).strip()

            # Generate embedding
            assistant_embedding = await generate_text_embedding(reply_content_clean)

            # Append only the new assistant message
            new_assistant_message = {
                "role": "assistant",
                "content": reply_content_clean,
                "embedding": assistant_embedding,
            }
            await chats_collection.update_one(
                {"user_id": user_id, "session_id": session_id},
                {
                    "$push": {"messages": new_assistant_message},
                    "$set": {
                        "last_updated": datetime.datetime.now(datetime.timezone.utc)
                    },
                },
            )

            # Handle background tasks
            if len(messages) >= 10:
                background_tasks.add_task(
                    store_long_term_memory,
                    user_id,
                    session_id,
                    chat_entry["messages"][-10:],
                )

            # Update query counts for used filenames
            for filename in used_filenames:
                await uploads_collection.update_many(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "filename": filename,
                    },
                    {"$inc": {"query_count": 1}},
                )

            # Remove old uploads if necessary
            cursor = uploads_collection.find(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "query_count": {"$gte": 15},
                }
            )
            documents_to_remove = set()
            async for chunk in cursor:
                documents_to_remove.add(chunk["filename"])
            for filename in documents_to_remove:
                await uploads_collection.delete_many(
                    {"user_id": user_id, "session_id": session_id, "filename": filename}
                )
                indices_to_remove = [
                    idx
                    for idx, m in file_doc_memory_map.items()
                    if m["filename"] == filename and m["session_id"] == session_id
                ]
                if indices_to_remove:
                    doc_index.remove_ids(np.array(indices_to_remove, dtype="int64"))
                    for idx in indices_to_remove:
                        del file_doc_memory_map[idx]
                code_indices_to_remove = [
                    idx
                    for idx, m in code_memory_map.items()
                    if m["filename"] == filename and m["session_id"] == session_id
                ]
                if code_indices_to_remove:
                    code_index.remove_ids(
                        np.array(code_indices_to_remove, dtype="int64")
                    )
                    for idx in code_indices_to_remove:
                        del code_memory_map[idx]

        return StreamingResponse(generate_stream(), media_type="text/plain")

    except Exception as e:
        logging.error(f"Error in /regenerate endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal error processing your request."
        )

@app.post("/Plan_my_week")
async def plan_my_week(request: PlanWeekRequest):
    user_id = request.user_id
    try:
        # Step 1: Get user's timezone
        user_info = await users_collection.find_one({"user_id": user_id})
        tz_name = user_info.get("time_zone", "UTC") if user_info else "UTC"
        try:
            user_tz = pytz.timezone(tz_name)
        except Exception:
            user_tz = pytz.UTC

        now_local = datetime.datetime.now(user_tz)
        today = now_local.date()

        # Step 2: Fetch active and in-progress goals
        goals_cursor = goals_collection.find({
            "user_id": user_id,
            "status": {"$in": ["active", "in progress"]}
        })

        all_goals_data = []
        async for goal in goals_cursor:
            all_goals_data.append(goal)

        if not all_goals_data:
            raise HTTPException(status_code=404, detail="No active goals found to plan for.")

        # Step 3: Prepare goals and tasks context
        goals_summary = []
        for goal in all_goals_data:
            tasks_summary = []
            for task in goal.get('tasks', []):
                task_status = task.get('status', 'not started')
                task_deadline_str = task.get('deadline')
                deadline_status_info = ""
                if task_deadline_str and task_status not in ['completed', 'cancelled']:
                    try:
                        deadline_dt = datetime.datetime.fromisoformat(task_deadline_str.split(' ')[0]).date()
                        if deadline_dt < today:
                            task_status = "Deadline Exceeded"
                            deadline_status_info = f"(Deadline: {task_deadline_str} - EXCEEDED)"
                        else:
                            deadline_status_info = f"(Deadline: {task_deadline_str})"
                    except (ValueError, TypeError):
                        deadline_status_info = f"(Deadline: {task_deadline_str})"
                
                tasks_summary.append(
                    f"    - Task: {task.get('title')} (ID: {task.get('task_id')}) "
                    f"(Status: {task_status}) {deadline_status_info}"
                )
        
            goals_summary.append(
                f"Goal: {goal.get('title')} (ID: {goal.get('goal_id')})\n"
                f"  Description: {goal.get('description', 'N/A')}\n"
                f"  Tasks:\n" + "\n".join(tasks_summary)
            )

        full_context = "\n\n".join(goals_summary)
        
        # Step 4: Determine the current week
        start_of_week = today - datetime.timedelta(days=today.weekday())
        end_of_week = start_of_week + datetime.timedelta(days=6)
        
        # Step 5: Construct the AI prompt
        prompt = f"""
You are an expert project manager and personal coach. Your task is to create a comprehensive and strategic weekly plan for a user based on their goals.

Current Date: {today.strftime('%Y-%m-%d')}
Current Calendar Week: {start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')}

Here is the complete context of the user's current goals and tasks:
---
{full_context}
---

Please generate a plan with the following structure in a single, valid JSON object:
{{
  "strategic_approach": "A high-level overview for the upcoming weeks until the final goal deadline.",
  "this_week_plan": [
    {{
      "date": "YYYY-MM-DD",
      "day_of_week": "Monday",
      "daily_focus": "Main objective for the day.",
      "tasks": [
        {{
          "task_id": "The original task ID.",
          "title": "The original task title.",
          "status": "The current status of the task.",
          "description": "Detailed, actionable guide on HOW to complete the task.",
          "sub_tasks": [
            {{
              "title": "A smaller, actionable sub-task.",
              "description": "Detailed 'how-to' guide for this sub-task."
            }}
          ]
        }}
      ]
    }}
  ]
}}

**CRITICAL RULES**:
1. Prioritize Overdue Tasks: Any task marked "Deadline Exceeded" MUST be scheduled for the first day of the plan.
2. Detailed 'How-To' Descriptions: Provide specific, step-by-step guidance for tasks and sub-tasks.
3. Logical Sequencing: Schedule tasks based on dependencies and logical workflow.
4. Full Week Plan: Generate a plan for all 7 days (Monday to Sunday). Use empty lists for days with no tasks.
5. Output ONLY JSON: Return a single, valid JSON object.

Now, create the weekly plan.
"""

        # Step 6: Call the AI
        try:
            async_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY_PLANNING"))
            response = await async_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert AI project manager. Output a single, valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            plan_data_str = response.choices[0].message.content
            plan_data = json.loads(plan_data_str)

            # Step 7: Save the plan to MongoDB
            existing_plan = await weekly_plans_collection.find_one({
                "user_id": user_id,
                "week_start_date": start_of_week.strftime("%Y-%m-%d")
            })

            if existing_plan:
                # Update existing plan
                await weekly_plans_collection.update_one(
                    {
                        "user_id": user_id,
                        "week_start_date": start_of_week.strftime("%Y-%m-%d")
                    },
                    {
                        "$set": {
                            "plan": plan_data,
                            "updated_at": datetime.datetime.now(pytz.UTC)
                        }
                    }
                )
            else:
                # Insert new plan
                await weekly_plans_collection.insert_one(
                    {
                        "user_id": user_id,
                        "week_start_date": start_of_week.strftime("%Y-%m-%d"),
                        "plan": plan_data,
                        "created_at": datetime.datetime.now(pytz.UTC),
                        "updated_at": datetime.datetime.now(pytz.UTC)
                    }
                )

            return plan_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {str(e)}\nResponse: {plan_data_str}")
            raise HTTPException(status_code=500, detail="Failed to generate a valid plan.")
        except Exception as e:
            logger.error(f"Error in /Plan_my_week endpoint: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    except Exception as e:
        logger.error(f"Error in /Plan_my_week endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
    
# # New Plan my weeks 
# @app.post("/Plan_my_week", response_model=MultiWeekPlanResponse)
# async def plan_my_weeks(request: PlanWeekRequest):
#     user_id = request.user_id
#     planning_horizon_weeks = request.planning_horizon_weeks

#     try:
#         # Step 1: Get user's timezone using PyMongo
#         user_info = users_collection.find_one({"user_id": user_id})
#         tz_name = user_info.get("time_zone", "UTC") if user_info else "UTC"
#         try:
#             user_tz = pytz.timezone(tz_name)
#         except Exception:
#             user_tz = pytz.UTC

#         now_local = datetime.datetime.now(user_tz)
#         today = now_local.date()

#         # Step 2: Fetch active and in-progress goals using PyMongo
#         all_goals_data = list(
#             goals_collection.find(
#                 {"user_id": user_id, "status": {"$in": ["active", "in progress"]}}
#             )
#         )

#         if not all_goals_data:
#             raise HTTPException(
#                 status_code=404, detail="No active goals found to plan for."
#             )

#         # Step 3: Calculate planning horizon if not specified
#         if planning_horizon_weeks is None:
#             planning_horizon_weeks = calculate_planning_horizon(all_goals_data)

#         # Step 4: Prepare goals and tasks context
#         goals_summary = []
#         for goal in all_goals_data:
#             tasks_summary = []
#             for task in goal.get("tasks", []):
#                 task_status = task.get("status", "not started")
#                 task_deadline_str = task.get("deadline")
#                 deadline_status_info = ""
#                 if task_deadline_str and task_status not in ["completed", "cancelled"]:
#                     try:
#                         deadline_dt = datetime.datetime.fromisoformat(
#                             task_deadline_str.split(" ")[0]
#                         ).date()
#                         if deadline_dt < today:
#                             task_status = "Deadline Exceeded"
#                             deadline_status_info = (
#                                 f"(Deadline: {task_deadline_str} - EXCEEDED)"
#                             )
#                         else:
#                             deadline_status_info = f"(Deadline: {task_deadline_str})"
#                     except (ValueError, TypeError):
#                         deadline_status_info = f"(Deadline: {task_deadline_str})"

#                 tasks_summary.append(
#                     f"    - Task: {task.get('title', 'Untitled')} (ID: {task.get('task_id', 'N/A')}) "
#                     f"(Status: {task_status}) {deadline_status_info}"
#                 )

#             goals_summary.append(
#                 f"Goal: {goal.get('title', 'Untitled')} (ID: {goal.get('goal_id', 'N/A')})\n"
#                 f"  Description: {goal.get('description', '')}\n"
#                 f"  Tasks:\n" + "\n".join(tasks_summary)
#             )

#         full_context = "\n\n".join(goals_summary)

#         # Step 5: Determine the current week and planning horizon
#         start_of_current_week = today - datetime.timedelta(days=today.weekday())

#         # Step 6: Construct the AI prompt for multi-week planning
#         prompt = f"""
# You are an expert project manager and personal coach. Your task is to create a comprehensive multi-week strategic plan for a user based on their goals.

# Current Date: {today.strftime('%Y-%m-%d')}
# Planning Horizon: {planning_horizon_weeks} weeks
# Current Calendar Week: {start_of_current_week.strftime('%Y-%m-%d')} to {(start_of_current_week + datetime.timedelta(days=6)).strftime('%Y-%m-%d')}

# Here is the complete context of the user's current goals and tasks:
# ---
# {full_context}
# ---

# Please generate a {planning_horizon_weeks}-week plan with the following structure in a single, valid JSON object:
# {{
#   "overall_strategic_approach": "A high-level overview for the entire planning horizon until the final goal deadline.",
#   "planning_horizon_weeks": {planning_horizon_weeks},
#   "weekly_plans": [
#     {{
#       "week_number": 1,
#       "week_start_date": "YYYY-MM-DD",
#       "week_end_date": "YYYY-MM-DD",
#       "strategic_focus": "Main objective for this week",
#       "days": [
#         {{
#           "date": "YYYY-MM-DD",
#           "day_of_week": "Monday",
#           "daily_focus": "Main objective for the day",
#           "tasks": [
#             {{
#               "task_id": "The original task ID",
#               "title": "The original task title",
#               "status": "The current status of the task",
#               "description": "Detailed, actionable guide on HOW to complete the task",
#               "sub_tasks": [
#                 {{
#                   "title": "A smaller, actionable sub-task",
#                   "description": "Detailed 'how-to' guide for this sub-task"
#                 }}
#               ]
#             }}
#           ]
#         }}
#       ]
#     }}
#   ]
# }}

# **CRITICAL RULES**:
# 1. Multi-Week Planning: Generate exactly {planning_horizon_weeks} weeks of planning.
# 2. Prioritize Overdue Tasks: Any task marked "Deadline Exceeded" MUST be scheduled in Week 1.
# 3. Deadline Awareness: Schedule tasks based on their deadlines across appropriate weeks.
# 4. Logical Sequencing: Arrange tasks based on dependencies and logical workflow progression.
# 5. Weekly Focus: Each week should have a clear strategic focus and build upon previous weeks.
# 6. Detailed 'How-To' Descriptions: Provide specific, step-by-step guidance for tasks and sub-tasks.
# 7. Full Week Plans: Generate plans for all 7 days in each week. Use empty arrays for days with no tasks.
# 8. Realistic Workload: Distribute tasks evenly across weeks, considering typical work capacity.
# 9. Output ONLY JSON: Return a single, valid JSON object.

# Now, create the {planning_horizon_weeks}-week strategic plan.
# """

#         # Step 7: Call the AI using our custom client
#         try:
#             async_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY_PLANNING"))
#             response = await async_client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": "You are an expert AI project manager. Output a single, valid JSON object."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 model="llama-3.3-70b-versatile",
#                 temperature=0.7,
#                 response_format={"type": "json_object"}
#             )
#             plan_data_str = response["choices"][0]["message"]["content"]
#             plan_data = json.loads(plan_data_str)

#             # Step 8: Save each weekly plan to MongoDB
#             for weekly_plan in plan_data.get("weekly_plans", []):
#                 week_start_date = weekly_plan.get("week_start_date")

#                 existing_plan = weekly_plans_collection.find_one(
#                     {"user_id": user_id, "week_start_date": week_start_date}
#                 )

#                 if existing_plan:
#                     # Update existing plan
#                     weekly_plans_collection.update_one(
#                         {"_id": existing_plan["_id"]},
#                         {
#                             "$set": {
#                                 "plan": weekly_plan,
#                                 "updated_at": datetime.datetime.now(pytz.UTC),
#                             }
#                         },
#                     )
#                 else:
#                     # Insert new plan
#                     weekly_plans_collection.insert_one(
#                         {
#                             "user_id": user_id,
#                             "week_start_date": week_start_date,
#                             "week_number": weekly_plan.get("week_number"),
#                             "plan": weekly_plan,
#                             "created_at": datetime.datetime.now(pytz.UTC),
#                             "updated_at": datetime.datetime.now(pytz.UTC),
#                         }
#                     )

#             return plan_data

#         except json.JSONDecodeError as e:
#             logger.error(
#                 f"Failed to parse AI response as JSON: {str(e)}\nResponse: {plan_data_str}"
#             )
#             raise HTTPException(
#                 status_code=500, detail="Failed to generate a valid plan."
#             )
#         except Exception as e:
#             logger.error(f"Error in /Plan_my_week endpoint: {str(e)}", exc_info=True)
#             raise HTTPException(
#                 status_code=500, detail=f"An unexpected error occurred: {str(e)}"
#             )

#     except Exception as e:
#         logger.error(f"Error in /Plan_my_week endpoint: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500, detail=f"An unexpected error occurred: {str(e)}"
#         )


# # New code - Get multi-week plans for a user
# @app.get("/multi_week_plans/{user_id}")
# async def get_multi_week_plans(user_id: str, weeks: int = 4):
#     try:
#         # Get plans for the specified number of weeks
#         today = datetime.datetime.now().date()
#         start_of_current_week = today - datetime.timedelta(days=today.weekday())

#         # Calculate date range for the requested weeks
#         end_date = start_of_current_week + datetime.timedelta(weeks=weeks)

#         plans = list(
#             weekly_plans_collection.find(
#                 {
#                     "user_id": user_id,
#                     "week_start_date": {
#                         "$gte": start_of_current_week.strftime("%Y-%m-%d"),
#                         "$lte": end_date.strftime("%Y-%m-%d"),
#                     },
#                 }
#             ).sort("week_start_date", pymongo.ASCENDING)
#         )

#         # Convert ObjectId to string for JSON serialization
#         for plan in plans:
#             plan["_id"] = str(plan["_id"])
#             if "created_at" in plan:
#                 plan["created_at"] = plan["created_at"].isoformat()
#             if "updated_at" in plan:
#                 plan["updated_at"] = plan["updated_at"].isoformat()

#         return {"weekly_plans": plans}

#     except Exception as e:
#         logger.error(f"Error getting multi-week plans: {str(e)}")
#         raise HTTPException(
#             status_code=500, detail=f"Failed to get multi-week plans: {str(e)}"
#         )


@app.post("/analyze-images")
async def analyze_images(
    user_id: str = Form(...),
    session_id: str = Form(...),
    query: str = Form(...),
    images: List[UploadFile] = File(...),
):

    # Validate image count
    if len(images) == 0:
        return {"error": "Please upload at least one image."}
    if len(images) > 3:
        return {"error": "Please upload up to 3 images."}

    # Construct user message with image filenames
    image_filenames = [image.filename for image in images]
    user_message_text = f"{query}"

    # Generate embedding for the user's message
    user_embedding = await generate_text_embedding(user_message_text)

    # Prepare content for Groq API (text query + image data)
    content = [{"type": "text", "text": query}]
    for image in images:
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{image.content_type};base64,{base64_image}"
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    # Use AsyncGroq for consistency with /generate
    client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
    response = await client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[
            {
                "role": "system",
                "content": "Return response in plain english. Do not use LaTeX",
            },
            {"role": "user", "content": content},
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
    )
    output = response.choices[0].message.content

    # Generate embedding for the assistant's response
    assistant_embedding = await generate_text_embedding(output)

    # Create new messages for chat history
    new_messages = [
        {"role": "user", "content": user_message_text, "embedding": user_embedding},
        {"role": "assistant", "content": output, "embedding": assistant_embedding},
    ]

    # Save to chat history in chats_collection
    chat_entry = await chats_collection.find_one(
        {"user_id": user_id, "session_id": session_id}
    )
    if chat_entry:
        # Update existing chat entry
        await chats_collection.update_one(
            {"user_id": user_id, "session_id": session_id},
            {
                "$push": {"messages": {"$each": new_messages}},
                "$set": {"last_updated": datetime.datetime.now(datetime.timezone.utc)},
            },
        )
    else:
        # Insert new chat entry
        await chats_collection.insert_one(
            {
                "user_id": user_id,
                "session_id": session_id,
                "messages": new_messages,
                "last_updated": datetime.datetime.now(datetime.timezone.utc),
            }
        )

    # Check and store long-term memory if necessary
    updated_chat_entry = await chats_collection.find_one(
        {"user_id": user_id, "session_id": session_id}
    )
    if updated_chat_entry and len(updated_chat_entry["messages"]) >= 10:
        await store_long_term_memory(
            user_id, session_id, updated_chat_entry["messages"][-10:]
        )

    # Return the analysis output
    return {"output": output}


@app.websocket("/goal_setting")
async def goal_setting_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Wait for initial message with user_id and session_id
    current_date = get_current_datetime()
    initial_data = await websocket.receive_json()
    user_id = initial_data.get("user_id")
    session_id = initial_data.get("session_id")
    # async_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY_GOAL_SETTING"))

    if not user_id or not session_id:
        await websocket.send_json({"error": "Missing user_id or session_id"})
        await websocket.close()
        return

    # Initialize conversation state
    conversation_history = []
    goal_details = {}
    plan = None

    try:
        # Start the conversation
        initial_prompt = "Hello! I'm here to help you set and achieve your goals. To start, what is the goal you have in mind?"
        await websocket.send_json({"message": initial_prompt})
        conversation_history.append({"role": "assistant", "content": initial_prompt})

        while True:
            user_input = await websocket.receive_text()
            conversation_history.append({"role": "user", "content": user_input})

            if plan and user_input.lower() == "confirm":
                break  # Exit loop to save the plan

            # Decide the next step: ask questions or generate a plan
            if not plan:
                # RAG pipeline to ask contextual questions or generate a plan
                decision_prompt = f"""
                You are an expert goal-setting assistant. Your task is to have a friendly conversation to help a user define their goal.
                Based on the conversation history, decide on the next step.
                
                Conversation History:
                {json.dumps(conversation_history, indent=2)}

                If the user has clearly stated their goal and you have enough information (what, why, how, when), respond with a JSON object to generate a plan:
                {{
                    "action": "generate_plan",
                    "goal_title": "The user's goal title"
                }}

                Otherwise, ask a friendly, specific question to get more details. The question should guide the user to provide information related to the SMART framework (Specific, Measurable, Achievable, Relevant, Time-bound) without explicitly mentioning it. Respond with a JSON object:
                {{
                    "action": "ask_question",
                    "question": "Your friendly, contextual question"
                }}
                f"Current date/time: {current_date}\n"
                """
                response = await client.chat.completions.create(
                    messages=[{"role": "system", "content": decision_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )
                decision = json.loads(response.choices[0].message.content)

                if decision.get("action") == "ask_question":
                    question = decision["question"]
                    await websocket.send_json({"message": question})
                    conversation_history.append(
                        {"role": "assistant", "content": question}
                    )
                    continue

                if decision.get("action") == "generate_plan":
                    goal_details["title"] = decision["goal_title"]
                    # The rest of the details are in conversation_history

            # Generate or revise the plan
            plan_generation_prompt = f"""
            Based on the following conversation, generate a detailed and actionable plan for the user's goal.
            
            Conversation:
            {json.dumps(conversation_history, indent=2)}
            
            Your task is to:
            1.  Create a list of small, manageable tasks.
            2.  For each task, provide a clear 'title' (the what) and a 'description' (the how). The description should give the user practical steps.
            3.  Suggest a 'deadline' for each task in 'YYYY-MM-DD' format.
            4.  The output must be a JSON object with a single key "plan" which is a list of task objects.
            
            Example of a task object:
            {{
                "title": "Research local gyms",
                "description": "Use online maps and search engines to find gyms near you. Check their websites for pricing, classes, and amenities. Make a list of 3-4 potential options.",
                "deadline": "2024-07-30"
            }}

            Now, generate the plan based on the conversation.
            f"Current date/time: {current_date}\n"
            """
            if plan and user_input.lower() != "confirm":
                plan_generation_prompt += f"\nThe user has requested the following adjustments: {user_input}. Please revise the plan accordingly."

            response = await client.chat.completions.create(
                messages=[{"role": "system", "content": plan_generation_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            plan_data = json.loads(response.choices[0].message.content)
            plan = plan_data.get("plan", [])

            await websocket.send_json(
                {
                    "message": "Here is a plan to get you started. Does this look right? You can ask for changes or type 'confirm' to save it.",
                    "plan": plan,
                }
            )
            conversation_history.append(
                {"role": "assistant", "content": json.dumps(plan)}
            )

        # Save the confirmed goal and tasks
        goal_id = str(uuid.uuid4())
        new_goal = {
            "user_id": user_id,
            "goal_id": goal_id,
            "session_id": session_id,
            "title": goal_details.get("title", "Untitled Goal"),
            "description": "\n".join(
                [
                    msg["content"]
                    for msg in conversation_history
                    if msg["role"] == "user"
                ]
            ),
            "status": "active",
            "created_at": datetime.datetime.now(datetime.timezone.utc),
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
            "tasks": [],
        }

        for task_item in plan:
            task_id = str(uuid.uuid4())
            new_task = {
                "task_id": task_id,
                "title": task_item.get("title"),
                "description": task_item.get("description"),
                "status": "not started",
                "created_at": datetime.datetime.now(datetime.timezone.utc),
                "updated_at": datetime.datetime.now(datetime.timezone.utc),
                "deadline": task_item.get("deadline"),
                "progress": [],
            }
            new_goal["tasks"].append(new_task)

        await goals_collection.insert_one(new_goal)
        await websocket.send_json(
            {
                "message": "Great! Your goal has been saved. You can track your progress in the goals section."
            }
        )

        # Save conversation to chat history with embeddings
        messages_to_save = []
        for msg in conversation_history:
            embedding = await generate_text_embedding(msg["content"])
            messages_to_save.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "embedding": (
                        embedding if embedding and len(embedding) == 768 else None
                    ),
                }
            )

        chat_entry = await chats_collection.find_one(
            {"user_id": user_id, "session_id": session_id}
        )
        if chat_entry:
            await chats_collection.update_one(
                {"_id": chat_entry["_id"]},
                {"$push": {"messages": {"$each": messages_to_save}}},
            )
        else:
            await chats_collection.insert_one(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "messages": messages_to_save,
                    "last_updated": datetime.datetime.now(datetime.timezone.utc),
                }
            )

    except WebSocketDisconnect:
        logging.info("Client disconnected from goal setting.")
    except Exception as e:
        logging.error(f"Error in goal setting endpoint: {e}", exc_info=True)
        try:
            await websocket.send_json(
                {"error": "An unexpected error occurred. Please try again."}
            )
        except:
            pass
    finally:
        await websocket.close()


# In main16.py, add these endpoints near your other app routes.


@app.post("/aiassist")
async def ai_assist(input_data: UserInput):
    """Endpoint to generate a caption based on user content description."""
    try:
        seed_keywords = await generate_seed_keywords(input_data.query)
        logger.info(f"Seed keywords: {seed_keywords}")
        trending_hashtags = await fetch_trending_hashtags(seed_keywords)
        logger.info(f"Trending hashtags: {trending_hashtags}")
        seo_keywords = await fetch_seo_keywords(seed_keywords)
        logger.info(f"SEO keywords: {seo_keywords}")
        caption = await generate_caption(
            input_data.query, seed_keywords, trending_hashtags, seo_keywords
        )
        return {
            "caption": caption,
            "keywords": seed_keywords,
            "hashtags": trending_hashtags,
            "seo_keywords": seo_keywords,
        }
    except Exception as e:
        logger.error(f"Error in /aiassist endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.websocket("/wss/aiassist")
async def websocket_ai_assist(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            user_input = json.loads(data)
            query = user_input["query"]

            await websocket.send_text(
                json.dumps({"step": "Initializing AI Assistant..."})
            )

            seed_keywords = await generate_seed_keywords(query)
            await websocket.send_text(
                json.dumps(
                    {"step": "Generated Seed Keywords", "keywords": seed_keywords}
                )
            )

            trending_hashtags = await fetch_trending_hashtags(seed_keywords)
            await websocket.send_text(
                json.dumps(
                    {"step": "Fetched Trending Hashtags", "hashtags": trending_hashtags}
                )
            )

            seo_keywords = await fetch_seo_keywords(seed_keywords)
            await websocket.send_text(
                json.dumps(
                    {"step": "Fetched SEO Keywords", "seo_keywords": seo_keywords}
                )
            )

            await websocket.send_text(
                json.dumps({"step": "Generating Final Caption..."})
            )
            caption = await generate_caption(
                query, seed_keywords, trending_hashtags, seo_keywords
            )

            await websocket.send_text(
                json.dumps(
                    {
                        "step": "Caption ready",
                        "caption": caption,
                        "keywords": seed_keywords,
                        "hashtags": trending_hashtags,
                        "seo_keywords": seo_keywords,
                    }
                )
            )
    except WebSocketDisconnect:
        logger.info("AI Assist WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in AI Assist WebSocket: {e}", exc_info=True)
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        await websocket.close()


# Post Generotar


@app.websocket("/wss/generate-post")
async def websocket_generate_post(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json(
        {
            "status": "connected",
            "message": "Proof: The websocket_generate_post function is running.",
        }
    )

    logger.info("Connection accepted for /ws/generate-post.")

    try:
        # --- Move all logic inside the try block ---

        # 1. Safely get the post_option parameter
        post_option_type = websocket.query_params.get("post_option")
        logger.info(
            f"--- GENERATE-POST: Received post_option: {post_option_type} ---"
        )  # DEBUG LOG
        if not post_option_type:
            logger.error("'post_option' query parameter is missing.")
            await websocket.send_json(
                {
                    "status": "error",
                    "message": "Error: Missing post_option parameter in URL.",
                }
            )
            return

        post_option_type = post_option_type.lower()
        logger.info(f"Post generation option received: {post_option_type}")

        # 2. Check if the Groq client was created successfully
        client = await get_groq_client()
        if not client:
            logger.error("Failed to get Groq client, likely a missing API key.")
            await websocket.send_json(
                {
                    "status": "error",
                    "message": "Server configuration error: Could not connect to AI service.",
                }
            )
            return

        # 3. Proceed with the rest of your logic

        # The frontend sends a string of the form: "integer1,integer2,integer3" etc. Get the integers and convert them to platforms
        platform_options: List = await websocket.receive_text()
        await websocket.send_json(
            {"status": "processing", "message": "Received platform options..."}
        )
        platform_options = platform_options.split(',')
        platform_options = [Platforms.platform_list[int(x)] for x in platform_options]
        logger.info(f"Selected platform options : {platform_options}")

        prompt = await websocket.receive_text()
        logger.info(f"Received prompt for post generation: '{prompt}'")

        await websocket.send_json(
            {"status": "processing", "message": "Classifying post type..."}
        )
        post_type = await classify_post_type(client, prompt)

        await websocket.send_json(
            {
                "status": "processing",
                "message": f"Post classified as {post_type}. Generating keywords...",
            }
        )
        seed_keywords = await generate_keywords_post(client, prompt)

        await websocket.send_json(
            {"status": "processing", "message": "Fetching trending hashtags..."}
        )
        trending_hashtags = await fetch_trending_hashtags_post(client, seed_keywords, platform_options)

        await websocket.send_json(
            {"status": "processing", "message": "Fetching SEO keywords..."}
        )
        seo_keywords = await fetch_seo_keywords_post(client, seed_keywords)

        html_code, captions, media, parsed_media = None, None, None, None

        if post_option_type == PostGenOptions.Text:
            await websocket.send_json(
                {"status": "processing", "message": "Generating text-based post..."}
            )
            html_code = await generate_html_code_post(client, prompt, post_type)
        else:
            await websocket.send_json(
                {"status": "processing", "message": "Finding relevant media..."}
            )
            media = await get_pexels_data(seed_keywords, post_option_type)
            parsed_media = await parse_media(media, post_option_type)

            await websocket.send_json(
                {"status": "processing", "message": "Crafting the perfect caption..."}
            )
            captions = await generate_caption_post(
                client, prompt, seed_keywords, trending_hashtags, platform_options
            )

        await websocket.send_json(
            {
                "status": "completed",
                "message": "Post Generated Successfully!",
                "trending_hashtags": trending_hashtags,
                "seo_keywords": seo_keywords,
                "captions": captions,
                "html_code": html_code,
                "media": parsed_media,
                "post_type": post_option_type,
            }
        )

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/generate-post")
    except Exception as e:
        # This will now catch any crash and send a useful error to the frontend
        logger.error(f"Post generation failed with an exception: {e}", exc_info=True)
        await websocket.send_json(
            {
                "status": "error",
                "message": "A critical error occurred while generating the post. Please check the server logs.",
            }
        )
    finally:
        await websocket.close()
        logger.info("WebSocket connection for /ws/generate-post has been closed.")


@app.on_event("startup")
async def startup_event():
    await load_faiss_indices()
    asyncio.create_task(notification_checker())
    asyncio.create_task(daily_checkin_scheduler())
    asyncio.create_task(proactive_checkin_scheduler())


async def load_faiss_indices():
    try:
        async for mem in memory_collection.find():
            vector = np.array(mem["vector"], dtype="float32").reshape(1, -1)
            idx = doc_index.ntotal
            doc_index.add(vector)
            user_memory_map[mem["user_id"]] = idx
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}", exc_info=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
