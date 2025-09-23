# Social Media Content Generator API Documentation

## Overview

This FastAPI application provides a comprehensive social media content generation service that creates captions, fetches media, generates HTML posts, and provides SEO optimization for multiple social media platforms.

## Features

- **Multi-platform Support**: Facebook, Instagram, LinkedIn, Pinterest, Threads, TikTok, YouTube
- **Content Generation**: Automated caption creation tailored to each platform
- **Media Integration**: Photo and video fetching from Pexels API
- **SEO Optimization**: Keyword generation and trending hashtag suggestions
- **HTML Post Generation**: Dynamic HTML content creation with templates
- **AI-Powered Classification**: Tech vs non-tech content classification

## Dependencies

```python
import os
import re
import requests
import random
import logging
from typing import List

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from groq import Groq, AsyncGroq
from html2image import Html2Image
from enum import Enum
from dotenv import load_dotenv
```

## Environment Variables

The following environment variables must be configured:

- `PEXELS_API_KEY`: API key for Pexels media service
- `GROQ_API_KEY_*`: Multiple Groq API keys (numbered suffixes for load balancing)

## Data Models

### PostGenOptions (Enum)
Defines the types of content that can be generated:
- `Text`: Text-based content
- `Photo`: Image-based content  
- `Video`: Video-based content

### Platforms (Enum)
Supported social media platforms:
- facebook
- instagram
- linkedin
- pinterest
- threads
- tiktok
- youtube

## Core Functions

### Media Functions

#### `get_pexels_data(seed_keywords: List, media_type: str, per_page=9)`
**Purpose**: Fetches photos or videos from Pexels API based on keywords.

**Parameters**:
- `seed_keywords`: List of keywords for search
- `media_type`: Type of media (photo/video)
- `per_page`: Number of results to return (default: 9)

**Returns**: JSON response from Pexels API or None on error

**Usage**:
```python
media_data = await get_pexels_data(["sunset", "beach"], "photo", 5)
```

#### `parse_media(media: dict, post_option_type: PostGenOptions)`
**Purpose**: Parses and structures media data from Pexels response.

**Parameters**:
- `media`: Raw media data from Pexels API
- `post_option_type`: Type of content (Photo/Video)

**Returns**: List of structured media objects with URLs, dimensions, and metadata

### AI Client Management

#### `get_groq_client()`
**Purpose**: Creates an AsyncGroq client with random API key selection for load balancing.

**Returns**: AsyncGroq client instance

**Features**:
- Automatic load balancing across multiple API keys
- Error handling for missing API keys

### Content Classification

#### `classify_post_type(client: AsyncGroq, prompt: str)`
**Purpose**: Classifies content as 'tech' or 'non-tech' using AI.

**Parameters**:
- `client`: AsyncGroq client instance
- `prompt`: Content to classify

**Returns**: String ('tech' or 'non-tech')

**Model Used**: `llama-3.3-70b-versatile`

### Keyword & SEO Functions

#### `generate_keywords_post(client: AsyncGroq, query: str)`
**Purpose**: Generates 3 seed keywords from user query.

**Parameters**:
- `client`: AsyncGroq client instance
- `query`: User input query

**Returns**: List of keywords

**Model Used**: `deepseek-r1-distill-llama-70b`

#### `fetch_trending_hashtags_post(client: AsyncGroq, seed_keywords: List, platform_options: List)`
**Purpose**: Generates 20 trending hashtags relevant to keywords and platforms.

**Parameters**:
- `client`: AsyncGroq client instance
- `seed_keywords`: List of relevant keywords
- `platform_options`: Target social media platforms

**Returns**: List of hashtags

**Model Used**: `compound-beta-mini`

#### `fetch_seo_keywords_post(client: AsyncGroq, seed_keywords: List)`
**Purpose**: Generates 15 SEO-optimized keywords.

**Parameters**:
- `client`: AsyncGroq client instance
- `seed_keywords`: Base keywords for SEO expansion

**Returns**: List of SEO keywords

**Model Used**: `compound-beta-mini`

### Caption Generation

#### `generate_caption_post(client: AsyncGroq, query: str, seed_keywords: List, trending_hashtags: List, platform_options: List)`
**Purpose**: Creates platform-specific captions using predefined prompts.

**Parameters**:
- `client`: AsyncGroq client instance
- `query`: Original user request
- `seed_keywords`: Generated keywords
- `trending_hashtags`: Relevant hashtags
- `platform_options`: Target platforms

**Returns**: Dictionary with platform-specific captions

**Features**:
- Platform-specific prompt templates
- Customized content for each social media platform
- Integration of keywords and hashtags

**Model Used**: `deepseek-r1-distill-llama-70b`

### HTML Generation

#### `generate_html_code_post(client: AsyncGroq, prompt: str, template_type: str)`
**Purpose**: Generates complete HTML documents for social media posts.

**Parameters**:
- `client`: AsyncGroq client instance
- `prompt`: Content description
- `template_type`: Style template ('tech' or 'non-tech')

**Returns**: Complete HTML document string

**Features**:
- Template-based styling
- Complete HTML structure
- UTF-8 character encoding
- Automatic code extraction from AI response

**Model Used**: `llama-3.3-70b-versatile`

## Configuration

### File Paths
```python
GENERATED_IMAGE_FILE_PATH = "./generated_images/output_image.jpg"
GENERATED_IMAGE_FOLDER_PATH = "./generated_images"
```

### CORS Configuration
```python
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling

The application includes comprehensive error handling:

- **API Key Validation**: Checks for missing Pexels and Groq API keys
- **Request Exception Handling**: Manages network errors and API failures
- **Logging**: Uses uvicorn logger for error tracking
- **Graceful Degradation**: Returns error messages instead of crashing

## Usage Examples

### Basic Workflow
1. **Initialize**: Load environment variables and create FastAPI app
2. **Classify**: Determine if content is tech or non-tech
3. **Generate Keywords**: Create seed keywords from user query
4. **Fetch Media**: Get relevant photos/videos from Pexels
5. **Create Hashtags**: Generate trending hashtags
6. **Generate Captions**: Create platform-specific content
7. **Build HTML**: Generate visual post layouts

### Integration Pattern
```python
# Initialize client
client = await get_groq_client()

# Generate content pipeline
keywords = await generate_keywords_post(client, user_query)
hashtags = await fetch_trending_hashtags_post(client, keywords, platforms)
captions = await generate_caption_post(client, user_query, keywords, hashtags, platforms)
media = await get_pexels_data(keywords, "photo")
```

## Dependencies Installation

```bash
pip install fastapi uvicorn groq html2image python-dotenv requests jinja2
```

## External Services

- **Groq API**: AI model services for content generation
- **Pexels API**: Stock photo and video content
- **Templates**: External template constants (imported from GEN_CONSTANTS)

## Notes

- The application uses multiple Groq API keys for load balancing
- Template constants are imported from `GEN_CONSTANTS` module
- HTML generation includes automatic code block extraction
- Media parsing handles both HD and standard quality videos
- All functions are async for optimal performance
