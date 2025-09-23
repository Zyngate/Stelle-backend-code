TEMPLATES = {
    "tech": [
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Did You Know Card</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body, html {
      width: 100%; height: 100%;
      display: flex; justify-content: center; align-items: center;
      background: #5ea79f;
      font-family: 'Segoe UI', Roboto, sans-serif;
    }
    .card-wrapper {
      position: relative;
      width: 320px;
      height: auto;
    }
    .card, .card::after {
      border-radius: 16px;
    }
    .card::after {
      content: "";
      position: absolute;
      top: 12px; left: 12px;
      width: 100%; height: 100%;
      background: #000;  
      opacity: .15;
      z-index: -1;
    }
    .card {
      position: relative;
      background: #fdecd3;
      padding: 24px;
      color: #333;
    }
    .close-btn {
      position: absolute;
      top: 12px; right: 12px;
      width: 24px; height: 24px;
      border: none;
      background: #000;
      border-radius: 50%;
      color: #fff;
      font-size: 16px;
      line-height: 1;
      cursor: pointer;
    }
    .icon {
      width: 32px; height: 32px;
      margin-bottom: 16px;
    }
    .icon svg {
      width: 100%; height: 100%;
      fill: #f08c36;
    }
    .title {
      font-family: 'Luckiest Guy', sans-serif;
      font-size: 28px;
      color: #e17e24;
      margin-bottom: 12px;
    }
    .text {
      font-size: 14px;
      line-height: 1.6;
      margin-bottom: 16px;
    }
    .handle {
      font-size: 13px;
      opacity: .7;
      margin-bottom: 16px;
    }
    .read-more {
      display: inline-block;
      padding: 8px 16px;
      background: #000;
      color: #fff;
      border-radius: 12px;
      font-size: 14px;
      text-decoration: none;
    }
  </style>
  <link href="https://fonts.googleapis.com/css2?family=Luckiest+Guy&display=swap" rel="stylesheet">
</head>
<body>
  <div class="card-wrapper">
    <div class="card">
      <button class="close-btn" aria-label="Close">×</button>
      <div class="icon">
        <svg viewBox="0 0 24 24">
          <path d="M9 21h6v-1H9v1zm3-19C8.13 2 5 5.13 5 9c0 2.38 1.19 4.47 3 5.74V17h8v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.87-3.13-7-7-7zm1 12h-2v-2h2v2zm1.07-7.75l-.9.92C10.45 8.9 10 9.5 10 11h2c0-1 .45-1.5.88-1.92l1.24-1.26a1.003 1.003 0 10-1.05-1.64z"/>
        </svg>
      </div>
      <div class="title">DID YOU KNOW?</div>
      <div class="text">
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
      </div>
      <div class="handle">@reallygreatsite</div>
      <a href="#" class="read-more">Read More</a>
    </div>
  </div>
</body>
</html>
        """,
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Mindful Tech Usage Post</title>

  <!-- Google Fonts -->
  <link
    href="https://fonts.googleapis.com/css2?family=Oswald:wght@700&family=Open+Sans:wght@400;600&display=swap"
    rel="stylesheet"
  />

  <style>
    /* page background with halftone dots */
    body {
      margin: 0;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background-color: #f2f2f2;
      background-image: radial-gradient(#cccccc 1px, transparent 1px);
      background-size: 20px 20px;
      font-family: 'Open Sans', sans-serif;
    }

    /* white card */
    .post {
      background: #ffffff;
      border-radius: 24px;
      box-shadow: 0 6px 0 rgba(0, 0, 0, 0.1);
      width: 500px;
      padding: 48px 32px;
      box-sizing: border-box;
      text-align: center;
    }

    /* author name */
    .post .author {
      font-size: 18px;
      font-weight: 600;
      color: #333333;
      margin-bottom: 16px;
    }

    /* main title */
    .post .title {
      font-family: 'Oswald', sans-serif;
      font-size: 46px;
      font-weight: 700;
      text-transform: uppercase;
      line-height: 1.2;
      color: #000;
      margin: 0 0 32px;
    }

    /* circle + arrow */
    .post .cta {
      width: 64px;
      height: 64px;
      margin: 0 auto 32px;
      background-color: #ff5722;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .post .cta svg {
      width: 24px;
      height: 24px;
      fill: #fff;
    }

    /* bottom handle */
    .post .handle {
      font-size: 18px;
      font-weight: 400;
      color: #333333;
    }
  </style>
</head>

<body>
  <div class="post">
    <div class="author">Aaron Loeb</div>

    <div class="title">
      The Road to<br />
      Mindful Tech Usage
    </div>

    <div class="cta">
      <!-- simple right arrow SVG -->
      <svg viewBox="0 0 24 24">
        <path d="M10 17l5-5-5-5v10z"/>
      </svg>
    </div>

    <div class="handle">@reallygreatsite</div>
  </div>
</body>
</html>

        """
    ],
    "non-tech": [
        """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: 'Georgia', serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff9f9;
            color: #000;
        }
        .post-container {
            background-color: white;
            border-radius: 8px;
            padding: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #f0e6e6;
        }
        h1 {
            color: #8e44ad;
            font-size: 22px;
            margin-bottom: 15px;
            font-weight: normal;
            font-style: italic;
        }
        p {
            line-height: 1.8;
            margin-bottom: 15px;
        }
        .quote {
            border-left: 3px solid #9b59b6;
            padding-left: 15px;
            font-style: italic;
            color: #666;
            margin: 20px 0;
        }
        .footer {
            margin-top: 20px;
            font-size: 12px;
            color: #bdc3c7;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="post-container">
        <h1>{{ title }}</h1>
        <p>{{ content }}</p>
        <div class="footer">
            {{ footer_text }}
        </div>
    </div>
</body>
</html>
        """,
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Social Media Post</title>
  <!-- Import Inter font to match the clean, geometric look -->
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
  <style>
    /* Reset & container */
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body, html { width: 100%; height: 100%; }
    body {
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f5f5f5;
      font-family: 'Inter', sans-serif;
    }

    .post {
      position: relative;
      width: 600px;
      height: 600px;
      background:
        /* pale blue glow top-left */
        radial-gradient(circle at top left, rgba(173,216,230,0.4), transparent 60%),
        /* pink glow bottom-right */
        radial-gradient(circle at bottom right, rgba(255,182,193,0.4), transparent 60%),
        /* base white */
        #ffffff;
      border-radius: 12px;
      overflow: hidden;
      padding: 40px 40px 80px;
      color: #111;
    }

    /* Top asterisk */
    .post::before {
      content: '*';
      position: absolute;
      top: 24px;
      left: 24px;
      font-size: 28px;
      color: #6CA0DC;
    }

    /* Main heading text */
    .headline {
      margin-top: 60px;
      line-height: 1.1;
    }
    .headline p {
      font-size: 64px;
      font-weight: 600;
      letter-spacing: -0.5px;
      position: relative;
    }
    /* Underline "Boost" */
    .headline .underline {
      display: inline-block;
      position: relative;
    }
    .headline .underline::after {
      content: '';
      position: absolute;
      left: 0;
      bottom: 4px;
      width: 100%;
      height: 8px;
      background: #FFC0CB;
      border-radius: 4px;
      z-index: -1;
    }

    /* Swipe button */
    .swipe-btn {
      display: inline-block;
      margin-top: 40px;
      padding: 12px 24px;
      border: 2px solid #111;
      border-radius: 24px;
      font-size: 18px;
      text-decoration: none;
      color: #111;
      font-weight: 500;
      transition: background 0.2s, color 0.2s;
    }
    .swipe-btn:hover {
      background: #111;
      color: #fff;
    }

    /* Footer */
    .footer {
      position: absolute;
      bottom: 24px;
      left: 40px;
      right: 40px;
      display: flex;
      justify-content: space-between;
      font-size: 14px;
      font-weight: 400;
      color: #555;
    }
  </style>
</head>
<body>

  <div class="post">
    <div class="headline">
      <p>How to</p>
      <p><span class="underline">Boost Your</span></p>
      <p>Social Media Presence</p>
    </div>

    <a href="#" class="swipe-btn">(CTA)</a>

    <div class="footer">
      <div>(Company website)</div>
      <div>Year</div>
    </div>
  </div>

</body>
</html>
        """,
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Did You Know Card</title>
  <!-- Google Fonts -->
  <link
    href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital@1&family=Montserrat:wght@400&display=swap"
    rel="stylesheet"
  >
  <style>
    /* page background */
    body {
      margin: 0;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f7f1e8; /* light beige */
    }

    /* card container */
    .card {
      position: relative;
      width: 320px;
      padding: 32px 24px 24px;
      background: #fffdfa;
      border: 1px solid #e2d6c3;
      border-radius: 16px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
      font-family: 'Montserrat', sans-serif;
    }

    /* close button */
    .card .close-btn {
      position: absolute;
      top: 16px;
      right: 16px;
      width: 24px;
      height: 24px;
      background: none;
      border: none;
      font-size: 18px;
      line-height: 1;
      color: #cbb69e;
      cursor: pointer;
    }
    .card .close-btn:focus {
      outline: none;
    }

    /* title */
    .card .card-title {
      margin: 0;
      font-family: 'Playfair Display', serif;
      font-style: italic;
      font-weight: 400;
      font-size: 2.25rem;
      color: #1a1a1a;
      line-height: 1.1;
    }

    /* body text */
    .card .card-text {
      margin-top: 16px;
      font-size: 0.9rem;
      line-height: 1.5;
      color: #333;
    }

    /* pagination dots */
    .card .pagination {
      display: flex;
      gap: 8px;
      margin-top: 20px;
    }
    .card .pagination .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #e2d6c3; /* inactive */
    }
    .card .pagination .dot.active {
      background: #cbb69e; /* active */
    }

    /* bottom link bar */
    .card .card-link {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 24px;
      padding: 10px 16px;
      background: #fcfaf8;
      border: 1px solid #e2d6c3;
      border-radius: 12px;
      text-decoration: none;
      font-size: 0.9rem;
      color: #555;
    }
    .card .card-link svg {
      width: 12px;
      height: 12px;
      fill: #cbb69e;
      flex-shrink: 0;
    }
  </style>
</head>
<body>
  <div class="card">
    <!-- close icon -->
    <button class="close-btn" aria-label="Close">&times;</button>

    <!-- heading -->
    <h1 class="card-title">DID YOU KNOW?</h1>

    <!-- text -->
    <p class="card-text">
      Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc vitae lacus lorem.
      Phasellus finibus, enim eget ultrices venenatis, nulla magna faucibus ex,
      elementum posuere diam dui tempus risus. Sed congue vitae libero vitae ultrices.
      Aliquam vehicula mi eget tellus commodo, a porttitor lectus dictum.
    </p>

    <!-- pagination dots -->
    <div class="pagination">
      <span class="dot active"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </div>

    <!-- link bar -->
    <a href="https://www.reallygreatsite.com" class="card-link">
      www.reallygreatsite.com
      <!-- simple right‑arrow icon -->
      <svg viewBox="0 0 20 20">
        <path d="M7 4l6 6-6 6" stroke="currentColor" stroke-width="2" fill="none" fill-rule="evenodd"/>
      </svg>
    </a>
  </div>
</body>
</html>
  """,
  """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IG Post: Reminder to Self</title>
  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600&family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Great+Vibes&display=swap" rel="stylesheet">
  <style>
    /* center the square on the page */
    body {
      margin: 0;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f2f2f2;
    }

    /* the Instagram‐style square */
    .container {
      /* 1080×1080px square, but will shrink to fit smaller screens */
      width: min(1080px, 100vmin);
      aspect-ratio: 1 / 1;
      background: linear-gradient(135deg,
        #f0f3f5 0%,
        #c8a3f5 30%,
        #f6976c 70%,
        #fcb045 100%);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2rem;
      box-sizing: border-box;
      text-align: center;
    }

    .pill {
      font-family: 'Montserrat', sans-serif;
      font-weight: 600;
      font-size: 0.9rem;
      letter-spacing: 1px;
      text-transform: uppercase;
      padding: 0.5rem 1.2rem;
      border: 2px solid #000;
      border-radius: 50px;
      margin-bottom: 2rem;
    }

    h1 {
      font-family: 'Playfair Display', serif;
      font-size: 2.8rem;
      font-weight: 700;
      line-height: 1.2;
      text-transform: uppercase;
      margin: 0;
    }

    /* script style for “yourself” */
    h1 .script {
      font-family: 'Great Vibes', cursive;
      font-style: normal;
      text-transform: none;
      font-size: 2.8rem;
      margin-left: 0.2ch;
      display: inline-block;
    }

    .sub {
      font-family: 'Playfair Display', serif;
      font-size: 1rem;
      font-style: italic;
      margin-top: 2rem;
    }

    @media (max-width: 600px) {
      h1 {
        font-size: 2rem;
      }
      h1 .script {
        font-size: 2rem;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="pill">Reminder to Self</div>
    <h1>
      YOU&rsquo;LL BLOOM<br>
      IF YOU TAKE THE TIME TO WATER
      <span class="script">yourself</span>
    </h1>
    <div class="sub">self love is the best love</div>
  </div>
</body>
</html>
  """
    ]
}

FACEBOOK_CAPTION_PROMPT: str = f"""
You are a Facebook content strategist. Create warm, community-focused posts that encourage meaningful connections with friends, 
family, and local communities.

Requirements:
- Length: 100-300 words (longer storytelling welcomed)
- Tone: Warm, inclusive, personal storytelling
- Focus: Community engagement and emotional connection
- Structure: Story → Emotional connection → Community question
- Only the organic post with no markdown or labels

Essential Elements:
✓ Warmth
✓ Community-building questions
✓ Multigenerational appeal (all ages)
✓ Strategic emojis for connection
✓ Clear engagement call-to-action

Content Types:
- Life updates: "[Milestone] and the story behind it"
- Family moments: "[Experience] that taught me [lesson]"
- Local community: "Supporting [local business] because [reason]"
- Gratitude: "Feeling grateful for [specific people/moments]"
- Advice seeking: "Parents/friends, need wisdom on [topic]"
- Celebrations: "[Achievement] couldn't do it without [people]"

Some Engagement Starters:
- "What's your favorite memory of..."
- "How do you handle..."
- "Who else remembers when..."
- "What advice would you give..."
- "Tag someone who..."
- "Drop a ❤️ if you relate"

Format:

[A story with context]
[Emotional connection/lesson]
[Community question + emojis]
"""
INSTAGRAM_CAPTION_PROMPT: str = f"""
You are an expert Instagram caption writer. Create an engaging, scroll-stopping caption based on the user's input.

Requirements:
- Length: 50-100 words
- Hook: Start with attention-grabbing first line (question, bold statement, or relatable moment)
- Structure: Hook → Context/Story → Call-to-action
- Tone: Match desired vibe (casual, professional, inspirational, etc.)
- Emojis: Use naturally throughout
- Engagement: Include questions or prompts that encourage comments
- Line breaks: Use for readability
- Only the organic post with no markdown or labels

Essential Elements:
✓ Attention-grabbing opening line
✓ Personal touch or authentic voice
✓ Value (entertainment, inspiration, education, or connection)
✓ Clear call-to-action or engaging question
✓ 8-12 strategic hashtags (mix popular + niche)

Format:

[Hook line with emoji]

[Story/context with line breaks]

[Question or CTA]

[Hashtags]


Content Types:
- Food: Sensory description → memory/tip → ask for favorites
- Fitness: Motivation → tip → ask about goals  
- Travel: Feeling/moment → discovery → invite stories
- Business: Benefit → quick tip → encourage saves
- Personal: Vulnerable moment → lesson → ask for experiences
"""
PINTEREST_CAPTION_PROMPT: str = f"""
You are a Pinterest SEO expert. Create search-optimized pin descriptions that drive saves and clicks.

Requirements:
- Length: 100-200 characters
- Focus: SEO keywords+ clear value proposition
- Structure: Benefit → Keywords → Action
- Tone: Descriptive, actionable, keyword-rich
- Only the organic post with no markdown or labels

Essential Elements:
✓ Value proposition in first 50 characters
✓ 2-3 natural search keywords
✓ Action words (Get, Learn, Discover, Try, Find)
✓ Specific details (time, difficulty, audience)
✓ No hashtags (Pinterest reads description text)

Content Formulas:
- DIY: "Easy [project] in [time] | [skill level] tutorial"
- Recipes: "[Dish] - [benefit] in [time] | [dietary info]"
- Home: "[Style] [room] ideas | [budget] [element] inspiration"
- Fashion: "[Occasion] outfits | [season] [style] looks"
- Travel: "[Destination] guide | [type] tips & hidden gems"
- Beauty: "[Technique] tutorial | [time] [type] routine"
- Business: "[Topic] tips for [audience] | [benefit] strategies"

Keywords to Include:
- Descriptive: easy, quick, simple, budget-friendly, beginner
- Action: how to, tutorial, guide, tips, ideas, step-by-step
- Specifics: 30-minute, DIY, cozy, elegant, healthy
- Audience: busy moms, small business, college students
- Occasions: summer, wedding, Christmas, back-to-school

Format:
[Clear benefit] | [Descriptive keywords] [Action words]
"""

THREADS_CAPTION_PROMPT: str = f"""
# Threads Caption Generator Prompt

You are a Threads content expert. Create casual, conversational posts that spark authentic discussions and community engagement.

Requirements:
- Length: 50-200 words
- Tone: Casual, authentic, like talking to friends
- Focus: Real-time conversations and relatability
- Structure: Hook → Personal moment → Discussion starter
- Only the organic post with no markdown or labels

Essential Elements:
✓ Conversational opener (continuing a chat)
✓ Personal authenticity and relatability
✓ Discussion-driving question or opinion
✓ Current/trending topics when relevant
✓ Minimal emojis, no formal hashtags

Content Types:
- Hot takes: "Unpopular opinion: [harmless controversial view]"
- Relatable: "Anyone else [common experience]?"
- Questions: "Genuine question: [thought-provoking ask]"
- Observations: "Just noticed [interesting pattern]"
- Personal: "[Life moment] and here's what I learned"

Engagement Starters:
- "Am I the only one who..."
- "Unpopular opinion but..."
- "Can we normalize..."
- "Genuine question:"
- "Anyone else notice..."
- "Hot take:"

Format:

[Conversational hook]
[Personal take/relatable moment/contemplative]
[Question or discussion starter]
"""
TIKTOK_CAPTION_PROMPT: str = f"""
You are a TikTok expert. Create viral-worthy, trend-savvy captions that maximize engagement and algorithm reach.
Requirements:

Length: 50-150 characters (short and punchy)
Tone: Playful, trendy, Gen Z energy
Focus: Viral potential and discoverability
Structure: Hook → Context → Strategic hashtags
Only the organic post with no markdown or labels

Essential Elements:
✓ Viral hook (attention in 3 seconds)
✓ Trending language and current slang
✓ Strategic hashtag mix (trending + niche)
✓ Algorithm-friendly engagement bait
✓ Authentic relatability

Content Types:
Trends: "POV: [scenario]" or "[Sound] but make it [niche]"
Relatable: "Tell me you're [group] without telling me"
Educational: "Things I wish I knew about [topic]"
Storytime: "Storytime: [experience]"
Reviews: "Rating [topic] as someone who [qualifier]"

Some Viral Hooks:

"This is your sign to..." "Hear me out..."
"POV:" "Tell me why..." "Not me..." "Nobody:"
"Plot twist:" "Main character moment"
"Why is nobody talking about..."

Hashtag Strategy:

Mix: 2-3 trending + 3-5 niche + content-specific
Sizes: Big (1M+), medium (100K+), small (10K+)
Essential: #fyp #foryou (use sparingly)

Format:
[Viral hook]
[Brief context]
[Strategic hashtags]
"""
seo_keywords = ""
YOUTUBE_CAPTION_PROMPT: str = f"""
YouTube Caption Generator Prompt
You are a YouTube strategist. Create SEO-optimized video descriptions that boost discoverability, engagement, and subscriber growth.
Requirements:

Length: 50-150 words (first 125 characters crucial)
Tone: Informative, engaging, personality-driven
Focus: SEO optimization and viewer retention. Here are some seo keywords : {seo_keywords}
Structure: Hook → Value → Description → CTAs
Only the organic post with no markdown or labels

Essential Elements:
✓ Compelling first 125 characters (visible in search)
✓ SEO keywords naturally integrated
✓ Clear value proposition
✓ Multiple calls-to-action (subscribe, like, comment)

Content Types:
Tutorial: "Learn [skill] in [time] | [difficulty] guide"
Review: "Honest [product] review after [time] | Worth it?"
Educational: "Everything about [topic] in [time]"
Vlog: "[Experience] that changed my perspective on [topic]"

Description Format:
[Hook - First 125 chars with keywords]
[A short value proposition - what viewers gain + short description]
[Call to action]

CALLS-TO-ACTION:
👍 LIKE if this helped!
🔔 SUBSCRIBE for more [topic] content
💬 COMMENT your thoughts

#Keyword1 #Keyword2 #Category
Engagement Starters:

"What's your experience with [topic]? Let me know!"
"Which part surprised you? Drop a timestamp!"
"Made it to the end? Comment [emoji]"
"What should I cover next?"
"""
LINKEDIN_CAPTION_PROMPT = """
You are a LinkedIn content strategist. Create professional, engaging posts that drive meaningful engagement and build thought leadership.
Requirements:

Length: 150-200 words (optimal for algorithm)
Tone: Professional yet conversational, authentic, insightful
Focus: Value-driven content and thought leadership
Structure: Hook → Context/Story → Key insight → CTA
Only return the organic post with no markdown or labels and nothing else

Essential Elements:
✓ Compelling hook (question, stat, bold statement)
✓ Professional storytelling with authenticity
✓ Clear takeaway or actionable insight
✓ Engagement-driving discussion prompt
✓ 3-5 relevant industry hashtags
✓ White space for readability (2-3 lines max per paragraph)

Content Types:
Career milestone: Achievement → journey → advice for others
Industry insight: Trend → analysis → future implications
Leadership lesson: Challenge → solution → broader principle
Personal story: Experience → lesson → professional application
Thought leadership: Opinion → evidence → discussion starter

Hook Examples:

"Here's what 5 years in [industry] taught me..."
"Unpopular opinion: [thoughtful controversial take]"
"The best career advice I never followed:"
"[Number]% of professionals make this mistake..."
"Yesterday, I learned something that changed how I think about [topic]"

Format:
[Hook - question, stat, or bold statement]

[Story/context with line breaks]

[Key insight or takeaway]

[Discussion question]

#Industry #Leadership #CareerGrowth #ProfessionalDevelopment
"""

HTML_FRONTEND = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Post Generator</title>
    <style>
      body {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
      }

      .container {
        background: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
      }

      h1 {
        text-align: center;
        color: #333;
        margin-bottom: 30px;
        font-size: 2.5em;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }

      .input-section {
        margin-bottom: 30px;
      }

      textarea {
        width: 100%;
        height: 120px;
        padding: 15px;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        font-size: 16px;
        resize: vertical;
        box-sizing: border-box;
        transition: border-color 0.3s ease;
      }

      textarea:focus {
        outline: none;
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
      }

      .radio-group {
        display: flex;
        gap: 20px;
        margin: 20px 0;
        justify-content: center;
      }

      .radio-option {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        border: 2px solid #e0e0e0;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        background: #f8f9fa;
      }

      .radio-option:hover {
        border-color: #667eea;
        background: #f0f2ff;
      }

      .radio-option input[type="radio"] {
        accent-color: #667eea;
      }

      .radio-option input[type="radio"]:checked + label {
        color: #667eea;
        font-weight: bold;
      }

      .radio-option:has(input:checked) {
        border-color: #667eea;
        background: #f0f2ff;
      }

      .checkbox-group {
        display: flex;
        gap: 15px;
        margin: 20px 0;
        justify-content: center;
        flex-wrap: wrap;
      }

      .checkbox-option {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 15px;
        border: 2px solid #e0e0e0;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        background: #f8f9fa;
      }

      .checkbox-option:hover {
        border-color: #667eea;
        background: #f0f2ff;
      }

      .checkbox-option input[type="checkbox"] {
        accent-color: #667eea;
      }

      .checkbox-option input[type="checkbox"]:checked + label {
        color: #667eea;
        font-weight: bold;
      }

      .checkbox-option:has(input:checked) {
        border-color: #667eea;
        background: #f0f2ff;
      }

      .generate-btn {
        display: block;
        width: 200px;
        margin: 20px auto;
        padding: 15px 30px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
      }

      .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
      }

      .generate-btn:disabled {
        opacity: 0.6;
        cursor: not-allowed;
        transform: none;
      }

      .results-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-top: 30px;
      }

      .result-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
      }

      .result-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
      }

      .result-title {
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .result-content {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        white-space: pre-wrap;
        word-wrap: break-word;
        min-height: 80px;
        font-size: 14px;
        line-height: 1.5;
      }

      .status {
        text-align: center;
        padding: 15px;
        margin: 20px 0;
        border-radius: 8px;
        font-weight: bold;
      }

      .status.connecting,
      .processing {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        text-align: center;
      }

      .status.connected,
      .completed {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        text-align: center;
      }

      .status.error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
      }

      .icon {
        width: 20px;
        height: 20px;
      }

      @media (max-width: 768px) {
        .radio-group {
          flex-direction: column;
          align-items: center;
        }

        .checkbox-group {
          flex-direction: column;
          align-items: center;
        }

        .results-container {
          grid-template-columns: 1fr;
        }
        .result-content {
          width: 100%;
          overflow: auto;
        }
        #post-preview {
          width: 100%;
          aspect-ratio: 1 / 3;
          max-width: 100%;
          box-sizing: border-box;
          border: 1px solid #2e1313;
          border-radius: 4px;
        }
      }
      #downloadBtn {
        background-color: #4caf50;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        margin: 10px 0;
      }

      #downloadBtn:hover {
        background-color: #45a049;
      }

      #downloadBtn:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>🚀 Social Media Post Generator</h1>

      <div class="input-section">
        <textarea
          id="contentInput"
          placeholder="Enter your content idea or description here..."
        ></textarea>
      </div>

      <div class="radio-group">
        <div class="radio-option">
          <input
            type="radio"
            id="text"
            name="PostGenOption"
            value="text"
            checked
          />
          <label for="text">📝 Text</label>
        </div>
        <div class="radio-option">
          <input type="radio" id="photo" name="PostGenOption" value="photo" />
          <label for="photo">📸 Photo</label>
        </div>
        <div class="radio-option">
          <input type="radio" id="video" name="PostGenOption" value="video" />
          <label for="video">🎥 Video</label>
        </div>
      </div>

      <div class="checkbox-group">
        <div class="checkbox-option">
          <input
            type="checkbox"
            id="Facebook"
            name="platforms"
            value="Facebook"
          />
          <label for="Facebook">📘 Facebook</label>
        </div>
        <div class="checkbox-option">
          <input
            type="checkbox"
            id="LinkedIn"
            name="platforms"
            value="LinkedIn"
          />
          <label for="LinkedIn">💼 LinkedIn</label>
        </div>
        <div class="checkbox-option">
          <input
            type="checkbox"
            id="Instagram"
            name="platforms"
            value="Instagram"
          />
          <label for="Instagram">📷 Instagram</label>
        </div>
        <div class="checkbox-option">
          <input
            type="checkbox"
            id="Pinterest"
            name="platforms"
            value="Pinterest"
          />
          <label for="Pinterest">📌 Pinterest</label>
        </div>
        <div class="checkbox-option">
          <input
            type="checkbox"
            id="Threads"
            name="platforms"
            value="Threads"
          />
          <label for="Threads">🧵 Threads</label>
        </div>
        <div class="checkbox-option">
          <input type="checkbox" id="TikTok" name="platforms" value="TikTok" />
          <label for="TikTok">🎵 TikTok</label>
        </div>
        <div class="checkbox-option">
          <input
            type="checkbox"
            id="YouTube"
            name="platforms"
            value="YouTube"
          />
          <label for="YouTube">📺 YouTube</label>
        </div>
      </div>

      <button class="generate-btn" onclick="generatePost()">Generate</button>

      <div id="status" class="status" style="display: none"></div>

      <div id="resultsContainer" class="results-container"></div>
    </div>

    <script>
      let ws = null;
      let platforms_selected = [];
      const platforms = [
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Pinterest",
        "Threads",
        "TikTok",
        "YouTube",
      ];

      function generatePost() {
        const checkedRadio = document.querySelector(
          'input[name="PostGenOption"]:checked'
        );
        const contentInput = document.getElementById("contentInput");
        const statusDiv = document.getElementById("status");
        const resultsContainer = document.getElementById("resultsContainer");
        const generateBtn = document.querySelector(".generate-btn");

        if (!checkedRadio) {
          alert("Please select a post type");
          return;
        }

        // Check if at least one platform is selected
        const checkedPlatforms = document.querySelectorAll(
          'input[name="platforms"]:checked'
        );
        if (checkedPlatforms.length === 0) {
          alert("Please select at least one platform");
          return;
        }

        // Update platforms_selected array
        platforms_selected = [];
        checkedPlatforms.forEach((platform) => {
          const index = platforms.indexOf(platform.value);
          if (index !== -1) {
            platforms_selected.push(index);
          }
        });

        const postOption = checkedRadio.value;
        const content = contentInput.value.trim();

        if (!content) {
          alert("Please enter some content");
          return;
        }

        // Clear previous results
        resultsContainer.innerHTML = "";

        // Show status and disable button
        statusDiv.style.display = "block";
        statusDiv.textContent = "Connecting...";
        statusDiv.className = "status connecting";
        generateBtn.disabled = true;
        generateBtn.textContent = "Generating...";

        // Close existing websocket if any
        if (ws) {
          ws.close();
        }

        // Create websocket connection with platforms_selected as query parameter
        const wsUrl = `ws://${window.location.host}/ws/generate-post?post_option=${postOption}`;
        ws = new WebSocket(wsUrl);
        console.log("The array of platform options sent is: " + platforms_selected);
        ws.onopen = function (event) {
          ws.send(platforms_selected.join(","));
          statusDiv.textContent = "Connected - Generating post...";
          statusDiv.className = "status connected";

          //Send the content to the websocket
          ws.send(content);
        };
        ws.onmessage = function (event) {
          try {
            const data = JSON.parse(event.data);
            const statusDiv = document.getElementById("status");
            if (data.status === "processing") {
              statusDiv.className = "processing";
              statusDiv.innerText = data.message;
            } else if (data.status === "completed") {
              statusDiv.className = "completed";
              statusDiv.innerText = data.message;

              //console log the recieved data
              console.log(data.seed_keywords);
              console.log(data.trending_hashtags);
              console.log(data.seo_keywords);
              console.log(data.captions);
              console.log(data.html_code);
              console.log(data.media);

              //Display data on the frontend
              displayResults(data);
              generateBtn.disabled = false;
              generateBtn.textContent = "Generate";
            }
          } catch (error) {
            console.error("Error parsing response:", error);
            statusDiv.textContent = "Error: Invalid response format";
            statusDiv.className = "status error";
          }
        };

        ws.onerror = function (error) {
          console.error("WebSocket error:", error);
          statusDiv.textContent = "Connection error. Please try again.";
          statusDiv.className = "status error";
          generateBtn.disabled = false;
          generateBtn.textContent = "Generate";
        };

        ws.onclose = function (event) {
          generateBtn.disabled = false;
          generateBtn.textContent = "Generate";

          if (event.code !== 1000) {
            statusDiv.textContent = "Connection closed unexpectedly";
            statusDiv.className = "status error";
          }
        };
      }
      async function downloadImage() {
        const button = document.getElementById("downloadBtn");
        try {
          // Disable button during download
          button.disabled = true;
          button.textContent = "Downloading...";

          // Make GET request to download endpoint
          const response = await fetch(
            `http://${window.location.host}/download-image`
          );

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          // Get the blob data
          const blob = await response.blob();

          // Create a temporary URL for the blob
          const url = window.URL.createObjectURL(blob);

          // Create a temporary anchor element and trigger download
          const a = document.createElement("a");
          a.href = url;
          a.download = "downloaded_image.jpg"; // Default filename
          document.body.appendChild(a);
          a.click();

          // Clean up
          document.body.removeChild(a);
          window.URL.revokeObjectURL(url);

          // Reset button
          button.disabled = false;
          button.textContent = "Download Image";
        } catch (error) {
          console.error("Download failed:", error);
          alert("Failed to download image. Please try again.");

          // Reset button
          button.disabled = false;
          button.textContent = "Download Image";
        }
      }

      function displayResults(data) {
        const resultsContainer = document.getElementById("resultsContainer");

        const results = [
          {
            title: "📈 Trending Hashtags",
            content: data.trending_hashtags || "No hashtags available",
            icon: "📈",
          },
          {
            title: "🔍 SEO Keywords",
            content: data.seo_keywords || "No keywords available",
            icon: "🔍",
          },
          {
            title: "✨ Caption",
            content: data.captions
              ? Object.entries(data.captions)
                  .map(([platform, caption]) => `${platform.toUpperCase()} - ${caption}`)
                  .join("\n\n")
              : "No caption available",
            icon: "✨",
          },
          {
            title: "🌐 HTML Content",
            content: data.html_code || "No HTML content available",
            icon: "🌐",
          },
          {
            title: "📱 Media",
            content: data.media || "No media available",
            icon: "📱",
          },
        ];

        results.forEach((result) => {
          const card = document.createElement("div");
          card.className = "result-card";

          let cardContent = `
                    <div class="result-title">
                        <span class="icon">${result.icon}</span>
                        ${result.title}
                    </div>
                    <div class="result-content">${result.content}</div>
                `;

          if (data.html_code !== null && result.title === "🌐 HTML Content") {
            cardContent = `<div class="result-title">
                        <span class="icon">${result.icon}</span>
                        ${result.title}
                    </div>
                    <div class="result-content"> Generated HTML:<iframe id="post-preview" srcdoc="${data.html_code.replace(
                      /"/g,
                      "&quot;"
                    )}"></iframe></div>`;
            cardContent += `<button id="downloadBtn" onclick="downloadImage()">
          Download Image
        </button>`;
          }

          card.innerHTML = cardContent;

          resultsContainer.appendChild(card);
        });
      }
      // Clean up websocket on page unload
      window.addEventListener("beforeunload", function () {
        if (ws) {
          ws.close();
        }
      });
    </script>
  </body>
</html>
"""
