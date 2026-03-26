"""
Content Pipeline Agent — ZyndAI Fleet #3

Takes long-form content (YouTube video, blog post, article URL) and repurposes into
blog post, tweet thread, LinkedIn post, newsletter snippet. Fully self-contained.

    cd /Users/sahil/work/zyndai/agents && uv run python content_pipeline_agent.py
"""

from zyndai_agent.agent import AgentConfig, ZyndAIAgent
from zyndai_agent.message import AgentMessage
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from apify_client import ApifyClient
from deploy_config import get_webhook_url
from dotenv import load_dotenv
import os
import re
import time
import signal
import threading

load_dotenv()

apify = ApifyClient(os.environ["APIFY_API_TOKEN"])


@tool
def extract_youtube_transcript(youtube_url: str) -> str:
    """Extract transcript from a YouTube video. Returns the full text for repurposing."""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", youtube_url)
    if not match:
        return f"Invalid YouTube URL: {youtube_url}"
    try:
        run = apify.actor("topaz_sharingan/Youtube-Transcript-Scraper").call(
            run_input={"urls": [youtube_url], "language": "en"},
            timeout_secs=120,
        )
        items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
        if not items:
            return f"No transcript available for {youtube_url}. Video may not have captions."

        item = items[0]
        title = item.get("title", "Unknown Video")
        transcript = item.get("transcript", item.get("text", item.get("content", "")))

        if isinstance(transcript, list):
            transcript = " ".join(
                seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in transcript
            )

        if not transcript:
            for val in item.values():
                if isinstance(val, str) and len(val) > 200:
                    transcript = val
                    break

        if not transcript:
            return f"Empty transcript. Video: {title}. Response keys: {list(item.keys())}"

        return f"Video: {title}\nURL: {youtube_url}\n\nTranscript ({len(transcript)} chars):\n\n{transcript[:8000]}"
    except Exception as e:
        return f"Transcript extraction failed: {e}"


@tool
def crawl_article(url: str) -> str:
    """Extract the main text content from a blog post or article URL."""
    try:
        run = apify.actor("apify/website-content-crawler").call(
            run_input={
                "startUrls": [{"url": url}],
                "maxCrawlPages": 1,
                "maxCrawlDepth": 0,
                "crawlerType": "cheerio",
            },
            timeout_secs=90,
        )
        items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
        if not items:
            return f"No content extracted from {url}"
        item = items[0]
        title = item.get("metadata", {}).get("title", "Unknown")
        text = item.get("text", "")[:8000]
        return f"Title: {title}\nURL: {url}\n\nContent ({len(text)} chars):\n\n{text}"
    except Exception as e:
        return f"Crawl failed: {e}"


SYSTEM_PROMPT = """You are a content repurposing specialist. Given a source (YouTube URL, blog URL, or raw text), extract it and transform into multiple platform-native formats.

You have 2 tools:
- extract_youtube_transcript: for YouTube URLs
- crawl_article: for blog/article URLs
If given raw text, skip tools and transform directly.

Output — produce ALL sections:

## Source Analysis
Title: [original title]
Key Themes: [3-5 bullet points]
Target Audience: [who would care]

---

## Blog Post (800-1200 words)
[Full blog post with title, intro hook, body sections with headers, conclusion with CTA]

---

## Tweet Thread (5-8 tweets)
1/ [Hook — most compelling insight]
2/ [Supporting point]
3/ [Data or example]
...
N/ [CTA or takeaway]

---

## LinkedIn Post (150-300 words)
[Professional tone, insight-driven, end with engagement question]

---

## Newsletter Snippet (100-150 words)
Subject Line: [compelling email subject]
[Concise summary with link CTA]

---

## Key Quotes
- "[Quote 1]"
- "[Quote 2]"
- "[Quote 3]"

Each format must be native to its platform."""


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    tools = [extract_youtube_transcript, crawl_article]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=4)


def run_server():
    agent_config = AgentConfig(
        name="Content Pipeline Agent",
        description="Content repurposing engine — takes YouTube videos, blog posts, or articles "
        "and transforms them into blog posts, tweet threads, LinkedIn posts, newsletter snippets, "
        "and key quotes. Platform-native output for each format.",
        capabilities={
            "ai": ["nlp", "content_generation", "summarization", "langchain"],
            "protocols": ["http"],
            "services": ["content_repurposing", "youtube_to_blog", "social_media_content",
                         "tweet_thread_generation", "linkedin_post_generation", "newsletter_writing"],
            "domains": ["content_marketing", "social_media", "marketing"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5012,
        registry_url="https://registry.zynd.ai",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-content-pipeline",
        use_ngrok=False,
        webhook_url=get_webhook_url("content-pipeline"),
        sync_response_timeout=300,
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    zynd_agent.set_langchain_agent(create_agent())

    def handle_message(message: AgentMessage, topic: str):
        def _process():
            print(f"\n[Content Pipeline] query={message.content} from={message.sender_id}")
            try:
                response = zynd_agent.invoke(message.content, chat_history=[])
                zynd_agent.set_response(message.message_id, response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                zynd_agent.set_response(message.message_id, f"Error: {e}")
        threading.Thread(target=_process, daemon=True).start()

    zynd_agent.add_message_handler(handle_message)
    print(f"\nContent Pipeline Agent | {zynd_agent.webhook_url} | {zynd_agent.agent_id}")

    if os.isatty(0):
        print("Type your query directly, or 'exit' to quit.\n")
        while True:
            try:
                query = input("> ").strip()
                if not query:
                    continue
                if query.lower() == "exit":
                    break
                t = time.time()
                result = zynd_agent.invoke(query, chat_history=[])
                print(f"\n{result}\n\n({time.time() - t:.1f}s)\n")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        threading.Event().wait()


if __name__ == "__main__":
    run_server()
