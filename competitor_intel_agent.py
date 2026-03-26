"""
Competitor Intel Agent — ZyndAI Fleet #2

Tracks competitors: pricing, features, job postings, news, social sentiment.
Discovers and cross-calls Lead Enrichment Agent via ZyndAI registry for deep dossiers.

    cd /Users/sahil/work/zyndai/agents && uv run python competitor_intel_agent.py
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
import json
import time
import signal
import threading
import requests

load_dotenv()

apify = ApifyClient(os.environ["APIFY_API_TOKEN"])
REGISTRY_URL = "https://registry.zynd.ai"


def _google(query: str, max_results: int = 10) -> list[dict]:
    run = apify.actor("apify/google-search-scraper").call(
        run_input={
            "queries": query,
            "maxPagesPerQuery": 1,
            "resultsPerPage": max_results,
            "languageCode": "en",
            "countryCode": "us",
        },
        timeout_secs=120,
    )
    items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
    return items[0].get("organicResults", []) if items else []


def _fmt(results: list[dict], limit: int = 10) -> str:
    lines = []
    for r in results[:limit]:
        lines.append(f"- {r.get('title', 'N/A')}\n  {r.get('description', 'N/A')}\n  {r.get('url', '')}")
    return "\n\n".join(lines) if lines else "No results."


def _call_zynd_agent(webhook_url: str, message_content: str, timeout: int = 180) -> str:
    sync_url = webhook_url.rstrip("/")
    if not sync_url.endswith("/sync"):
        sync_url = sync_url.replace("/webhook", "/webhook/sync")
    try:
        resp = requests.post(
            sync_url,
            json={"content": message_content, "prompt": message_content,
                   "sender_id": "competitor-intel-agent", "message_type": "query"},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "timeout":
                return "Agent timed out. Try a simpler query."
            return data.get("response", json.dumps(data)[:3000])
        return f"HTTP {resp.status_code}: {resp.text[:300]}"
    except requests.exceptions.ConnectionError:
        return "Agent not reachable."
    except requests.exceptions.Timeout:
        return f"No response within {timeout}s."
    except Exception as e:
        return f"Call failed: {e}"


@tool
def search_competitor_news(query: str) -> str:
    """Search Google for recent news, announcements, and updates about a competitor."""
    try:
        return _fmt(_google(query))
    except Exception as e:
        return f"News search failed: {e}"


@tool
def search_jobs_and_hiring(company_name: str) -> str:
    """Search for job postings to detect hiring signals — what roles a competitor is filling reveals strategic direction."""
    try:
        return _fmt(_google(
            f"{company_name} jobs careers site:linkedin.com/jobs OR site:lever.co OR site:greenhouse.io OR site:ashbyhq.com"
        ))
    except Exception as e:
        return f"Job search failed: {e}"


@tool
def search_pricing_features_sentiment(company_name: str) -> str:
    """Search for pricing, features, and social discussion about a competitor in one call. Covers pricing pages, changelogs, Reddit, and Hacker News."""
    try:
        return _fmt(_google(
            f"{company_name} pricing OR changelog OR review site:reddit.com OR site:news.ycombinator.com OR site:g2.com"
        ))
    except Exception as e:
        return f"Search failed: {e}"


@tool
def crawl_page(url: str) -> str:
    """Crawl a specific page (pricing, blog, changelog) for detailed content extraction."""
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
            return f"No content from {url}"
        text = items[0].get("text", "")[:4000]
        title = items[0].get("metadata", {}).get("title", "")
        return f"### {title}\n{text}"
    except Exception as e:
        return f"Crawl failed: {e}"


@tool
def scrape_competitor_tweets(company_name: str) -> str:
    """Scrape live tweets mentioning a competitor. Returns actual tweets with engagement data — better than Google site:twitter.com search."""
    try:
        run = apify.actor("apidojo/tweet-scraper").call(
            run_input={
                "searchTerms": [company_name],
                "maxTweets": 15,
                "sort": "Latest",
            },
            timeout_secs=120,
        )
        items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
        if not items:
            return f"No tweets found mentioning {company_name}"
        lines = []
        for t in items[:15]:
            author = t.get("author", {}).get("userName", t.get("user", {}).get("screen_name", "unknown"))
            text = t.get("text", t.get("full_text", ""))[:280]
            likes = t.get("likeCount", t.get("favorite_count", 0))
            retweets = t.get("retweetCount", t.get("retweet_count", 0))
            date = t.get("createdAt", t.get("created_at", ""))[:10]
            lines.append(f"- @{author} ({date}) [{likes} likes, {retweets} RTs]\n  {text}")
        return f"Live tweets about '{company_name}':\n\n" + "\n\n".join(lines)
    except Exception as e:
        return f"Twitter scrape failed: {e}"


@tool
def get_company_dossier(company_name: str) -> str:
    """Call the Lead Enrichment Agent on ZyndAI to get a deep company dossier — website, tech stack, leadership, funding. Discovers the agent via registry search."""
    try:
        resp = requests.get(
            f"{REGISTRY_URL}/agents",
            params={"keyword": "lead enrichment", "limit": 3},
            timeout=15,
        )
        if resp.status_code != 200:
            return f"Registry search failed: HTTP {resp.status_code}"
        agents = resp.json().get("data", [])
        enrichment_agent = None
        for a in agents:
            if "enrichment" in a.get("name", "").lower() and a.get("httpWebhookUrl"):
                enrichment_agent = a
                break
        if not enrichment_agent:
            return "Lead Enrichment Agent not found on registry."
        return _call_zynd_agent(enrichment_agent["httpWebhookUrl"], f"Enrich: {company_name}")
    except Exception as e:
        return f"Discovery failed: {e}"


SYSTEM_PROMPT = """You are a competitive intelligence analyst. Given a competitor name, produce a structured intel report.

IMPORTANT: Be efficient. You have 6 tools — use at most 3 per request:
- search_competitor_news: recent news, launches, partnerships
- search_jobs_and_hiring: job postings reveal strategic direction
- search_pricing_features_sentiment: pricing + features + Reddit/HN sentiment in ONE call
- crawl_page: deep-dive a specific URL found via search
- scrape_competitor_tweets: LIVE tweets with engagement data (likes, RTs) — use for social sentiment
- get_company_dossier: call Lead Enrichment Agent for deep background (use sparingly, takes time)

Typical flow: search_competitor_news + search_pricing_features_sentiment. Add search_jobs_and_hiring only if asked about strategy.

Output format:

## Competitive Intelligence: [Competitor]

### Executive Summary
[3-4 sentences on competitive position]

### Recent Moves
- [Key announcements, launches]

### Pricing
- Current model: [details]
- Recent changes: [if any]

### Features
- Core: [list]
- Recent additions: [list]

### Hiring Signals
- Roles: [categories]
- Strategic read: [what it means]

### Sentiment
- Reddit/HN: [summary]
- Overall: Positive / Mixed / Negative

### Strengths & Vulnerabilities
- Strengths: [list]
- Vulnerabilities: [list]

### Recommendations
- [Actionable items]

Cite sources. Mark unavailable data clearly."""


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_competitor_news, search_jobs_and_hiring, search_pricing_features_sentiment,
             crawl_page, scrape_competitor_tweets, get_company_dossier]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=8)


def run_server():
    agent_config = AgentConfig(
        name="Competitor Intel Agent",
        description="Competitive intelligence analyst — tracks pricing, features, job postings, "
        "news, and social sentiment. Discovers and cross-calls Lead Enrichment Agent "
        "via ZyndAI registry for deep dossiers.",
        capabilities={
            "ai": ["nlp", "competitive_analysis", "sentiment_analysis", "langchain"],
            "protocols": ["http"],
            "services": ["competitive_intelligence", "pricing_monitoring", "feature_tracking",
                         "hiring_signal_detection", "market_research"],
            "domains": ["business_intelligence", "strategy", "market_analysis"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5011,
        registry_url="https://registry.zynd.ai",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-competitor-intel",
        use_ngrok=False,
        webhook_url=get_webhook_url("competitor-intel"),
        sync_response_timeout=300,
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    zynd_agent.set_langchain_agent(create_agent())

    def handle_message(message: AgentMessage, topic: str):
        def _process():
            print(f"\n[Competitor Intel] query={message.content} from={message.sender_id}")
            try:
                response = zynd_agent.invoke(message.content, chat_history=[])
                zynd_agent.set_response(message.message_id, response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                zynd_agent.set_response(message.message_id, f"Error: {e}")
        threading.Thread(target=_process, daemon=True).start()

    zynd_agent.add_message_handler(handle_message)
    print(f"\nCompetitor Intel Agent | {zynd_agent.webhook_url} | {zynd_agent.agent_id}")

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
