"""
AdRecon Agent — ZyndAI Fleet #8

Ad creative intelligence — monitors competitor ads on Meta Ad Library, Google,
and TikTok. Extracts ad copy, analyzes messaging strategy, identifies winning formats.

    cd /Users/sahil/work/zyndai/agents && uv run python adrecon_agent.py
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


def _call_zynd_agent(webhook_url: str, content: str) -> str:
    sync_url = webhook_url.replace("/webhook", "/webhook/sync") if "/sync" not in webhook_url else webhook_url
    try:
        resp = requests.post(
            sync_url,
            json={"content": content, "prompt": content, "sender_id": "adrecon-agent", "message_type": "query"},
            headers={"Content-Type": "application/json"},
            timeout=180,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "timeout":
                return "Agent timed out."
            return data.get("response", json.dumps(data)[:3000])
        return f"HTTP {resp.status_code}"
    except Exception as e:
        return f"Call failed: {e}"


@tool
def search_meta_ads(advertiser_name: str) -> str:
    """Search for a company's ads on Meta Ad Library (Facebook, Instagram). Returns ad copy, creative types, and active status."""
    try:
        results = _google(f"site:facebook.com/ads/library {advertiser_name} OR \"{advertiser_name}\" meta ad library")
        if not results:
            results = _google(f"{advertiser_name} facebook ads OR instagram ads OR meta ads examples")
        if not results:
            return f"No Meta ads found for {advertiser_name}"
        return _fmt(results)
    except Exception as e:
        return f"Meta ad search failed: {e}"


@tool
def search_google_ads(advertiser_name: str) -> str:
    """Search for a company's Google Ads — search ads, display ads, and YouTube ads via Google Ads Transparency Center."""
    try:
        results = _google(f"site:adstransparency.google.com {advertiser_name} OR \"{advertiser_name}\" google ads transparency")
        if not results:
            results = _google(f"{advertiser_name} google ads examples OR search ads OR display ads 2024 2025")
        if not results:
            return f"No Google ads found for {advertiser_name}"
        return _fmt(results)
    except Exception as e:
        return f"Google ads search failed: {e}"


@tool
def search_ad_strategy(company_name: str) -> str:
    """Search for analysis of a company's advertising strategy, creative approaches, and messaging frameworks. Covers ad spy tools, marketing blogs, and case studies."""
    try:
        results = _google(f"{company_name} advertising strategy OR ad creative OR marketing campaign OR ad copy analysis site:panoramata.co OR site:bigspy.com OR site:adbeat.com OR site:semrush.com")
        if not results:
            results = _google(f"{company_name} advertising strategy creative campaign 2024 2025")
        if not results:
            return f"No ad strategy analysis found for {company_name}"
        return _fmt(results)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def crawl_landing_page(url: str) -> str:
    """Crawl an ad's landing page to analyze messaging, value proposition, CTA, and conversion flow."""
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
        text = items[0].get("text", "")[:5000]
        title = items[0].get("metadata", {}).get("title", "")
        return f"### {title}\n{text}"
    except Exception as e:
        return f"Crawl failed: {e}"


@tool
def get_competitor_context(company_name: str) -> str:
    """Call Competitor Intel Agent via ZyndAI registry for competitive positioning context to enrich ad analysis."""
    try:
        resp = requests.get(f"{REGISTRY_URL}/agents", params={"keyword": "competitor intelligence", "limit": 3}, timeout=15)
        if resp.status_code != 200:
            return "Registry unavailable."
        agents = resp.json().get("data", [])
        for a in agents:
            if "competitor" in a.get("name", "").lower() and a.get("httpWebhookUrl"):
                return _call_zynd_agent(a["httpWebhookUrl"], f"Quick competitive overview of {company_name}")
        return "Competitor Intel Agent not found."
    except Exception as e:
        return f"Discovery failed: {e}"


SYSTEM_PROMPT = """You are AdRecon, an ad creative intelligence analyst. You monitor competitor advertising, deconstruct messaging strategies, and identify what's working.

Tools — use at most 3 per request:
- search_meta_ads: find competitor's Facebook/Instagram ads
- search_google_ads: find competitor's Google/YouTube ads
- search_ad_strategy: find analysis of ad strategy and creative approaches
- crawl_landing_page: analyze an ad's landing page for messaging/CTA
- get_competitor_context: call Competitor Intel Agent for competitive positioning (use sparingly)

Output format:

## Ad Intelligence: [Company]

### Active Campaigns
- Meta (FB/IG): [what's running]
- Google/YouTube: [what's running]
- Estimated spend: [if available]

### Creative Analysis
| Element | What They're Doing | Why It Works |
|---------|-------------------|-------------|
| Hook | [first line/visual] | [psychology] |
| Value prop | [main message] | [positioning] |
| CTA | [call to action] | [urgency/benefit] |
| Format | [video/image/carousel] | [engagement pattern] |

### Messaging Framework
- Primary angle: [pain point / aspiration / social proof / urgency]
- Tone: [professional / casual / provocative / educational]
- Target audience: [who the ads speak to]

### Landing Page Analysis
- Headline: [main headline]
- Conversion path: [steps to purchase/signup]
- Key objection handling: [how they address hesitations]

### What's Working
- [Patterns in their best-performing creative]

### Creative Recommendations
- [What you could do differently to compete]

Analyze ads like a performance marketing strategist, not just a reporter."""


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_meta_ads, search_google_ads, search_ad_strategy,
             crawl_landing_page, get_competitor_context]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6)


def run_server():
    agent_config = AgentConfig(
        name="AdRecon - Ad Creative Intelligence Agent",
        description="Ad creative intelligence — monitors competitor advertising on Meta, Google, "
        "and TikTok. Deconstructs messaging strategies, identifies winning creative formats, "
        "analyzes landing pages, and generates competitive creative briefs.",
        capabilities={
            "ai": ["nlp", "ad_analysis", "creative_intelligence", "langchain"],
            "protocols": ["http"],
            "services": ["ad_intelligence", "creative_analysis", "ad_monitoring",
                         "landing_page_analysis", "messaging_strategy"],
            "domains": ["marketing", "advertising", "growth", "performance_marketing"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5017,
        registry_url="https://registry.zynd.ai",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-adrecon",
        use_ngrok=False,
        webhook_url=get_webhook_url("adrecon"),
        sync_response_timeout=300,
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    zynd_agent.set_langchain_agent(create_agent())

    def handle_message(message: AgentMessage, topic: str):
        def _process():
            print(f"\n[AdRecon] query={message.content} from={message.sender_id}")
            try:
                response = zynd_agent.invoke(message.content, chat_history=[])
                zynd_agent.set_response(message.message_id, response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                zynd_agent.set_response(message.message_id, f"Error: {e}")
        threading.Thread(target=_process, daemon=True).start()

    zynd_agent.add_message_handler(handle_message)
    print(f"\nAdRecon Agent | {zynd_agent.webhook_url} | {zynd_agent.agent_id}")

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
