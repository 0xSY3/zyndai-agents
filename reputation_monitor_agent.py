"""
Reputation Monitor Agent — ZyndAI Fleet #4

Monitors brand/person reputation across news, Reddit, Hacker News, review sites, Twitter.
Discovers Lead Enrichment Agent via ZyndAI registry for entity context.

    cd /Users/sahil/work/zyndai/agents && uv run python reputation_monitor_agent.py
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
        url = r.get("url", "")
        platform = "Web"
        for domain, label in [("reddit.com", "Reddit"), ("news.ycombinator.com", "HN"),
                              ("twitter.com", "Twitter"), ("x.com", "Twitter"),
                              ("g2.com", "G2"), ("trustpilot.com", "Trustpilot"),
                              ("capterra.com", "Capterra"), ("producthunt.com", "ProductHunt")]:
            if domain in url:
                platform = label
                break
        lines.append(f"- [{platform}] {r.get('title', 'N/A')}\n  {r.get('description', 'N/A')}\n  {url}")
    return "\n\n".join(lines) if lines else "No results."


def _call_zynd_agent(webhook_url: str, message_content: str) -> str:
    sync_url = webhook_url.rstrip("/")
    if not sync_url.endswith("/sync"):
        sync_url = sync_url.replace("/webhook", "/webhook/sync")
    try:
        resp = requests.post(
            sync_url,
            json={"content": message_content, "prompt": message_content,
                   "sender_id": "reputation-monitor-agent", "message_type": "query"},
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
def search_news(entity_name: str) -> str:
    """Search for recent news mentions of a brand, company, or person."""
    try:
        return _fmt(_google(f"{entity_name} news"))
    except Exception as e:
        return f"News search failed: {e}"


@tool
def search_community_and_reviews(entity_name: str) -> str:
    """Search Reddit, Hacker News, G2, Trustpilot, and ProductHunt for discussions and reviews in one call."""
    try:
        return _fmt(_google(
            f"{entity_name} site:reddit.com OR site:news.ycombinator.com OR site:g2.com OR site:trustpilot.com OR site:producthunt.com"
        ))
    except Exception as e:
        return f"Community search failed: {e}"


@tool
def scrape_twitter(entity_name: str) -> str:
    """Scrape live tweets from Twitter/X mentioning a brand or person. Returns actual tweet text, engagement counts, dates, and authors."""
    try:
        run = apify.actor("apidojo/tweet-scraper").call(
            run_input={
                "searchTerms": [entity_name],
                "maxTweets": 20,
                "sort": "Latest",
            },
            timeout_secs=120,
        )
        items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
        if not items:
            return f"No tweets found mentioning {entity_name}"
        lines = []
        for t in items[:20]:
            author = t.get("author", {}).get("userName", t.get("user", {}).get("screen_name", "unknown"))
            text = t.get("text", t.get("full_text", ""))[:280]
            likes = t.get("likeCount", t.get("favorite_count", 0))
            retweets = t.get("retweetCount", t.get("retweet_count", 0))
            date = t.get("createdAt", t.get("created_at", ""))[:10]
            lines.append(f"- @{author} ({date}) [{likes} likes, {retweets} RTs]\n  {text}")
        return f"Live tweets mentioning '{entity_name}':\n\n" + "\n\n".join(lines)
    except Exception as e:
        return f"Twitter scrape failed: {e}"


@tool
def get_entity_background(entity_name: str) -> str:
    """Discover and call Lead Enrichment Agent via ZyndAI registry for deep background on the entity."""
    try:
        resp = requests.get(
            f"{REGISTRY_URL}/agents",
            params={"keyword": "lead enrichment", "limit": 3},
            timeout=15,
        )
        if resp.status_code != 200:
            return f"Registry search failed: HTTP {resp.status_code}"
        agents = resp.json().get("data", [])
        for a in agents:
            if "enrichment" in a.get("name", "").lower() and a.get("httpWebhookUrl"):
                return _call_zynd_agent(a["httpWebhookUrl"], f"Enrich: {entity_name}")
        return "Lead Enrichment Agent not found on registry."
    except Exception as e:
        return f"Discovery failed: {e}"


SYSTEM_PROMPT = """You are a reputation monitoring analyst. Given a brand, company, or person, scan the internet and produce a reputation report with sentiment analysis.

IMPORTANT: Be efficient. You have 4 tools — use at most 3 per request:
- search_news: recent news coverage
- search_community_and_reviews: Reddit + HN + G2 + Trustpilot + ProductHunt in ONE call
- scrape_twitter: LIVE tweets with engagement data (likes, RTs, dates, authors) — real-time sentiment
- get_entity_background: deep background via Lead Enrichment Agent (use sparingly)

Typical flow: search_news + search_community_and_reviews. Add search_twitter only if social sentiment is specifically needed.

Output format:

## Reputation Report: [Entity]
Overall Sentiment: Positive / Mixed / Negative (confidence %)

### Executive Summary
[3-4 sentences on current reputation]

### News Coverage
Sentiment: [Positive/Mixed/Negative]
- [Key headlines]

### Community & Reviews
Sentiment: [Positive/Mixed/Negative]
- Reddit: [themes]
- HN: [themes]
- G2/Trustpilot: [ratings if found]

### Social (Twitter/X)
Sentiment: [Positive/Mixed/Negative]
- [Trends]

### Risks
- [List]

### Strengths
- [List]

### Recommendations
- [Actionable items]

Cite sources. Mark unavailable data clearly."""


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_news, search_community_and_reviews, scrape_twitter, get_entity_background]

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
        name="Reputation Monitor Agent",
        description="Brand and reputation monitoring — scans news, Reddit, Hacker News, "
        "review sites, and Twitter/X. Produces sentiment analysis and reputation reports. "
        "Discovers Lead Enrichment Agent via ZyndAI registry for entity context.",
        capabilities={
            "ai": ["nlp", "sentiment_analysis", "reputation_monitoring", "langchain"],
            "protocols": ["http"],
            "services": ["reputation_monitoring", "brand_monitoring", "sentiment_analysis",
                         "news_monitoring", "social_listening", "review_aggregation"],
            "domains": ["public_relations", "brand_management", "marketing", "risk_management"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5013,
        registry_url="https://registry.zynd.ai",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-reputation-monitor",
        use_ngrok=False,
        webhook_url=get_webhook_url("reputation-monitor"),
        sync_response_timeout=300,
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    zynd_agent.set_langchain_agent(create_agent())

    def handle_message(message: AgentMessage, topic: str):
        def _process():
            print(f"\n[Reputation Monitor] query={message.content} from={message.sender_id}")
            try:
                response = zynd_agent.invoke(message.content, chat_history=[])
                zynd_agent.set_response(message.message_id, response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                zynd_agent.set_response(message.message_id, f"Error: {e}")
        threading.Thread(target=_process, daemon=True).start()

    zynd_agent.add_message_handler(handle_message)
    print(f"\nReputation Monitor Agent | {zynd_agent.webhook_url} | {zynd_agent.agent_id}")

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
