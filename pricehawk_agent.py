"""
PriceHawk Agent — ZyndAI Fleet #6

Pricing intelligence — monitors competitor pricing across Amazon, Google Shopping,
and e-commerce sites. Detects price changes, MAP violations, dynamic pricing patterns.

    cd /Users/sahil/work/zyndai/agents && uv run python pricehawk_agent.py
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
            json={"content": content, "prompt": content, "sender_id": "pricehawk-agent", "message_type": "query"},
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
def search_product_pricing(query: str) -> str:
    """Search Google Shopping and e-commerce sites for product pricing. Use for finding current prices, comparing across retailers, and detecting pricing patterns."""
    try:
        results = _google(f"{query} price OR pricing OR buy OR shop site:amazon.com OR site:walmart.com OR site:bestbuy.com OR site:target.com")
        if not results:
            return "No pricing data found."
        return _fmt(results)
    except Exception as e:
        return f"Pricing search failed: {e}"


@tool
def search_competitor_pricing_pages(company_or_product: str) -> str:
    """Search for a competitor's pricing page, plans, and pricing history. Finds SaaS pricing pages, changelog pricing updates, and pricing discussions."""
    try:
        results = _google(f"{company_or_product} pricing plans OR {company_or_product} pricing page OR {company_or_product} how much cost")
        if not results:
            return f"No pricing pages found for {company_or_product}"
        return _fmt(results)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def crawl_pricing_page(url: str) -> str:
    """Crawl a specific pricing page to extract plan details, features, and price points."""
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
def search_price_comparison(product_name: str) -> str:
    """Search for price comparison data, review sites, and deal trackers for a specific product or service category."""
    try:
        results = _google(f"{product_name} price comparison OR review OR vs OR alternative site:g2.com OR site:capterra.com OR site:trustradius.com OR site:pcmag.com")
        if not results:
            return f"No comparison data found for {product_name}"
        return _fmt(results)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def get_competitor_context(company_name: str) -> str:
    """Call Lead Enrichment Agent via ZyndAI registry for company background to contextualize pricing data."""
    try:
        resp = requests.get(f"{REGISTRY_URL}/agents", params={"keyword": "lead enrichment", "limit": 3}, timeout=15)
        if resp.status_code != 200:
            return "Registry unavailable."
        agents = resp.json().get("data", [])
        for a in agents:
            if "enrichment" in a.get("name", "").lower() and a.get("httpWebhookUrl"):
                return _call_zynd_agent(a["httpWebhookUrl"], f"Enrich: {company_name}")
        return "Lead Enrichment Agent not found."
    except Exception as e:
        return f"Discovery failed: {e}"


SYSTEM_PROMPT = """You are PriceHawk, a pricing intelligence analyst. You monitor competitor pricing, detect changes, analyze pricing strategies, and provide actionable pricing recommendations.

Tools — use at most 3 per request:
- search_product_pricing: find current retail/e-commerce prices for products
- search_competitor_pricing_pages: find SaaS/service pricing pages and plans
- crawl_pricing_page: deep-dive a specific pricing URL
- search_price_comparison: find comparison/review data with pricing context
- get_competitor_context: call Lead Enrichment Agent for company background (use sparingly)

Output format:

## Pricing Intelligence: [Product/Company]

### Current Pricing
| Plan/SKU | Price | Key Features |
|----------|-------|-------------|
| [tier] | [price] | [features] |

### Competitor Comparison
| Competitor | Price | Positioning |
|------------|-------|------------|
| [name] | [price] | [cheap/mid/premium] |

### Pricing Strategy Analysis
- Model: [freemium/tiered/usage-based/flat/per-seat]
- Positioning: [budget/mid-market/enterprise]
- Recent changes: [any detected price moves]

### Market Context
- Industry avg: [if available]
- Price trend: [rising/stable/falling]

### Recommendations
- [Actionable pricing insights]

Be specific with dollar amounts. Cite sources."""


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_product_pricing, search_competitor_pricing_pages, crawl_pricing_page,
             search_price_comparison, get_competitor_context]

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
        name="PriceHawk - Pricing Intelligence Agent",
        description="Pricing intelligence — monitors competitor pricing across Amazon, Google Shopping, "
        "SaaS pricing pages, and e-commerce sites. Detects price changes, compares plans, "
        "analyzes pricing strategies, and provides actionable recommendations.",
        capabilities={
            "ai": ["nlp", "pricing_analysis", "competitive_intelligence", "langchain"],
            "protocols": ["http"],
            "services": ["pricing_intelligence", "price_monitoring", "competitor_pricing",
                         "price_comparison", "pricing_strategy_analysis"],
            "domains": ["e_commerce", "saas", "business_intelligence", "market_analysis"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5015,
        registry_url="https://registry.zynd.ai",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-pricehawk",
        use_ngrok=False,
        webhook_url=get_webhook_url("pricehawk"),
        sync_response_timeout=300,
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    zynd_agent.set_langchain_agent(create_agent())

    def handle_message(message: AgentMessage, topic: str):
        def _process():
            print(f"\n[PriceHawk] query={message.content} from={message.sender_id}")
            try:
                response = zynd_agent.invoke(message.content, chat_history=[])
                zynd_agent.set_response(message.message_id, response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                zynd_agent.set_response(message.message_id, f"Error: {e}")
        threading.Thread(target=_process, daemon=True).start()

    zynd_agent.add_message_handler(handle_message)
    print(f"\nPriceHawk Agent | {zynd_agent.webhook_url} | {zynd_agent.agent_id}")

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
