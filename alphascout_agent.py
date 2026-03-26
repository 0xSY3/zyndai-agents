"""
AlphaScout Agent — ZyndAI Fleet #9

Financial signals — monitors SEC filings, insider trading, funding rounds, M&A signals.
Extracts financials, detects material events, cross-references with news sentiment.

    cd /Users/sahil/work/zyndai/agents && uv run python alphascout_agent.py
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
            json={"content": content, "prompt": content, "sender_id": "alphascout-agent", "message_type": "query"},
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
def search_sec_filings(company_name: str) -> str:
    """Search for SEC filings (10-K, 10-Q, 8-K, proxy statements) for a public company. Covers EDGAR, SEC.gov, and financial data sites."""
    try:
        results = _google(f"{company_name} SEC filing 10-K OR 10-Q OR 8-K site:sec.gov OR site:last10k.com OR site:bamsec.com")
        if not results:
            results = _google(f"{company_name} annual report financial statements SEC")
        if not results:
            return f"No SEC filings found for {company_name}"
        return _fmt(results)
    except Exception as e:
        return f"SEC search failed: {e}"


@tool
def search_funding_and_valuation(company_name: str) -> str:
    """Search for funding rounds, valuations, investors, and M&A activity for a company. Covers Crunchbase, PitchBook proxies, and funding news."""
    try:
        results = _google(f"{company_name} funding round OR valuation OR acquisition OR series OR raised site:crunchbase.com OR site:techcrunch.com OR site:pitchbook.com")
        if not results:
            return f"No funding data found for {company_name}"
        return _fmt(results)
    except Exception as e:
        return f"Funding search failed: {e}"


@tool
def search_insider_trading(company_name: str) -> str:
    """Search for insider trading activity — Form 4 filings, executive stock purchases/sales, and insider transaction patterns."""
    try:
        results = _google(f"{company_name} insider trading OR insider buying OR insider selling OR Form 4 site:openinsider.com OR site:secform4.com OR site:finviz.com")
        if not results:
            return f"No insider trading data for {company_name}"
        return _fmt(results)
    except Exception as e:
        return f"Insider search failed: {e}"


@tool
def search_financial_news(company_name: str) -> str:
    """Search for financial news, earnings reports, analyst ratings, and market commentary for a company."""
    try:
        results = _google(f"{company_name} earnings OR revenue OR financial results OR analyst rating site:bloomberg.com OR site:reuters.com OR site:seekingalpha.com OR site:yahoo.com/finance")
        if not results:
            return f"No financial news for {company_name}"
        return _fmt(results)
    except Exception as e:
        return f"Financial news search failed: {e}"


@tool
def crawl_financial_page(url: str) -> str:
    """Crawl a specific financial document, filing, or analysis page for detailed data extraction."""
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


SYSTEM_PROMPT = """You are AlphaScout, a financial signals analyst. You monitor SEC filings, insider trading, funding rounds, and financial news to detect material events and investment signals.

Tools — use at most 3 per request:
- search_sec_filings: SEC EDGAR filings (10-K, 10-Q, 8-K)
- search_funding_and_valuation: funding rounds, valuations, M&A
- search_insider_trading: insider buying/selling patterns
- search_financial_news: earnings, analyst ratings, market commentary
- crawl_financial_page: deep-dive a specific financial document

Output format:

## Financial Intelligence: [Company]

### Company Financial Profile
- Public/Private: [status]
- Market Cap / Valuation: [$X]
- Revenue: [$X] (growth: X%)
- Last funding: [Series X, $Xm, date]

### Recent Filings & Events
- [Material events from 8-K filings]
- [Key numbers from latest 10-Q/10-K]
- [Leadership changes]

### Insider Activity
| Insider | Role | Action | Shares | Date |
|---------|------|--------|--------|------|
| [name] | [title] | Buy/Sell | [count] | [date] |
- Signal: [bullish/bearish/neutral based on patterns]

### Funding & M&A
- Recent rounds: [details]
- Acqui-hire/acquisition signals: [if any]
- Investor sentiment: [follow-on rounds = confidence]

### Financial Health Indicators
- Burn rate: [if available]
- Cash position: [if available]
- Debt: [if available]
- Profitability: [profitable / path to profit / burning]

### Risk Factors
- [Key risks from filings or analysis]

### Signal Summary
[1-2 paragraph synthesis — is this company growing, stable, or struggling? What does the financial data suggest about next 6-12 months?]

Be precise with numbers. Cite filing dates. Flag when data is estimated vs confirmed."""


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_sec_filings, search_funding_and_valuation, search_insider_trading,
             search_financial_news, crawl_financial_page]

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
        name="AlphaScout - Financial Signals Agent",
        description="Financial signals intelligence — monitors SEC filings, insider trading, "
        "funding rounds, M&A activity, and financial news. Detects material events, "
        "analyzes financial health, and produces investment research briefs.",
        capabilities={
            "ai": ["nlp", "financial_analysis", "alternative_data", "langchain"],
            "protocols": ["http"],
            "services": ["financial_intelligence", "sec_filing_analysis", "insider_trading_monitoring",
                         "funding_tracking", "financial_news_analysis", "investment_research"],
            "domains": ["finance", "investing", "alternative_data", "business_intelligence"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5018,
        registry_url="https://registry.zynd.ai",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-alphascout",
        use_ngrok=False,
        webhook_url=get_webhook_url("alphascout"),
        sync_response_timeout=300,
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    zynd_agent.set_langchain_agent(create_agent())

    def handle_message(message: AgentMessage, topic: str):
        def _process():
            print(f"\n[AlphaScout] query={message.content} from={message.sender_id}")
            try:
                response = zynd_agent.invoke(message.content, chat_history=[])
                zynd_agent.set_response(message.message_id, response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                zynd_agent.set_response(message.message_id, f"Error: {e}")
        threading.Thread(target=_process, daemon=True).start()

    zynd_agent.add_message_handler(handle_message)
    print(f"\nAlphaScout Agent | {zynd_agent.webhook_url} | {zynd_agent.agent_id}")

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
