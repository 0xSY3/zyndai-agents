"""
DealFlow Agent — ZyndAI Fleet #10

Opportunity scout — monitors government RFPs (SAM.gov), grants, commercial opportunities,
and startup funding. Matches opportunities against company profile, scores by fit.

    cd /Users/sahil/work/zyndai/agents && uv run python dealflow_agent.py
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
            json={"content": content, "prompt": content, "sender_id": "dealflow-agent", "message_type": "query"},
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
def search_government_rfps(keywords: str) -> str:
    """Search for government RFPs, contracts, and procurement opportunities on SAM.gov and state/federal portals."""
    try:
        results = _google(f"{keywords} RFP OR solicitation OR contract site:sam.gov OR site:grants.gov OR site:fbo.gov OR government procurement")
        if not results:
            return f"No government RFPs found for: {keywords}"
        return _fmt(results)
    except Exception as e:
        return f"RFP search failed: {e}"


@tool
def search_grants(keywords: str) -> str:
    """Search for grants, funding opportunities, and awards across federal, state, and foundation sources."""
    try:
        results = _google(f"{keywords} grant OR funding opportunity OR award site:grants.gov OR site:nsf.gov OR site:nih.gov OR foundation grant")
        if not results:
            return f"No grants found for: {keywords}"
        return _fmt(results)
    except Exception as e:
        return f"Grant search failed: {e}"


@tool
def search_commercial_opportunities(keywords: str) -> str:
    """Search for commercial RFPs, vendor opportunities, partnership requests, and business opportunities."""
    try:
        results = _google(f"{keywords} RFP OR vendor OR partner OR supplier opportunity OR request for proposal 2024 2025 2026")
        if not results:
            return f"No commercial opportunities found for: {keywords}"
        return _fmt(results)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def crawl_opportunity_page(url: str) -> str:
    """Crawl a specific RFP, grant, or opportunity page to extract requirements, deadlines, budget, and eligibility details."""
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
def get_issuer_intel(organization_name: str) -> str:
    """Call Lead Enrichment Agent via ZyndAI registry to research the organization issuing an RFP/grant — understand their budget, priorities, and past awards."""
    try:
        resp = requests.get(f"{REGISTRY_URL}/agents", params={"keyword": "lead enrichment", "limit": 3}, timeout=15)
        if resp.status_code != 200:
            return "Registry unavailable."
        agents = resp.json().get("data", [])
        for a in agents:
            if "enrichment" in a.get("name", "").lower() and a.get("httpWebhookUrl"):
                return _call_zynd_agent(a["httpWebhookUrl"], f"Enrich: {organization_name}")
        return "Lead Enrichment Agent not found."
    except Exception as e:
        return f"Discovery failed: {e}"


SYSTEM_PROMPT = """You are DealFlow, an opportunity scout. You find and qualify business opportunities — government RFPs, grants, commercial contracts, and partnership opportunities.

Tools — use at most 3 per request:
- search_government_rfps: federal/state government contracts and solicitations
- search_grants: grants from government, NSF, NIH, foundations
- search_commercial_opportunities: commercial RFPs, vendor requests, partnerships
- crawl_opportunity_page: deep-dive a specific opportunity for requirements/deadlines
- get_issuer_intel: call Lead Enrichment Agent to research the issuing organization (use sparingly)

Output format:

## Opportunity Report: [Search Topic]

### Opportunities Found

#### 1. [Opportunity Title]
- **Source**: [SAM.gov / grants.gov / commercial]
- **Issuer**: [organization]
- **Value**: [$X estimated]
- **Deadline**: [date]
- **Type**: [RFP / Grant / Contract / Partnership]
- **Requirements**: [key eligibility/capability requirements]
- **Fit Score**: [High/Medium/Low] — [why]
- **URL**: [link]

#### 2. [Next opportunity...]

### Market Context
- Active opportunities in this space: [count/trend]
- Competition density: [high/medium/low]
- Budget trends: [growing/stable/shrinking]

### Recommendations
- **Best fit**: [which opportunity to pursue first and why]
- **Preparation needed**: [what capabilities or certifications to highlight]
- **Timeline**: [key deadlines and milestones]

Score opportunities by fit. Be specific about deadlines and dollar amounts. Flag expired opportunities clearly."""


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_government_rfps, search_grants, search_commercial_opportunities,
             crawl_opportunity_page, get_issuer_intel]

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
        name="DealFlow - Opportunity Scout Agent",
        description="Opportunity scout — monitors government RFPs (SAM.gov), grants (grants.gov, NSF, NIH), "
        "commercial contracts, and partnership opportunities. Matches against company capabilities, "
        "scores by fit, and provides deadline-aware recommendations.",
        capabilities={
            "ai": ["nlp", "opportunity_matching", "government_contracting", "langchain"],
            "protocols": ["http"],
            "services": ["rfp_monitoring", "grant_discovery", "opportunity_scoring",
                         "procurement_intelligence", "business_development"],
            "domains": ["government_contracting", "grants", "business_development", "sales"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5019,
        registry_url="https://registry.zynd.ai",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-dealflow",
        use_ngrok=False,
        webhook_url=get_webhook_url("dealflow"),
        sync_response_timeout=300,
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    zynd_agent.set_langchain_agent(create_agent())

    def handle_message(message: AgentMessage, topic: str):
        def _process():
            print(f"\n[DealFlow] query={message.content} from={message.sender_id}")
            try:
                response = zynd_agent.invoke(message.content, chat_history=[])
                zynd_agent.set_response(message.message_id, response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                zynd_agent.set_response(message.message_id, f"Error: {e}")
        threading.Thread(target=_process, daemon=True).start()

    zynd_agent.add_message_handler(handle_message)
    print(f"\nDealFlow Agent | {zynd_agent.webhook_url} | {zynd_agent.agent_id}")

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
