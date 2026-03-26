"""
TalentRadar Agent — ZyndAI Fleet #7

Talent intelligence — scrapes job boards for hiring signals, salary benchmarks,
hiring velocity, and skill demand trends. Hiring patterns = competitive strategy signals.

    cd /Users/sahil/work/zyndai/agents && uv run python talentradar_agent.py
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
            json={"content": content, "prompt": content, "sender_id": "talentradar-agent", "message_type": "query"},
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
def search_job_postings(query: str) -> str:
    """Search for job postings across LinkedIn, Indeed, Lever, Greenhouse, and Ashby. Use to find what roles a company is hiring for and detect hiring velocity."""
    try:
        results = _google(f"{query} site:linkedin.com/jobs OR site:indeed.com OR site:lever.co OR site:greenhouse.io OR site:ashbyhq.com OR site:jobs.lever.co")
        if not results:
            return "No job postings found."
        return _fmt(results)
    except Exception as e:
        return f"Job search failed: {e}"


@tool
def search_salary_data(role_and_location: str) -> str:
    """Search for salary benchmarks for a specific role. Covers Glassdoor, Levels.fyi, Payscale, and Salary.com."""
    try:
        results = _google(f"{role_and_location} salary OR compensation site:glassdoor.com OR site:levels.fyi OR site:payscale.com OR site:salary.com")
        if not results:
            return f"No salary data found for {role_and_location}"
        return _fmt(results)
    except Exception as e:
        return f"Salary search failed: {e}"


@tool
def search_hiring_news(company_name: str) -> str:
    """Search for hiring news, layoffs, team growth, and workforce changes for a company."""
    try:
        results = _google(f"{company_name} hiring OR layoffs OR team growth OR headcount OR workforce 2024 2025 2026")
        if not results:
            return f"No hiring news for {company_name}"
        return _fmt(results)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def search_tech_talent(skill_or_role: str) -> str:
    """Search for talent supply data — how many people have a specific skill, where they work, demand vs supply signals."""
    try:
        results = _google(f"{skill_or_role} talent shortage OR demand OR market OR engineers OR developers site:linkedin.com OR site:stackoverflow.com OR site:github.com")
        if not results:
            return f"No talent data for {skill_or_role}"
        return _fmt(results)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def get_company_context(company_name: str) -> str:
    """Call Lead Enrichment Agent via ZyndAI registry for company background — funding, size, industry context for hiring analysis."""
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


SYSTEM_PROMPT = """You are TalentRadar, a talent intelligence analyst. You analyze hiring patterns as competitive intelligence signals — who's hiring, what roles, at what salary, and what it reveals about company strategy.

Tools — use at most 3 per request:
- search_job_postings: find open roles at a company or in a skill area
- search_salary_data: salary benchmarks for specific roles
- search_hiring_news: news about hiring, layoffs, team changes
- search_tech_talent: talent supply/demand data for skills
- get_company_context: call Lead Enrichment Agent for company background (use sparingly)

Output format:

## Talent Intelligence: [Company/Role/Skill]

### Hiring Velocity
- Open roles: [count by category]
- Hot areas: [which departments are growing fastest]
- Trend: Accelerating / Steady / Slowing

### Role Breakdown
| Category | Roles | Signal |
|----------|-------|--------|
| Engineering | [count] | [what it means] |
| Sales | [count] | [what it means] |
| Product | [count] | [what it means] |

### Salary Intelligence
| Role | Range | Market Position |
|------|-------|----------------|
| [role] | [$X-$Y] | Above/At/Below market |

### Strategic Read
[What hiring patterns reveal about company direction — new product bets, market expansion, cost-cutting, pivot signals]

### Talent Market Context
- Supply: [talent availability for key roles]
- Competition: [who else is hiring for same skills]
- Trend: [skill demand rising/falling]

### Recommendations
- [Actionable insights]

Interpret hiring data as strategy signals. A company hiring 20 ML engineers means something different than one hiring 20 sales reps."""


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_job_postings, search_salary_data, search_hiring_news,
             search_tech_talent, get_company_context]

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
        name="TalentRadar - Talent Intelligence Agent",
        description="Talent intelligence — scrapes job boards for hiring signals, salary benchmarks, "
        "hiring velocity, and skill demand trends. Interprets hiring patterns as competitive "
        "strategy signals. Cross-calls Lead Enrichment Agent for company context.",
        capabilities={
            "ai": ["nlp", "talent_analytics", "competitive_intelligence", "langchain"],
            "protocols": ["http"],
            "services": ["talent_intelligence", "hiring_analysis", "salary_benchmarking",
                         "workforce_analytics", "skill_demand_tracking"],
            "domains": ["human_resources", "recruiting", "business_intelligence", "strategy"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5016,
        registry_url="https://registry.zynd.ai",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-talentradar",
        use_ngrok=False,
        webhook_url=get_webhook_url("talentradar"),
        sync_response_timeout=300,
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    zynd_agent.set_langchain_agent(create_agent())

    def handle_message(message: AgentMessage, topic: str):
        def _process():
            print(f"\n[TalentRadar] query={message.content} from={message.sender_id}")
            try:
                response = zynd_agent.invoke(message.content, chat_history=[])
                zynd_agent.set_response(message.message_id, response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                zynd_agent.set_response(message.message_id, f"Error: {e}")
        threading.Thread(target=_process, daemon=True).start()

    zynd_agent.add_message_handler(handle_message)
    print(f"\nTalentRadar Agent | {zynd_agent.webhook_url} | {zynd_agent.agent_id}")

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
