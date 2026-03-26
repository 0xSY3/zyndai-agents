"""
Orchestrator Agent — ZyndAI Fleet #5

The brain of the fleet. Discovers specialized agents via ZyndAI registry,
delegates tasks, calls them in parallel, and synthesizes executive briefings.

Does NOT hardcode agent URLs — discovers them dynamically from the registry.

    cd /Users/sahil/work/zyndai/agents && uv run python orchestrator_agent.py
"""

from zyndai_agent.agent import AgentConfig, ZyndAIAgent
from zyndai_agent.message import AgentMessage
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from concurrent.futures import ThreadPoolExecutor, as_completed
from deploy_config import get_webhook_url
from dotenv import load_dotenv
import os
import json
import time
import signal
import threading
import requests

load_dotenv()

REGISTRY_URL = "https://registry.zynd.ai"

_agent_cache: dict[str, dict] = {}


def _discover_agent(keyword: str) -> dict | None:
    if keyword in _agent_cache:
        return _agent_cache[keyword]
    try:
        resp = requests.get(f"{REGISTRY_URL}/agents", params={"keyword": keyword, "limit": 5}, timeout=15)
        if resp.status_code != 200:
            return None
        agents = resp.json().get("data", [])
        for a in agents:
            if a.get("httpWebhookUrl"):
                _agent_cache[keyword] = a
                return a
        return None
    except Exception:
        return None


def _call_zynd_agent(agent: dict, message_content: str, timeout: int = 180) -> str:
    webhook = agent.get("httpWebhookUrl", "")
    sync_url = webhook.rstrip("/")
    if not sync_url.endswith("/sync"):
        sync_url = sync_url.replace("/webhook", "/webhook/sync")
    try:
        resp = requests.post(
            sync_url,
            json={"content": message_content, "prompt": message_content,
                   "sender_id": "orchestrator-agent", "message_type": "query"},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "timeout":
                return f"Agent '{agent.get('name')}' timed out."
            return data.get("response", json.dumps(data)[:3000])
        return f"Agent '{agent.get('name')}' returned HTTP {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return f"Agent '{agent.get('name')}' not reachable at {webhook}"
    except requests.exceptions.Timeout:
        return f"Agent '{agent.get('name')}' did not respond within {timeout}s."
    except Exception as e:
        return f"Call to '{agent.get('name')}' failed: {e}"


@tool
def call_lead_enrichment(query: str) -> str:
    """Discover and call Lead Enrichment Agent — get a full company/person dossier (website, social, tech stack, funding)."""
    agent = _discover_agent("lead enrichment")
    if not agent:
        return "Lead Enrichment Agent not found on ZyndAI registry."
    return _call_zynd_agent(agent, f"Enrich: {query}")


@tool
def call_competitor_intel(query: str) -> str:
    """Discover and call Competitor Intel Agent — get competitive analysis (pricing, features, hiring, sentiment)."""
    agent = _discover_agent("competitor intelligence")
    if not agent:
        return "Competitor Intel Agent not found on ZyndAI registry."
    return _call_zynd_agent(agent, query)


@tool
def call_content_pipeline(query: str) -> str:
    """Discover and call Content Pipeline Agent — repurpose YouTube/blog into tweet thread, LinkedIn post, newsletter."""
    agent = _discover_agent("content pipeline repurpose")
    if not agent:
        return "Content Pipeline Agent not found on ZyndAI registry."
    return _call_zynd_agent(agent, query)


@tool
def call_reputation_monitor(query: str) -> str:
    """Discover and call Reputation Monitor Agent — brand monitoring (news, Reddit, HN, reviews, Twitter sentiment)."""
    agent = _discover_agent("reputation monitor brand")
    if not agent:
        return "Reputation Monitor Agent not found on ZyndAI registry."
    return _call_zynd_agent(agent, query)


@tool
def call_agents_parallel(tasks_json: str) -> str:
    """Call multiple fleet agents in parallel. Input: JSON array e.g. [{"agent": "lead enrichment", "query": "Stripe"}, {"agent": "reputation monitor", "query": "Stripe"}]"""
    try:
        tasks = json.loads(tasks_json)
    except json.JSONDecodeError:
        return "Invalid JSON. Provide a list of {agent, query} objects."

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for task in tasks:
            keyword = task.get("agent", "")
            query = task.get("query", "")
            agent = _discover_agent(keyword)
            if not agent:
                results[keyword] = f"Agent '{keyword}' not found on registry."
                continue
            futures[pool.submit(_call_zynd_agent, agent, query)] = keyword

        for future in as_completed(futures, timeout=300):
            keyword = futures[future]
            try:
                results[keyword] = future.result()
            except Exception as e:
                results[keyword] = f"Error: {e}"

    sections = []
    for name, result in results.items():
        sections.append(f"## {name.replace('_', ' ').title()}\n\n{result}")
    return "\n\n---\n\n".join(sections)


@tool
def search_zynd_registry(keyword: str) -> str:
    """Search the ZyndAI registry for any agents matching a keyword. Returns names, descriptions, and webhook URLs."""
    try:
        resp = requests.get(f"{REGISTRY_URL}/agents", params={"keyword": keyword, "limit": 10}, timeout=15)
        if resp.status_code != 200:
            return f"Registry search failed: HTTP {resp.status_code}"
        agents = resp.json().get("data", [])
        if not agents:
            return f"No agents found for '{keyword}'"
        lines = []
        for a in agents:
            status = "ONLINE" if a.get("httpWebhookUrl") else "OFFLINE"
            lines.append(f"- {a.get('name', 'Unknown')} [{status}]\n  {a.get('description', 'N/A')[:150]}\n  Webhook: {a.get('httpWebhookUrl', 'none')}")
        return f"Found {len(agents)} agents:\n\n" + "\n\n".join(lines)
    except Exception as e:
        return f"Registry search failed: {e}"


SYSTEM_PROMPT = """You are the ZyndAI Orchestrator — executive coordinator of an AI agent fleet.

You discover agents dynamically from the ZyndAI registry. Available specialized agents:
1. Lead Enrichment — deep company/person dossiers
2. Competitor Intel — competitive analysis
3. Content Pipeline — content repurposing
4. Reputation Monitor — brand monitoring

You also have search_zynd_registry to find ANY agent on the network.

Your job:
- Analyze the request, determine which agents to involve
- For multi-agent requests, use call_agents_parallel for speed
- Synthesize outputs into a unified executive briefing

Common patterns:
- "Morning briefing on X" → call_agents_parallel with lead_enrichment + competitor_intel + reputation_monitor
- "Full analysis of X" → parallel call to all relevant agents
- "Repurpose and analyze" → content_pipeline + competitor_intel

Output for multi-agent briefings:

# Executive Briefing: [Topic]

## Key Takeaways
- [Top 3-5 actionable insights across all reports]

## Detailed Analysis
[Synthesized narrative from all agents]

## Risk Factors
- [Cross-referenced risks]

## Recommended Actions
1. [Prioritized items]"""


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [call_lead_enrichment, call_competitor_intel, call_content_pipeline,
             call_reputation_monitor, call_agents_parallel, search_zynd_registry]

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
        name="Orchestrator Agent",
        description="Executive coordinator of the ZyndAI agent fleet — discovers specialized agents "
        "via the ZyndAI registry, delegates tasks, calls them in parallel, and synthesizes "
        "unified executive briefings. The single entry point for multi-agent workflows.",
        capabilities={
            "ai": ["nlp", "agent_orchestration", "synthesis", "langchain"],
            "protocols": ["http"],
            "services": ["executive_briefing", "multi_agent_orchestration", "morning_briefing",
                         "strategic_analysis", "comprehensive_research", "agent_discovery"],
            "domains": ["business_intelligence", "strategy", "executive", "operations"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5014,
        registry_url="https://registry.zynd.ai",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-orchestrator",
        use_ngrok=False,
        webhook_url=get_webhook_url("orchestrator"),
        sync_response_timeout=300,
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    zynd_agent.set_langchain_agent(create_agent())

    def handle_message(message: AgentMessage, topic: str):
        def _process():
            print(f"\n[Orchestrator] query={message.content} from={message.sender_id}")
            try:
                response = zynd_agent.invoke(message.content, chat_history=[])
                zynd_agent.set_response(message.message_id, response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                zynd_agent.set_response(message.message_id, f"Error: {e}")
        threading.Thread(target=_process, daemon=True).start()

    zynd_agent.add_message_handler(handle_message)
    print(f"\nOrchestrator Agent | {zynd_agent.webhook_url} | {zynd_agent.agent_id}")
    print("Discovers fleet agents via ZyndAI registry at runtime")

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
