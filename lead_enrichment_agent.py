"""
Lead Enrichment Agent — ZyndAI Fleet #1

Takes a company or person name → full enrichment dossier.
Foundation agent — other fleet agents cross-call this one via ZyndAI webhooks.

    cd /Users/sahil/work/zyndai/agents && uv run python lead_enrichment_agent.py
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
import time
import signal
import threading

load_dotenv()

apify = ApifyClient(os.environ["APIFY_API_TOKEN"])


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


@tool
def google_search(query: str) -> str:
    """Search Google for information about a company or person. Returns top results with titles, descriptions, and URLs."""
    try:
        return _fmt(_google(query))
    except Exception as e:
        return f"Search failed: {e}"


@tool
def crawl_website(url: str) -> str:
    """Crawl a website and extract its main text content. Use for company website deep-dives."""
    try:
        run = apify.actor("apify/website-content-crawler").call(
            run_input={
                "startUrls": [{"url": url}],
                "maxCrawlPages": 3,
                "maxCrawlDepth": 1,
                "crawlerType": "cheerio",
            },
            timeout_secs=120,
        )
        items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
        if not items:
            return f"No content extracted from {url}"
        pages = []
        for item in items[:3]:
            title = item.get("metadata", {}).get("title", item.get("url", "Unknown"))
            text = item.get("text", "")[:2000]
            pages.append(f"### {title}\n{text}")
        return "\n\n---\n\n".join(pages)
    except Exception as e:
        return f"Crawl failed: {e}"


@tool
def search_social_and_tech(name: str) -> str:
    """Find social profiles AND tech stack info for a company or person in one search. Covers LinkedIn, Twitter, GitHub, Crunchbase, StackShare, BuiltWith."""
    try:
        results = _google(
            f"{name} site:linkedin.com OR site:twitter.com OR site:github.com OR site:crunchbase.com OR site:stackshare.io OR site:builtwith.com"
        )
        if not results:
            return f"No profiles or tech info found for {name}"
        lines = []
        for r in results[:12]:
            url = r.get("url", "")
            platform = "Web"
            for domain, label in [("linkedin.com", "LinkedIn"), ("twitter.com", "Twitter/X"),
                                  ("x.com", "Twitter/X"), ("github.com", "GitHub"),
                                  ("crunchbase.com", "Crunchbase"), ("stackshare.io", "StackShare"),
                                  ("builtwith.com", "BuiltWith")]:
                if domain in url:
                    platform = label
                    break
            lines.append(f"- [{platform}] {r.get('title', 'N/A')}\n  {r.get('description', 'N/A')}\n  {url}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


SYSTEM_PROMPT = """You are a lead enrichment specialist. Given a company or person name, build a dossier.

IMPORTANT: Be efficient with tool calls. You have 3 tools:
- google_search: general info, news, funding, leadership
- crawl_website: deep-dive into a specific URL you found via search
- search_social_and_tech: social profiles + tech stack in ONE call

Typical flow: google_search first, then search_social_and_tech. Only crawl_website if you need deeper detail from a specific page.

Output format:

## Entity: [Name]
**Type**: Company | Person | Organization

### Overview
[2-3 sentence summary]

### Key Facts
- Founded: [year]
- HQ: [location]
- Size: [employee count range]
- Industry: [sector]
- Funding: [if applicable]

### Leadership
- [Name] — [Role]

### Products / Services
- [List]

### Tech Stack
- [Languages, frameworks, infra]

### Social Presence
- LinkedIn: [url]
- Twitter: [url]
- GitHub: [url]
- Crunchbase: [url]

### Recent News
- [Headlines with dates]

### Assessment
[Brief analysis of company health, growth trajectory, competitive position]

Mark unavailable data as "Not found"."""


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [google_search, crawl_website, search_social_and_tech]

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
        name="Lead Enrichment Agent",
        description="Takes a company or person name and returns a full enrichment dossier — "
        "website content, Google search intelligence, social profiles, tech stack, "
        "funding history, leadership team, and recent news.",
        capabilities={
            "ai": ["nlp", "data_enrichment", "web_scraping", "langchain"],
            "protocols": ["http"],
            "services": ["lead_enrichment", "company_research", "person_research",
                         "tech_stack_analysis", "social_profile_discovery"],
            "domains": ["sales", "marketing", "business_intelligence", "research"],
        },
        webhook_host="0.0.0.0",
        webhook_port=5010,
        registry_url="https://registry.zynd.ai",
        api_key=os.environ["ZYND_API_KEY"],
        config_dir=".agent-lead-enrichment",
        use_ngrok=False,
        webhook_url=get_webhook_url("lead-enrichment"),
        sync_response_timeout=300,
    )

    zynd_agent = ZyndAIAgent(agent_config=agent_config)
    zynd_agent.set_langchain_agent(create_agent())

    def handle_message(message: AgentMessage, topic: str):
        def _process():
            print(f"\n[Lead Enrichment] query={message.content} from={message.sender_id}")
            try:
                response = zynd_agent.invoke(message.content, chat_history=[])
                zynd_agent.set_response(message.message_id, response)
                print(f"[Lead Enrichment] done ({len(response)} chars)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                zynd_agent.set_response(message.message_id, f"Error: {e}")
        threading.Thread(target=_process, daemon=True).start()

    zynd_agent.add_message_handler(handle_message)
    print(f"\nLead Enrichment Agent | {zynd_agent.webhook_url} | {zynd_agent.agent_id}")

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
