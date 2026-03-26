"""
Shared deployment config — maps agent slugs to ports and computes public webhook URLs.

When RENDER_EXTERNAL_URL is set, agents register with their public URL.
Locally, agents use their internal host:port.
"""

import os

AGENT_MAP: dict[str, int] = {
    "lead-enrichment": 5010,
    "competitor-intel": 5011,
    "content-pipeline": 5012,
    "reputation-monitor": 5013,
    "orchestrator": 5014,
    "pricehawk": 5015,
    "talentradar": 5016,
    "adrecon": 5017,
    "alphascout": 5018,
    "dealflow": 5019,
}


def get_webhook_url(slug: str) -> str | None:
    """Return public webhook URL if running on Render, else None (SDK uses internal URL)."""
    base = os.environ.get("RENDER_EXTERNAL_URL")
    if not base:
        return None
    return f"{base.rstrip('/')}/{slug}"
