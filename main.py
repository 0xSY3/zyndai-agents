"""
ZyndAI Agent Fleet — Consolidated launcher.

Starts all 10 agents in threads + reverse proxy on PORT (default 10000).
Single-process deployment for Render free tier.

    python main.py
"""

import threading
import signal
import sys
import time
import importlib

from proxy import run_proxy

AGENT_MODULES = [
    "lead_enrichment_agent",
    "competitor_intel_agent",
    "content_pipeline_agent",
    "reputation_monitor_agent",
    "orchestrator_agent",
    "pricehawk_agent",
    "talentradar_agent",
    "adrecon_agent",
    "alphascout_agent",
    "dealflow_agent",
]


def start_agent(module_name: str) -> None:
    try:
        mod = importlib.import_module(module_name)
        mod.run_server()
    except Exception as e:
        print(f"[main] FAILED to start {module_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    print(f"[main] Starting ZyndAI fleet ({len(AGENT_MODULES)} agents + proxy)")

    # Start agents in daemon threads
    threads: list[threading.Thread] = []
    for mod_name in AGENT_MODULES:
        t = threading.Thread(target=start_agent, args=(mod_name,), daemon=True, name=mod_name)
        t.start()
        threads.append(t)
        # Stagger startup to avoid registry rate limits
        time.sleep(2)

    # Start proxy on main thread (blocks)
    print(f"[main] All agents launched, starting proxy...")

    def handle_signal(sig, frame):
        print(f"\n[main] Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    run_proxy()


if __name__ == "__main__":
    main()
