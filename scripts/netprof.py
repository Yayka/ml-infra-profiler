"""
ml-netprof agent integration helper.

Add these lines to any training script before wandb.init():
    from netprof import setup_agent
    setup_agent()
Or copy the 5-line snippet from the README.
"""
import urllib.request

import wandb

AGENT_HEALTHZ = "http://localhost:9100/healthz"
AGENT_METRICS = "http://localhost:9100/metrics"


def setup_agent() -> bool:
    """Configure W&B to scrape the ml-netprof agent. Call before wandb.init().

    Returns True if the agent was detected and W&B was configured.
    """
    try:
        urllib.request.urlopen(AGENT_HEALTHZ, timeout=1)
    except Exception:
        return False
    wandb.setup(
        settings=wandb.Settings(
            x_stats_open_metrics_endpoints={"agent": AGENT_METRICS},
        )
    )
    return True
