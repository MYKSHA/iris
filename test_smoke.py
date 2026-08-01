"""Non-interactive smoke test: provider resolution, plain chat, and a memory tool call."""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from iris import memory
from iris.agent import Agent

agent = Agent()
print(f"[1] provider: {agent.description}")

reply = agent.chat("Reply with exactly: IRIS ONLINE")
print(f"[2] chat reply: {reply.strip()[:200]}")

reply = agent.chat(
    "Please save a memory that my favorite smoke-test fruit is mango, then confirm."
)
print(f"[3] memory turn reply: {reply.strip()[:300]}")

print(f"[4] memory index now:\n{memory.index()}")
