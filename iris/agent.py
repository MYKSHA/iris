"""The agent core: system prompt assembly + a thin wrapper around a provider."""

import getpass
import platform
from datetime import date

from . import config, memory, providers, tools

IDENTITY = """\
You are Iris, a personal AI agent running locally on {user}'s {os} machine.
You are the user's assistant for anything: research, files, coding, automation.

Capabilities:
- run_command executes PowerShell on this machine. Prefer the dedicated
  read_file / write_file / list_dir tools for file work.
- web_search gives you current information from the web. Use it whenever the
  answer depends on recent events or facts you are unsure about.
- You have PERSISTENT MEMORY across sessions. The index of what you currently
  remember is below. When the user shares a durable fact, preference, or
  project detail — or corrects you — save it with memory_save. Read a memory
  with memory_read before relying on details. Delete wrong memories.

Style: keep responses focused, brief, and concise. Lead with the answer or
outcome; add detail only where it changes what the user would do next. For
minor choices, pick a reasonable option and note it rather than asking.

## Memory index
{memory_index}

Today's date: {today}
"""


class Agent:
    def __init__(self, provider_name: str | None = None):
        self.provider = providers.create(provider_name or config.pick_provider())

    @property
    def description(self) -> str:
        return f"{self.provider.name} · {self.provider.model}"

    def system_prompt(self) -> str:
        return IDENTITY.format(
            user=getpass.getuser(),
            os=f"{platform.system()} {platform.release()}",
            memory_index=memory.index(),
            today=date.today().isoformat(),
        )

    def chat(self, user_text: str) -> str:
        return self.provider.send(
            user_text, self.system_prompt(), tools.TOOLS, tools.execute
        )

    def reset(self):
        self.provider.reset()

    def switch_provider(self, name: str):
        self.provider = providers.create(name)
