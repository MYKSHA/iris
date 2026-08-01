"""Interactive CLI chat for Iris."""

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import config, memory, tools
from .agent import Agent

console = Console()

HELP = """\
/new                 start a fresh conversation
/provider claude     switch brain (claude | nous) — resets the conversation
/memory              show the persistent-memory index
/help                this help
/quit                exit
"""


def _confirm_shell(command: str) -> bool:
    console.print(
        Panel(command, title="[yellow]Iris wants to run this shell command[/]", border_style="yellow")
    )
    answer = console.input("[yellow]Allow? \\[y/N][/] ").strip().lower()
    return answer in ("y", "yes")


def main():
    tools.confirm_callback = _confirm_shell

    try:
        agent = Agent()
    except Exception as e:
        console.print(f"[red]Could not start a provider:[/] {e}")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold]Iris[/] — your local agent  ·  brain: [cyan]{agent.description}[/]\n"
            "Type /help for commands.",
            border_style="cyan",
        )
    )

    while True:
        try:
            # lstrip("﻿"): piped stdin on Windows can carry a UTF-8 BOM
            text = console.input("\n[bold cyan]you ›[/] ").lstrip("﻿").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nbye!")
            break
        if not text:
            continue

        if text in ("/quit", "/exit"):
            console.print("bye!")
            break
        if text == "/help":
            console.print(HELP)
            continue
        if text == "/new":
            agent.reset()
            console.print("[dim]conversation reset[/]")
            continue
        if text == "/memory":
            console.print(memory.index())
            continue
        if text.startswith("/provider"):
            parts = text.split()
            if len(parts) == 2 and parts[1] in ("claude", "nous"):
                try:
                    agent.switch_provider(parts[1])
                    console.print(f"[dim]switched to {agent.description} (new conversation)[/]")
                except Exception as e:
                    console.print(f"[red]switch failed:[/] {e}")
            else:
                console.print("usage: /provider claude|nous")
            continue

        try:
            console.print("[dim]thinking…[/]")
            reply = agent.chat(text)
        except KeyboardInterrupt:
            console.print("[dim]interrupted[/]")
            continue
        except Exception as e:
            console.print(f"[red]error:[/] {e}")
            continue

        console.print()
        console.print(Markdown(reply))


if __name__ == "__main__":
    main()
