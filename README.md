# Iris — your personal AI agent

A compact, Hermes-style personal agent that runs locally on this machine:
an LLM with tools (shell, files, web search), **persistent memory across
sessions**, an interactive CLI, and a Telegram gateway so you can chat with it
from your phone.

Two interchangeable brains:

| Provider | Model | Cost | Notes |
|---|---|---|---|
| `claude` | `claude-opus-5` (Anthropic API) | pay per token | Best quality. Set `ANTHROPIC_API_KEY` in `.env`. Server-side refusal fallbacks are enabled by default. |
| `nous` | `tencent/hy3:free` (Nous Portal) | free | Works out of the box — reuses the OAuth token Hermes Agent keeps in `%LOCALAPPDATA%\hermes\auth.json`, or set `NOUS_API_KEY`. |

`IRIS_PROVIDER=auto` (default) picks Claude when a key is present, otherwise Nous.

## Run it

`iris` is on your PATH (launcher at `%LOCALAPPDATA%\Microsoft\WindowsApps\iris.cmd`):

```powershell
iris            # CLI chat, from any directory
iris telegram   # Telegram gateway
```

(Equivalent long form: `C:\Users\manku\iris\venv\Scripts\python.exe -m iris`.)

CLI commands: `/new` (reset conversation), `/provider claude|nous`, `/memory`,
`/help`, `/quit`.

## Telegram setup (one-time)

1. In Telegram, talk to **@BotFather** → `/newbot` → copy the token.
2. Put `TELEGRAM_BOT_TOKEN=...` in `.env`.
3. Start the gateway and message your bot once — it replies with your numeric
   user id. Put that in `.env` as `TELEGRAM_ALLOWED_USERS=<id>` and restart.

Only allowlisted user ids can talk to the bot; everyone else is refused.

## Memory

One markdown file per fact under `memory/`. The index (name + first line) is
injected into the system prompt every turn, so Iris knows what it remembers
across sessions and providers. Iris saves/reads/deletes memories itself via
tools; you can also just edit the files.

## Tools

`run_command` (PowerShell), `read_file`, `write_file`, `list_dir`,
`web_search` (DuckDuckGo), `memory_save` / `memory_read` / `memory_delete`.

### Shell safety

- **CLI:** every shell command asks for y/N confirmation before running.
- **Telegram:** there is no confirmation prompt, so shell commands are refused
  unless you set `IRIS_YOLO=1` in `.env`. Leave it `0` unless you trust every
  allowlisted user with a PowerShell prompt on this machine.

## Troubleshooting

- *"The Nous token borrowed from Hermes has expired"* — run any hermes command
  (e.g. `hermes -z "hi"`) to refresh it, or set `NOUS_API_KEY` in `.env`.
- Web search occasionally rate-limits (DuckDuckGo); retry after a moment.
- To upgrade the brain: create a key at https://platform.claude.com, add
  `ANTHROPIC_API_KEY=sk-ant-...` to `.env` — the next run uses Claude Opus 5.

## Layout

```
iris/
  agent.py             system prompt + provider wrapper
  providers.py         Claude + Nous adapters (each owns its tool loop)
  tools.py             tool registry + implementations
  memory.py            persistent memory files
  cli.py               interactive terminal chat
  telegram_gateway.py  Telegram bot
  config.py            .env loading + credential resolution
memory/                the agent's persistent memories
test_smoke.py          non-interactive end-to-end check
```
