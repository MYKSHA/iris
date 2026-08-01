"""LLM provider adapters. Each keeps its own conversation history and runs the
tool-use loop until the model produces a final text answer.

ClaudeProvider — official Anthropic SDK, claude-opus-5, with server-side
refusal fallbacks enabled by default.
NousProvider — OpenAI-compatible endpoint of Nous Portal (free tier),
reusing the Hermes Agent OAuth token when no NOUS_API_KEY is set.
"""

import json

from . import config

MAX_TOOL_ITERATIONS = 25


def create(name: str):
    if name == "claude":
        return ClaudeProvider()
    if name == "nous":
        return NousProvider()
    raise ValueError(f"unknown provider '{name}' (expected 'claude' or 'nous')")


class ClaudeProvider:
    name = "claude"

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = config.CLAUDE_MODEL
        self.messages = []

    def reset(self):
        self.messages = []

    def _drop_last_exchange(self):
        # Remove everything back to (and including) the user turn that started
        # this exchange, so history stays valid for the next request.
        while self.messages and self.messages[-1].get("role") != "user":
            self.messages.pop()
        if self.messages:
            self.messages.pop()

    def send(self, user_text: str, system: str, tools: list, execute_tool) -> str:
        tools_payload = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tools
        ]
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=self.messages,
                tools=tools_payload,
                # Server-side refusal fallbacks: if Claude's safety classifiers
                # decline, the API re-runs the request on the recommended
                # fallback model instead of returning an empty refusal.
                extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
                extra_body={"fallbacks": "default"},
            )

            if response.stop_reason == "refusal":
                self._drop_last_exchange()
                return (
                    "[Claude declined this request (safety filters), and the fallback "
                    "chain declined too. Try rephrasing.]"
                )

            # Append the full content (thinking + tool_use blocks must be
            # preserved and passed back unchanged).
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue  # server-side work paused; re-send to resume

            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        output = execute_tool(block.name, dict(block.input or {}))
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output,
                            }
                        )
                # All results for parallel calls go back in ONE user message.
                self.messages.append({"role": "user", "content": results})
                continue

            text = "".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason == "max_tokens":
                text += "\n\n[response truncated: hit max_tokens]"
            return text

        return "[stopped: exceeded the tool-iteration limit for one message]"


class NousProvider:
    name = "nous"

    def __init__(self):
        from openai import OpenAI

        api_key, base_url = config.resolve_nous_credentials()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = config.NOUS_MODEL
        self.messages = []
        self.tools_supported = True

    def reset(self):
        self.messages = []

    def send(self, user_text: str, system: str, tools: list, execute_tool) -> str:
        # Keep the system prompt as message 0, refreshed each turn (memory
        # index and date change over time).
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = system
        else:
            self.messages.insert(0, {"role": "system", "content": system})
        self.messages.append({"role": "user", "content": user_text})

        tools_payload = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

        for _ in range(MAX_TOOL_ITERATIONS):
            kwargs = {"model": self.model, "messages": self.messages}
            if self.tools_supported:
                kwargs["tools"] = tools_payload
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                # Some free-tier models reject the tools parameter entirely —
                # degrade to plain chat rather than failing.
                if self.tools_supported and "tool" in str(e).lower():
                    self.tools_supported = False
                    continue
                raise

            msg = resp.choices[0].message
            entry = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self.messages.append(entry)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    output = execute_tool(tc.function.name, args)
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": output}
                    )
                continue

            return msg.content or "(empty response)"

        return "[stopped: exceeded the tool-iteration limit for one message]"
