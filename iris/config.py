"""Configuration for Iris. Values come from environment variables / the .env file."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MEMORY_DIR = BASE_DIR / "memory"

CLAUDE_MODEL = os.environ.get("IRIS_CLAUDE_MODEL", "claude-opus-5")
NOUS_MODEL = os.environ.get("IRIS_NOUS_MODEL", "tencent/hy3:free")
NOUS_BASE_URL = os.environ.get(
    "IRIS_NOUS_BASE_URL", "https://inference-api.nousresearch.com/v1"
)

PROVIDER = os.environ.get("IRIS_PROVIDER", "auto").lower()  # auto | claude | nous
YOLO = os.environ.get("IRIS_YOLO", "0") == "1"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USERS = {
    int(x)
    for x in os.environ.get("TELEGRAM_ALLOWED_USERS", "").replace(";", ",").split(",")
    if x.strip().isdigit()
}

HERMES_AUTH_FILE = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hermes" / "auth.json"
)


def resolve_nous_credentials() -> tuple[str, str]:
    """Return (api_key, base_url) for the Nous inference API.

    Order: NOUS_API_KEY env var, then the OAuth token Hermes Agent keeps
    refreshed in %LOCALAPPDATA%/hermes/auth.json.
    """
    key = os.environ.get("NOUS_API_KEY")
    if key:
        return key, NOUS_BASE_URL

    if HERMES_AUTH_FILE.exists():
        data = json.loads(HERMES_AUTH_FILE.read_text(encoding="utf-8"))
        nous = data.get("providers", {}).get("nous", {})
        token = nous.get("access_token")
        if token:
            expires_at = nous.get("expires_at")
            if expires_at:
                try:
                    exp = datetime.fromisoformat(expires_at)
                    if exp.tzinfo is None:
                        exp = exp.replace(tzinfo=timezone.utc)
                    if exp < datetime.now(timezone.utc):
                        raise RuntimeError(
                            "The Nous token borrowed from Hermes has expired. Run any "
                            'hermes command (e.g. `hermes -z "hi"`) to refresh it, or '
                            "set NOUS_API_KEY in .env."
                        )
                except ValueError:
                    pass  # unparseable timestamp — try the token anyway
            return token, nous.get("inference_base_url") or NOUS_BASE_URL

    raise RuntimeError(
        "No Nous credentials found. Set NOUS_API_KEY in .env, or log into "
        "Nous Portal via Hermes Agent (`hermes setup --portal`)."
    )


def pick_provider() -> str:
    if PROVIDER in ("claude", "nous"):
        return PROVIDER
    return "claude" if os.environ.get("ANTHROPIC_API_KEY") else "nous"
