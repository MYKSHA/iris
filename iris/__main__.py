"""Entry point: `python -m iris` for the CLI, `python -m iris telegram` for the bot."""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "telegram":
        from . import telegram_gateway

        telegram_gateway.main()
    else:
        from . import cli

        cli.main()


if __name__ == "__main__":
    main()
