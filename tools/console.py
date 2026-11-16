"""Interactive RCON console.

Usage:
    python -m tools.console
"""

import sys

from vsai import config
from vsai.rcon import RconClient


def main():
    if not config.RCON_PASSWORD:
        sys.exit("VS_RCON_PASSWORD is not set — add it to .env (see .env.example)")

    client = RconClient(config.RCON_HOST, config.RCON_PORT, config.RCON_PASSWORD, timeout=10)
    try:
        client.run()  # check connection and password
    except Exception as e:
        sys.exit(f"Connection failed: {e}")

    print("Connected to RCON. Type 'exit' to quit.")
    while True:
        cmd = input("VS> ")
        if cmd.lower() in ("exit", "quit"):
            break
        if cmd.strip():
            try:
                print(client.run(cmd)[0])
            except Exception as e:
                print(f"RCON error: {e}")


if __name__ == "__main__":
    main()
