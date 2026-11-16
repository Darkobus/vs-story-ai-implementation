import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))
from rcon import RconClient  # noqa: E402  (bot/rcon.py — same client the bot uses)


def _load_env():
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    _load_env()
    host = os.environ.get("VS_RCON_HOST", "127.0.0.1")
    port = int(os.environ.get("VS_RCON_PORT", "42421"))
    password = os.environ.get("VS_RCON_PASSWORD")
    if not password:
        sys.exit("VS_RCON_PASSWORD is not set — add it to .env (see .env.example)")

    client = RconClient(host, password=password, port=port, timeout=10)
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
