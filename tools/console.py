import os
import sys
from pathlib import Path

from mcrcon import MCRcon



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
        sys.exit("Brak VS_RCON_PASSWORD — ustaw go w .env (patrz .env.example)")

    try:
        with MCRcon(host, password, port=port, timeout=10) as mcr:
            print("Connected to RCON. Type 'exit' to quit.")
            while True:
                cmd = input("VS> ")
                if cmd.lower() in ['exit', 'quit']:
                    break
                if cmd.strip():
                    response = mcr.command(cmd)
                    print(response)
    except Exception as e:
        print(f"Connection failed: {e}")


if __name__ == "__main__":
    main()
