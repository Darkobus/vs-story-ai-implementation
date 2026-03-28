import sys
from mcrcon import MCRcon

def main():
    host = "127.0.0.1"
    password = "klawisz"
    port = 42421

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
