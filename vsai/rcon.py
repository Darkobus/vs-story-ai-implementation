"""Minimal RCON client (Source RCON protocol — the same one mcrcon speaks).

Why not mcrcon: mcrcon implements its timeout with `signal.alarm`, which only works
in the main thread ("signal only works in main thread"). That forced the bot to queue
everything and send it from the log loop, so replies produced by background threads
waited for the next log line. Here the timeout is a plain `socket.settimeout`, so the
client can be called from any thread. A lock keeps messages from interleaving.

Packet layout (little-endian):  int32 length | int32 id | int32 type | body \\0 | \\0
"""

import select
import socket
import struct
import threading

SERVERDATA_AUTH = 3
SERVERDATA_EXECCOMMAND = 2


class RconError(Exception):
    pass


class RconClient:
    def __init__(self, host: str, port: int, password: str, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.password = password
        self.timeout = timeout
        self._lock = threading.Lock()

    def run(self, *commands: str) -> list[str]:
        """Connect, authenticate and run the commands in order. Returns the server responses.

        One connection per call (as with mcrcon before) — survives game server restarts,
        and traffic is tiny (a few messages a minute)."""
        with self._lock:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                sock.settimeout(self.timeout)
                self._request(sock, SERVERDATA_AUTH, self.password)
                return [self._request(sock, SERVERDATA_EXECCOMMAND, cmd) for cmd in commands]

    # --- protocol ---

    def _request(self, sock: socket.socket, packet_type: int, body: str) -> str:
        payload = struct.pack("<ii", 0, packet_type) + body.encode("utf-8") + b"\x00\x00"
        sock.sendall(struct.pack("<i", len(payload)) + payload)

        response = ""
        while True:
            (length,) = struct.unpack("<i", self._read(sock, 4))
            packet = self._read(sock, length)
            if len(packet) < 10 or packet[-2:] != b"\x00\x00":
                raise RconError("Malformed RCON packet")
            (request_id,) = struct.unpack("<i", packet[:4])
            if request_id == -1:
                raise RconError("RCON authentication failed (wrong password?)")
            response += packet[8:-2].decode("utf-8", errors="replace")
            # Long responses arrive split across packets — keep reading while data is waiting.
            if not select.select([sock], [], [], 0)[0]:
                return response

    @staticmethod
    def _read(sock: socket.socket, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise RconError("RCON server closed the connection")
            data += chunk
        return data
