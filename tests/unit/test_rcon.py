"""RconClient against a fake Source-RCON server on localhost."""
import socket
import struct
import threading

import pytest

from vsai.rcon import RconClient, RconError

PASSWORD = "secret"


def _read_packet(conn):
    (length,) = struct.unpack("<i", conn.recv(4, socket.MSG_WAITALL))
    data = b""
    while len(data) < length:
        data += conn.recv(length - len(data))
    request_id, ptype = struct.unpack("<ii", data[:8])
    return request_id, ptype, data[8:-2].decode()


def _send_packet(conn, request_id, ptype, body):
    payload = struct.pack("<ii", request_id, ptype) + body.encode() + b"\x00\x00"
    conn.sendall(struct.pack("<i", len(payload)) + payload)


@pytest.fixture
def fake_server():
    """Accepts connections, checks the password, echoes commands back; records them."""
    srv = socket.create_server(("127.0.0.1", 0))
    received = []

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                rid, ptype, body = _read_packet(conn)
                assert ptype == 3
                _send_packet(conn, rid if body == PASSWORD else -1, 2, "")
                if body != PASSWORD:
                    continue
                while True:
                    try:
                        rid, ptype, body = _read_packet(conn)
                    except (struct.error, OSError):
                        break
                    received.append(body)
                    _send_packet(conn, rid, 0, f"ok: {body}")

    threading.Thread(target=serve, daemon=True).start()
    yield srv.getsockname()[1], received
    srv.close()


def test_runs_commands_and_returns_responses(fake_server):
    port, received = fake_server
    client = RconClient("127.0.0.1", port, PASSWORD, timeout=2)
    assert client.run("announce AI: hej", "time") == ["ok: announce AI: hej", "ok: time"]
    assert received == ["announce AI: hej", "time"]


def test_wrong_password_raises(fake_server):
    port, _ = fake_server
    with pytest.raises(RconError, match="authentication failed"):
        RconClient("127.0.0.1", port, "wrong", timeout=2).run("time")


def test_works_from_background_thread(fake_server):
    """The whole point vs mcrcon: no signal-based timeout, so threads can send."""
    port, _ = fake_server
    client = RconClient("127.0.0.1", port, PASSWORD, timeout=2)
    results, errors = [], []

    def worker(i):
        try:
            results.append(client.run(f"announce {i}"))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert not errors
    assert sorted(r[0] for r in results) == [f"ok: announce {i}" for i in range(5)]
