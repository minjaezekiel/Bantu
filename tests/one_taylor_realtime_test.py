#!/usr/bin/env python3
"""
one_taylor_realtime_test.py — the WebSocket hub: messaging, presence, calls.

Written against the standard library only. There is no WebSocket client in
Bantu and adding a pip dependency to run one test is a worse trade than the
sixty lines of RFC 6455 below — a client handshake, a masked text frame and an
unmasked read are the whole protocol as far as this test needs it.

What it proves, in order of how much it would hurt to get wrong:

  1. A socket cannot do anything until it has spent a valid ticket, and a
     ticket cannot be spent twice.
  2. A message sent by one user ARRIVES at the other user's socket.
  3. THE AGE WALL HOLDS ON THE SOCKET. A minor and an adult cannot message
     each other, cannot call each other, and cannot learn the other exists —
     over the WebSocket, which is a completely separate code path from the
     HTTP routes the shell test covers.
  4. A grooming message from a minor's account is refused and NOT delivered.
  5. Call signalling relays an offer and an answer between two peers, and
     refuses a call across the wall.

Run:  python3 tests/one_taylor_realtime_test.py            (starts its own server)
      ONE_TAYLOR_RT_PORT=4210 python3 ... --no-server      (against a running one)
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BANTU = HERE / "bantu"
BACKEND = HERE / "samples" / "one-taylor" / "backend"
PORT = int(os.environ.get("ONE_TAYLOR_RT_PORT", "4231"))
BASE = f"http://127.0.0.1:{PORT}"
ORIGIN = "http://localhost:3000"

PASS = 0
FAIL = 0


def ok(name: str) -> None:
    global PASS
    PASS += 1
    print(f"  ok    {name}")


def bad(name: str, detail: str = "") -> None:
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}")
    if detail:
        print(f"          {detail}")


def check(cond: bool, name: str, detail: str = "") -> bool:
    if cond:
        ok(name)
    else:
        bad(name, detail)
    return bool(cond)


# ── HTTP ────────────────────────────────────────────────────────────────────
def api(method: str, path: str, token: str | None = None, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method)
    req.add_header("content-type", "application/json")
    if token:
        req.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read() or b"{}")
        except Exception:
            return {"ok": False, "error": f"http {e.code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def signup(username: str, dob: str) -> str:
    r = api("POST", "/api/auth/signup", None, {
        "username": username, "email": f"{username}@example.com",
        "password": "correct-horse-7", "dob": dob, "gender": "other"})
    return r.get("token", "")


# ── a WebSocket client, RFC 6455, the parts this test needs ─────────────────
class WS:
    def __init__(self, port: int, origin: str = ORIGIN, timeout: float = 10.0):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.buf = b""
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        req = (
            f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
            f"Origin: {origin}\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("the server closed during the handshake")
            head += chunk
        line = head.split(b"\r\n", 1)[0].decode(errors="replace")
        if "101" not in line:
            raise ConnectionError(f"upgrade refused: {line}")
        self.buf = head.split(b"\r\n\r\n", 1)[1]

    def send(self, obj: dict) -> None:
        payload = json.dumps(obj).encode()
        # A client frame MUST be masked; a server rejects an unmasked one.
        header = bytearray([0x81])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = secrets.token_bytes(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def _fill(self, n: int) -> None:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            self.buf += chunk

    def recv(self) -> dict | None:
        """One frame. Control frames are skipped; a close returns None."""
        while True:
            self._fill(2)
            b0, b1 = self.buf[0], self.buf[1]
            opcode = b0 & 0x0F
            length = b1 & 0x7F
            offset = 2
            if length == 126:
                self._fill(4)
                length = struct.unpack(">H", self.buf[2:4])[0]
                offset = 4
            elif length == 127:
                self._fill(10)
                length = struct.unpack(">Q", self.buf[2:10])[0]
                offset = 10
            self._fill(offset + length)
            payload = self.buf[offset:offset + length]
            self.buf = self.buf[offset + length:]
            if opcode == 0x8:
                return None
            if opcode in (0x9, 0xA):
                continue
            try:
                return json.loads(payload)
            except Exception:
                return {"type": "?", "raw": payload.decode(errors="replace")}

    def until(self, *types: str, tries: int = 25) -> dict | None:
        """Read until one of these frame types arrives. Presence and typing
        frames interleave with everything, so a test that reads exactly one
        frame is a test that fails intermittently."""
        for _ in range(tries):
            try:
                f = self.recv()
            except (socket.timeout, TimeoutError):
                return None
            if f is None:
                return None
            if not types or f.get("type") in types:
                return f
        return None

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def hello(port: int, token: str) -> tuple[WS, dict | None]:
    t = api("POST", "/api/realtime/ticket", token, {})
    ws = WS(port)
    ws.send({"type": "hello", "ticket": t.get("ticket", "")})
    return ws, ws.until("welcome", "error")


def main() -> int:
    print("=" * 64)
    print("  One Taylor — WebSocket hub: messaging, presence, calls")
    print("=" * 64)

    proc = None
    work = tempfile.mkdtemp()
    if "--no-server" not in sys.argv:
        env = dict(os.environ)
        env.update({
            "ONE_TAYLOR_SECRET": secrets.token_hex(32),
            "ONE_TAYLOR_DB": str(Path(work) / "rt.db"),
            "PORT": str(PORT),
        })
        log = open(Path(work) / "server.log", "wb")
        proc = subprocess.Popen([str(BANTU), "-q", "run", "server.b"],
                                cwd=str(BACKEND), stdout=log, stderr=log, env=env)
        for _ in range(120):
            if api("GET", "/api/health").get("ok"):
                break
            time.sleep(0.5)

    try:
        if not api("GET", "/api/health").get("ok"):
            bad("the server came up")
            return 1
        ok("the server is up")

        # ── accounts: two adults who follow each other, and two minors ─────
        ada = signup("adaws", "1996-03-03")
        bay = signup("bayws", "1995-03-03")
        mina = signup("minaws", "2011-03-03")
        nia = signup("niaws", "2010-03-03")
        if not all([ada, bay, mina, nia]):
            bad("created four accounts", "one signup returned no token")
            return 1
        ok("created two adults and two minors")

        for a, b in (("adaws", "bayws"), ("bayws", "adaws"),
                     ("minaws", "niaws"), ("niaws", "minaws")):
            tok = {"adaws": ada, "bayws": bay, "minaws": mina, "niaws": nia}[a]
            api("POST", f"/api/users/{b}/follow", tok, {})
        ok("both pairs follow each other")

        # ── 1. tickets ─────────────────────────────────────────────────────
        print("\n-- tickets --")
        ws = WS(PORT)
        ws.send({"type": "message", "to": "bayws", "body": "before hello"})
        f = ws.until("error")
        check(f is not None and "hello" in str(f.get("message", "")).lower(),
              "a socket can do nothing before it says hello", str(f))
        ws.close()

        ws = WS(PORT)
        ws.send({"type": "hello", "ticket": "not-a-real-ticket"})
        f = ws.until("error", "welcome")
        check(f is not None and f.get("type") == "error",
              "an invalid ticket is refused", str(f))
        ws.close()

        t1 = api("POST", "/api/realtime/ticket", ada, {}).get("ticket", "")
        w1 = WS(PORT)
        w1.send({"type": "hello", "ticket": t1})
        check((w1.until("welcome", "error") or {}).get("type") == "welcome",
              "a valid ticket is accepted")
        w2 = WS(PORT)
        w2.send({"type": "hello", "ticket": t1})
        f = w2.until("error", "welcome")
        check(f is not None and f.get("type") == "error",
              "the same ticket cannot be spent twice", str(f))
        w2.close()
        w1.close()

        # ── 2. messaging ───────────────────────────────────────────────────
        print("\n-- messaging --")
        wa, welcomeA = hello(PORT, ada)
        wb, welcomeB = hello(PORT, bay)
        check(welcomeA is not None and welcomeA.get("type") == "welcome",
              "ada's socket is welcomed")
        check(welcomeB is not None and welcomeB.get("type") == "welcome",
              "bayo's socket is welcomed")

        wa.send({"type": "message", "to": "bayws", "body": "fancy a game of chess?"})
        sent = wa.until("message-sent", "error")
        check(sent is not None and sent.get("type") == "message-sent"
              and sent.get("verdict") == "allow",
              "the sender gets their own message back", str(sent))
        got = wb.until("message", "error")
        check(got is not None and got.get("type") == "message"
              and got.get("message", {}).get("body") == "fancy a game of chess?",
              "AND IT ARRIVES at the other socket", str(got))
        check(got is not None and got.get("message", {}).get("mine") is False,
              "marked as theirs, not mine, on arrival")

        wa.send({"type": "typing", "to": "bayws", "on": True})
        f = wb.until("typing", "error")
        check(f is not None and f.get("type") == "typing" and f.get("on") is True,
              "a typing indicator relays")

        # ── 3. the age wall, on the socket ─────────────────────────────────
        print("\n-- THE AGE WALL, over the WebSocket --")
        wm, _ = hello(PORT, mina)
        wm.send({"type": "message", "to": "adaws", "body": "hello adult"})
        f = wm.until("error", "message-sent")
        check(f is not None and f.get("type") == "error"
              and "no such account" in str(f.get("message", "")),
              "a minor cannot message an adult, and is told nothing else", str(f))

        wa.send({"type": "message", "to": "minaws", "body": "hello child"})
        f = wa.until("error", "message-sent")
        check(f is not None and f.get("type") == "error"
              and "no such account" in str(f.get("message", "")),
              "an adult cannot message a minor", str(f))

        wa.send({"type": "call-offer", "to": "minaws",
                 "sdp": {"type": "offer", "sdp": "v=0"}, "video": True})
        f = wa.until("error", "call-ringing", "call-reject")
        check(f is not None and f.get("type") == "error",
              "AND CANNOT CALL ONE EITHER", str(f))

        # Nothing at all reached the minor's socket.
        wm.sock.settimeout(1.5)
        leaked = wm.until("message", "call-ringing", tries=3)
        check(leaked is None, "nothing crossed the wall onto the minor's socket",
              str(leaked))
        wm.sock.settimeout(10.0)

        # ── 4. moderation on a direct message ──────────────────────────────
        print("\n-- moderation, where grooming actually happens --")
        wn, _ = hello(PORT, nia)
        wm.send({"type": "message", "to": "niaws",
                 "body": "dont tell your parents, this is our secret"})
        f = wm.until("message-sent", "error")
        blocked = (f or {}).get("message", {}).get("status") == "blocked"
        check(blocked, "a grooming message from a minor's account is refused", str(f))
        wn.sock.settimeout(1.5)
        delivered = wn.until("message", tries=3)
        check(delivered is None, "and is NOT delivered to the other child",
              str(delivered))
        wn.sock.settimeout(10.0)

        wm.send({"type": "message", "to": "niaws",
                 "body": "scored twice at practice today, best session all term"})
        f = wm.until("message-sent", "error")
        # "live" whatever the classifier thought: on this path only a lexicon
        # hit stops a message, because not delivering one is the irreversible
        # action. See the comment in realtime/chat.b send().
        check((f or {}).get("message", {}).get("status") == "live",
              "while an innocent message from the same account goes through", str(f))
        check((wn.until("message", "error") or {}).get("type") == "message",
              "and arrives")

        # ── 5. call signalling ─────────────────────────────────────────────
        print("\n-- call signalling --")
        wa.send({"type": "call-offer", "to": "bayws",
                 "sdp": {"type": "offer", "sdp": "v=0\r\na=ada"}, "video": True})
        ring = wb.until("call-ringing", "error")
        check(ring is not None and ring.get("type") == "call-ringing"
              and ring.get("video") is True,
              "an offer rings the other peer", str(ring))
        check(ring is not None and ring.get("sdp", {}).get("sdp", "").endswith("ada"),
              "and carries the offer SDP through unchanged")

        wb.send({"type": "call-answer", "to": "adaws",
                 "sdp": {"type": "answer", "sdp": "v=0\r\na=bayo"}})
        ans = wa.until("call-answer", "error")
        check(ans is not None and ans.get("type") == "call-answer"
              and ans.get("sdp", {}).get("sdp", "").endswith("bayo"),
              "the answer relays back", str(ans))

        wa.send({"type": "call-ice", "to": "bayws",
                 "candidate": {"candidate": "candidate:1 1 udp 1 10.0.0.1 1 typ host"}})
        ice = wb.until("call-ice", "error")
        check(ice is not None and ice.get("type") == "call-ice",
              "ICE candidates trickle through", str(ice))

        wa.send({"type": "call-end", "to": "bayws"})
        check((wb.until("call-end", "error") or {}).get("type") == "call-end",
              "hanging up reaches the other peer")

        # A second offer after the call ended must work: the busy flag has to
        # be cleared, or one call makes the pair permanently unreachable.
        wa.send({"type": "call-offer", "to": "bayws",
                 "sdp": {"type": "offer", "sdp": "v=0"}, "video": False})
        check((wb.until("call-ringing", "error") or {}).get("type") == "call-ringing",
              "and a new call can be placed afterwards")
        wa.send({"type": "call-end", "to": "bayws"})
        wb.until("call-end")

        # ── 5b. two tabs: the regression that broke every call ─────────────
        #
        # A user with two tabs has two sockets, and the hub rings both — which
        # is right, they may not be looking at the one holding the thread. The
        # client bug this guards was on the other side: a second copy of
        # call-ringing hit an "already in a call" branch and auto-REJECTED,
        # so a call placed to anyone with a second socket (which React
        # StrictMode gives every tab in development) never connected.
        print("\n-- two tabs, one user --")
        wb2, w2welcome = hello(PORT, bay)
        check((w2welcome or {}).get("type") == "welcome",
              "bayo opens a second tab")

        wa.send({"type": "call-offer", "to": "bayws",
                 "sdp": {"type": "offer", "sdp": "v=0\r\na=two-tabs"}, "video": False})
        r1 = wb.until("call-ringing", "error")
        r2 = wb2.until("call-ringing", "error")
        check(r1 is not None and r1.get("type") == "call-ringing"
              and r2 is not None and r2.get("type") == "call-ringing",
              "BOTH tabs ring", f"{r1} / {r2}")

        # The first tab answers. The second must be told to stop ringing, or
        # the user is left dismissing a call that is already connected.
        wb.send({"type": "call-answer", "to": "adaws",
                 "sdp": {"type": "answer", "sdp": "v=0\r\na=tab1"}})
        check((wa.until("call-answer", "error") or {}).get("type") == "call-answer",
              "the answer reaches the caller")
        check((wb2.until("call-end", "call-ringing", "error") or {}).get("type") == "call-end",
              "and the OTHER tab is told the call was taken")

        wa.send({"type": "call-end", "to": "bayws"})
        wb.until("call-end")
        wb2.until("call-end", tries=4)
        wb2.close()

        # ── 6. an unknown frame is an error, not a crash ───────────────────
        print("\n-- robustness --")
        wa.send({"type": "nonsense-frame"})
        check((wa.until("error") or {}).get("type") == "error",
              "an unknown frame type is answered with an error")
        check(api("GET", "/api/health").get("ok") is True,
              "and the server is still healthy")

        for w in (wa, wb, wm, wn):
            w.close()

    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    print()
    print("=" * 64)
    print(f"  PASS: {PASS}   FAIL: {FAIL}")
    if FAIL == 0:
        print("  RESULT: ALL GREEN")
        return 0
    print("  RESULT: FAILED")
    print(f"  server log: {Path(work) / 'server.log'}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
