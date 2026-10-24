#!/usr/bin/env python3
"""
tls-dev-proxy.py — a TLS front end for local WebRTC testing.

WHY THIS EXISTS
    navigator.mediaDevices.getUserMedia is only available in a secure
    context. http://localhost counts as one; http://192.168.x.x does NOT.
    So a phone pointed at this machine's LAN address over plain HTTP cannot
    open a microphone at all -- navigator.mediaDevices is simply undefined.

    sua cannot serve HTTPS (sua.server has no TLS of any kind; sua.http's TLS
    options are for the client side), so this terminates TLS in front of it.

WHAT IT DOES
    Generates a self-signed certificate whose subjectAltName carries this
    machine's LAN IP, listens on 8443, and pipes raw bytes to the Bantu
    server on 8080. Because it proxies TCP rather than parsing HTTP, the
    WebSocket upgrade passes through untouched -- and so does Host, which is
    what sua's same-origin WebSocket check compares against.

    The handshake runs on the connection's own thread, never on the accept
    loop: see the note in serve_one for the outage that caused.

    This is a development tool for one LAN. It is not a production proxy:
    no certificate verification of any kind is being claimed, and a
    self-signed certificate is exactly why the phone asks you to confirm.

USE
    python3 tls-dev-proxy.py                # 8443 -> 127.0.0.1:8080
    python3 tls-dev-proxy.py --target 4000  # a server on another port
    python3 tls-dev-proxy.py --listen 9443

    Then open the https:// URL it prints on the phone and accept the warning
    once. Standard library only -- nothing to install.
"""

import argparse
import os
import socket
import ssl
import subprocess
import sys
import tempfile
import threading

CERT_DIR = os.path.join(tempfile.gettempdir(), "bantu-talk-tls")


def lan_ip() -> str:
    """This machine's address on the LAN.

    A UDP socket is 'connected' to an off-machine address to make the kernel
    choose a source interface; nothing is sent. Reading the socket's own name
    is the only portable way to learn which of several interfaces the LAN
    actually uses -- the hostname often resolves to 127.0.0.1.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(ip: str):
    """A self-signed cert for this IP, reused across runs.

    The IP goes in subjectAltName, not just the common name: browsers have
    ignored CN for host matching for years, and a cert without a matching SAN
    is rejected outright rather than merely warned about -- which looks like
    the proxy is broken.
    """
    os.makedirs(CERT_DIR, exist_ok=True)
    cert = os.path.join(CERT_DIR, f"{ip}.crt")
    key = os.path.join(CERT_DIR, f"{ip}.key")
    if os.path.exists(cert) and os.path.exists(key):
        return cert, key

    conf = os.path.join(CERT_DIR, f"{ip}.cnf")
    with open(conf, "w") as fh:
        fh.write(
            "[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=v3\n"
            "[dn]\nCN=bantu-talk\n"
            "[v3]\nbasicConstraints=CA:FALSE\n"
            "keyUsage=digitalSignature,keyEncipherment\n"
            "extendedKeyUsage=serverAuth\n"
            f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost\n"
        )

    print(f"  generating a self-signed certificate for {ip} …")
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256",
             "-days", "365", "-nodes", "-keyout", key, "-out", cert,
             "-config", conf],
            check=True, capture_output=True,
        )
    except FileNotFoundError:
        sys.exit("  openssl is not on PATH — install it, or use the Chrome flag route (see README).")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"  openssl failed:\n{exc.stderr.decode(errors='replace')}")
    return cert, key


def pump(src: socket.socket, dst: socket.socket):
    """Copy one direction until it ends, then close the write side.

    Half-closing rather than closing both is what lets a WebSocket shut down
    cleanly in one direction while the other is still draining.
    """
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


# How long a client gets to finish the TLS handshake. Long enough for a phone
# on slow Wi-Fi, short enough that a socket which will never speak is gone
# before anyone notices.
HANDSHAKE_TIMEOUT = 15.0

# A connection costs a thread, so the number of them is bounded. 400 silent
# sockets produced 401 threads and real requests were still served in 3ms, so
# this is set well above anything a LAN of phones and laptops will reach; it is
# here so a flood costs a refused connection instead of the whole process.
# ponytail: a counter, not a thread pool -- rewrite as a pool only if this
# proxy ever has to serve more than a room full of devices.
MAX_CONNECTIONS = 512
_slots = threading.BoundedSemaphore(MAX_CONNECTIONS)


def serve_one(raw: socket.socket, peer: tuple, ctx: ssl.SSLContext, target: tuple):
    try:
        _serve_one(raw, peer, ctx, target)
    finally:
        _slots.release()


def _serve_one(raw: socket.socket, peer: tuple, ctx: ssl.SSLContext, target: tuple):
    # [found] THE TLS HANDSHAKE USED TO RUN ON THE ACCEPT THREAD.
    #
    # ctx.wrap_socket() was called in the accept loop, on a socket with no
    # timeout, so it blocked until the client sent a ClientHello. A client that
    # opens a TCP connection and sends nothing therefore stopped this proxy
    # accepting ANY further connection, for as long as it stayed open --
    # already-running connections kept working, because each has its own pump
    # threads, so the symptom was "my phone and laptop are fine but a third
    # device cannot load the page at all".
    #
    # Browsers open exactly such sockets: Safari and Chrome both preconnect
    # speculatively and may never send bytes on the spare. Reproduced with one
    # silent socket -- every later request timed out, and the proxy recovered
    # the instant that socket closed.
    #
    # Two changes, both needed. The handshake happens HERE, on the connection's
    # own thread, so a slow or silent client delays only itself; and it has a
    # timeout, so it cannot hold a thread forever either.
    try:
        raw.settimeout(HANDSHAKE_TIMEOUT)
        tls_sock = ctx.wrap_socket(raw, server_side=True)
    except (ssl.SSLError, OSError) as exc:
        # A browser probing for plain HTTP, a port scan, or a user who refused
        # the certificate. One connection's problem, not the proxy's.
        print(f"  handshake with {peer[0]} failed: {exc}")
        try:
            raw.close()
        except OSError:
            pass
        return

    try:
        upstream = socket.create_connection(target, timeout=10)
    except OSError as exc:
        print(f"  upstream {target[0]}:{target[1]} refused the connection ({exc}) — is the Bantu server running?")
        tls_sock.close()
        return

    # [found] THE timeout passed to create_connection() above applies to the
    # CONNECT *and to every later recv() on that socket*. Leaving it at 10s
    # meant this proxy killed any connection that went 10 seconds without
    # traffic -- and a chat WebSocket is idle almost all the time. The phone
    # reconnected 58 times in one session while the two peers on plain
    # localhost, which do not go through here, reconnected 5 times between
    # them. Clearing it leaves a connect timeout without a read timeout,
    # which is what a relay wants: it must sit still for as long as the peers
    # are quiet.
    upstream.settimeout(None)
    tls_sock.settimeout(None)

    # Nagle off in both directions: signalling frames are small and latency
    # here is call-setup latency.
    for s in (tls_sock, upstream):
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
    # Only ONE extra thread: this one carries the other direction itself.
    # Two threads per connection instead of three -- at a few hundred devices
    # that difference is real, and the thread that owns the sockets is the one
    # that closes them.
    up = threading.Thread(target=pump, args=(tls_sock, upstream), daemon=True)
    up.start()
    pump(upstream, tls_sock)
    up.join()
    for s in (tls_sock, upstream):
        try:
            s.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="TLS front end for the Bantu WebRTC chat sample")
    ap.add_argument("--listen", type=int, default=8443, help="TLS port to listen on (default 8443)")
    ap.add_argument("--target", type=int, default=8080, help="plain-HTTP Bantu server port (default 8080)")
    ap.add_argument("--host", default="127.0.0.1", help="host the Bantu server is on (default 127.0.0.1)")
    args = ap.parse_args()

    ip = lan_ip()
    cert, key = ensure_cert(ip)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", args.listen))
    except OSError as exc:
        sys.exit(f"  cannot listen on port {args.listen}: {exc}")
    srv.listen(64)

    print("")
    print("  TLS front end for Bantu Talk")
    print("  ───────────────────────────────────────────────")
    print(f"  forwarding  https://0.0.0.0:{args.listen}  ->  http://{args.host}:{args.target}")
    print("")
    print(f"  On your phone, on the same Wi-Fi:   https://{ip}:{args.listen}")
    print(f"  On this machine:                    https://localhost:{args.listen}")
    print("")
    print("  The certificate is self-signed, so the phone will warn once:")
    print("    iOS Safari  — Show Details -> visit this website")
    print("    Android     — Advanced -> Proceed")
    print("")
    print("  Ctrl-C to stop.")
    print("")

    try:
        while True:
            try:
                raw, peer = srv.accept()
            except OSError:
                continue
            if not _slots.acquire(blocking=False):
                print(f"  refusing {peer[0]}: {MAX_CONNECTIONS} connections already open")
                try:
                    raw.close()
                except OSError:
                    pass
                continue

            # Nothing but accept() happens on this thread: the handshake is
            # the connection's own problem, on the connection's own thread.
            threading.Thread(
                target=serve_one,
                args=(raw, peer, ctx, (args.host, args.target)),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
