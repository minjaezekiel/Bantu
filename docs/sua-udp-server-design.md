# A UDP server that is two lines, and multicast

`sua.udp` already has the low-level surface: `socket`, `bind`, `send_to`, `recvfrom`,
`getsockname`, `close`, and the one-shot `send`. Two things the original feature request
(AsseySilivestir/Bantu#3) asked for were never built: a **callback server**, and **multicast**.
This is how both work.

## 1. The callback form

```bantu
sua.udp.bind(3478, def($packet) {
    print("recv " + str(len($packet.data)) + " bytes from " + $packet.from);
    $packet.reply($packet.data);          // straight back to the sender
});
sua.udp.listen();                         // serve, forever
```

**Why overload `bind` rather than add a name.** The request asked for exactly this spelling, and
the two forms cannot be confused: `bind($sock, "host:port")` takes a socket handle first,
`bind(port, handler)` takes a number. A third spelling for the same verb would be the worse
outcome. The port may also be given as an address string, `bind("127.0.0.1:3478", handler)`, for a
server that should not listen on every interface.

**`$packet`** is a dict with three entries:

| | |
|---|---|
| `from` | `"host:port"` of the sender, the same spelling `send_to` accepts |
| `data` | the datagram as a list of byte values, as `recvfrom` returns and `send_to` takes |
| `reply` | a function: `$packet.reply($bytes)` sends back to `from` on the same socket |

`reply` exists because the alternative is making every handler re-parse `from` and call `send_to`
with the socket it was never handed. It is the single most common thing a UDP server does.

**A handler that raises does not stop the server.** The error is printed with the sender's address
and the next datagram is served. A UDP server's peers are unauthenticated by definition, so a
malformed packet that trips a handler must cost that packet and nothing more.

**Datagrams are read until the socket is empty**, not one per wakeup, because one readable event can
cover many queued datagrams and leaving them queued adds latency to every one behind. The buffer is
65536 bytes, above the 65507-byte maximum a UDP datagram can carry, so nothing is ever truncated.

## 2. Where it runs

The callback sockets are served by an event loop, never by a thread each.

- **`sua.udp.listen()`** runs a loop over the registered UDP sockets. It uses the same
  `bantu_loop::Backend` (kqueue, epoll, poll) as the HTTP server, so there is one readiness
  implementation in the tree, not two.
- **`sua.server.listen(port)`** also serves them: the HTTP loop adds every registered UDP socket to
  its backend, so a program that speaks HTTP and UDP needs one loop and one process. This is the
  case the feature request actually had in mind — a signaling server over HTTP and a TURN relay over
  UDP, in one Bantu program.
- Calling `sua.udp.listen()` when the HTTP server is already running is refused with a clear error
  rather than quietly starting a second loop over the same sockets.

Handlers run on the loop thread, like HTTP handlers, with the same consequence: a slow handler
delays the next datagram. `sua.server.workers(n)` does not shard UDP sockets, and the design note
says so rather than pretending otherwise — the kernel would deliver each datagram to one worker
arbitrarily, which is wrong for any protocol that keeps per-peer state, and TURN and STUN both do.

## 3. Multicast

```bantu
$s = sua.udp.socket({});
sua.udp.bind($s, "0.0.0.0:5353");
sua.udp.join_group($s, "224.0.0.251");     // mDNS
```

`join_group($sock, group, interface?)` and `leave_group($sock, group, interface?)`. The interface
defaults to "any", which lets the kernel choose; a host with several interfaces can name one by
address (IPv4) or by index (IPv6).

- IPv4 uses `IP_ADD_MEMBERSHIP` / `IP_DROP_MEMBERSHIP` with `ip_mreq`.
- IPv6 uses `IPV6_JOIN_GROUP` / `IPV6_LEAVE_GROUP` with `ipv6_mreq`, and the socket must have been
  created with `{"family": "ipv6"}` — a group address is checked against the socket's family, so
  joining an IPv4 group on an IPv6 socket is an error with a message that says which is which,
  rather than the bare `EINVAL` the OS returns.
- **The socket must be bound before joining.** Joining an unbound socket succeeds on Linux and
  fails on macOS, so the check is ours and the error names the fix.
- Binding to the group address rather than `0.0.0.0` works on BSD and macOS and fails on Linux, so
  the documentation says to bind the wildcard address and join afterwards, which works everywhere.

Windows takes the same option names through `ws2tcpip.h`; what differs is that `ip_mreq` uses
`imr_interface` as a `struct in_addr` there too, so one code path serves both once the address is
built with `inet_pton` rather than the older `inet_addr`.

## 4. What this deliberately does not add

- **No STUN or TURN implementation.** They are protocols, not primitives, and they belong in a Bantu
  package on top of these calls — which is what the feature request itself proposed.
- **No broadcast helper.** `SO_BROADCAST` is one option and nothing in the request asked for it; it
  can join later with a use case behind it.
- **No per-source rate limiting.** A UDP server that needs it needs it shaped to its protocol, and a
  wrong default here would silently drop legitimate traffic.
