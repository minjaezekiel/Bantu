# Persistent connections, chunked bodies and cacheable files

How the sua server's HTTP/1.1 layer works after the four fixes measured in the audit of
2026-09-19, and why each one is shaped the way it is. The event loop underneath is unchanged;
everything here sits between the loop and the route handlers.

## What was wrong

| | measured |
|---|---|
| Every response closed the connection | `ab -k` reported `Keep-Alive requests: 0` |
| The close was not announced | no `Connection` header, so an HTTP/1.1 client reuses a socket the server is closing |
| Pipelined requests were dropped | two requests in one write, one response back |
| Chunked request bodies vanished | handler saw `null`, no error |
| Static files were re-read and re-copied every time | 2 MB file: 79 req/s, 162 MB/s, memory-copy bound |
| A small static file was slower than running a handler | 17.9k vs 23.0k req/s |

The cost of the first one, measured on the same machine with the same client against a reference
server: **16.3k req/s closing per request, 79.7k req/s with keep-alive.** About 5×.

## 1. Persistent connections

**The decision is per request, from the request itself.** HTTP/1.1 keeps the connection unless the
request says `Connection: close`; HTTP/1.0 closes unless it says `Connection: keep-alive`. That is
the rule in RFC 7230 §6.3, and it is what every client already expects.

**Every response now states which it did.** `Connection: keep-alive` or `Connection: close`,
always. The absence of that header was its own defect: a client following the spec reused a socket
the server was about to close, and paid for a retry.

Closing still wins over keeping whenever the server cannot be sure the connection is clean: a
malformed request, a body the server refused, a payload over the limit, a 431 or 413, or a
backpressure drop. A connection whose framing is in doubt is never reused — a wrong guess there is
how request smuggling starts.

**Pipelining falls out of it.** The read loop already kept unconsumed bytes; it now keeps
dispatching while another complete request is buffered. Responses stay in arrival order because
the loop is one thread and dispatch is synchronous.

The exception is a suspendable route, whose handler can finish after the loop has moved on. While a
connection has a suspended handler, the next buffered request waits, and the connection is put on a
resume queue that the loop drains after each event batch. Without that wait, response two could
overtake response one on the same socket, and a pipelining client matches responses to requests by
order alone. The queue is a loop, not recursion: a client can pipeline thousands of requests into
one buffer, and recursion there would be a stack overflow reachable from the network.

**Idle keep-alive connections get their own timeout class.** The reaper walks each class in
last-activity order and stops at the first entry still inside its timeout, which is only valid when
every connection in a list shares one timeout. So a third list holds connections that have answered
at least one request and have no partial request buffered: `keepAliveTimeoutMs`, default 65 s. A
connection that has sent nothing yet stays in the 10 s header-timeout class, which is what keeps the
slowloris defence intact — otherwise opening a socket and saying nothing would buy 65 s instead of
10 s.

## 2. Chunked request bodies

Framing is decided in one place, before dispatch, because the loop has to know when a request is
complete and the handler has to see the same bytes.

- `Transfer-Encoding: chunked` walks the chunk framing to find the end of the body. It never
  searches for the terminator as a byte string: a chunk's own payload may contain `0\r\n\r\n`, and
  taking that as the end is how a smuggled request gets in.
- The decoded size is checked against `maxBodyBytes` as it grows, not after, so a chunked body
  cannot exhaust memory by never ending.
- **Both `Content-Length` and `Transfer-Encoding` present is refused with 400, and the connection
  closes.** RFC 7230 §3.3.3 says to ignore the length, but the disagreement itself is the signature
  of a smuggling attempt against a proxy in front. Refusing is the answer that cannot be smuggled
  through. Before this change the request hung until it timed out.
- Trailers after the last chunk are skipped, not passed to the handler.

## 3. Static files

Three changes, all inside `bantuServeStaticFile`.

**Conditional requests.** Each file gets a validator from its size and modification time, sent as
`ETag` and `Last-Modified`. `If-None-Match` and `If-Modified-Since` are answered `304` with no body.
A browser reloading a page it already has now transfers headers instead of megabytes.

**Ranges.** A single `Range: bytes=...` is answered `206` with `Content-Range`; an unsatisfiable one
gets `416`. This is what video seeking and resumable downloads need. Multi-range requests are served
whole, which the spec allows and which is what most servers do in practice.

**Fewer copies, and a small cache.** The file used to be copied about four times on the way out:
`ifstream` to `stringstream`, to a `string`, into an `ostringstream` with the headers, and into the
connection's output buffer. It is now read once into a string and appended to the output buffer
after the headers, and files up to 1 MB are held in a cache keyed by path and validated against
size and mtime on every request, so an edited file is never served stale. The cache has a byte
budget; over it, it is cleared wholesale rather than evicted in order, because a static directory
that overflows the budget is better served by a CDN than by a smarter policy here.

## 4. Protocol details

- **`HEAD` runs the matching `GET` route** and sends the headers with no body, `Content-Length`
  intact. Returning a body to `HEAD` is forbidden by RFC 7231 §4.3.2, and it corrupts any client
  that then reads the next response off the same connection — which only matters now that
  connections are reused.
- **A known path with an unknown method is `405`**, with the `Allow` header listing what it does
  accept, instead of the `404` it used to give.
- **Path parameters are percent-decoded**, so `/user/a%20b` hands the handler `a b`. Static paths
  are decoded too, and the traversal check runs after decoding, where `%2e%2e` is visible as `..`.
- **Absolute-form request targets** (`GET http://host/path`) have the scheme and authority stripped,
  as RFC 7230 §5.3.2 requires of every server.

## What is deliberately not here

- **Handler-level concurrency.** One slow handler still blocks its worker; `sua.server.workers(n)`
  still reduces that to 1/n. Fixing it properly means moving handler execution off the loop thread,
  which needs an interpreter that can run in parallel. That is a separate piece of work with a much
  larger blast radius.
- **HTTP/2, TLS and SSE.** Unchanged, still absent; a reverse proxy in front remains the answer.
- **A response body streamed from a handler.** Handlers still build the whole body in memory. Static
  files are the case that actually gets large, and they no longer copy it four times.

## Compatibility

Every existing handler keeps working unchanged, and no API changed. The visible differences are that
responses carry a `Connection` header, that a client which reuses a connection now gets served on it,
and that `HEAD` and unknown-method requests answer correctly. A program that depended on the
connection closing after each response is the one thing that could notice; sending
`Connection: close` on the request restores exactly the old behaviour.
