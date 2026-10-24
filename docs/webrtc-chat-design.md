# A WebRTC chat that actually places calls

`samples/webrtc-chat` claimed messaging, voice and video. It did none of them, and it did not
run: `server.b` had four parse errors, so the sample had never once executed. This is the design
of the replacement — what the server does, what the browser does, which failures are real and
which are the platform's, and how to live-test it from two browsers and from a phone.

## 1. What was there

```
$ bantu run server.b
  [error] line 27, col 18: Expected '{' (got: 'return')
  [error] line 33, col 1: Unexpected token: '}'
  [error] line 57, col 1: Expected ')' after arguments (got: 'sua')
  [error] line 61, col 1: Unexpected token: '}'
  [FATAL] [LINT] compilation blocked by 4 error(s)
```

Beyond the syntax, the architecture was wrong in a way that could not have worked:

- **Handlers were invoked, not passed.** `sua.server.get("/health", health())` calls `health()`
  immediately and registers its *return value* as the handler. The two route registrations also
  sat **inside** `create_call()`'s body, so they could never run.
- **`$req` and `$res` were read as globals** by `health()` and `create_call()`, which take no
  parameters. sua passes them as arguments.
- **It signalled over `POST`.** Signalling is bidirectional and server-initiated: the callee has
  to be *told* a call is incoming. A POST endpoint cannot push, so there was no path by which one
  browser could ever learn about the other.
- **It created WebRTC peers on the server.** `sua.webrtc.peer(...)` makes a *server-side* peer. In
  a browser-to-browser call the peers are the two browsers; the server must only relay their SDP
  and ICE. The sample generated server-side offers that no browser ever consumed.
- **The browser never created an `RTCPeerConnection` at all.** `send()` POSTed to `/send` and then
  printed the message into the sender's own log. The other browser received nothing. There was no
  `getUserMedia`, no `<video>`, no audio path.

So: not a working sample with bugs, but a façade. The rewrite keeps the directory and the premise
and replaces the contents.

## 2. Division of labour

```
  Browser A                    Bantu (sua)                    Browser B
  ─────────                    ───────────                    ─────────
  RTCPeerConnection            signalling relay               RTCPeerConnection
  getUserMedia                 + peer roster                  getUserMedia
       │                        + chat fan-out                     │
       │   ws: offer/answer/ice      │                             │
       ├────────────────────────────►├────────────────────────────►│
       │                             │                             │
       │◄════════════ audio/video, peer-to-peer ═══════════════════►│
                      (never touches the server)
```

**The server relays; it is not a peer.** It holds no `RTCPeerConnection`, terminates no media and
never sees a frame of audio or video. `sua.webrtc.*` is deliberately unused — it is a server-side
peer API and this sample has no use for one. Media is direct between browsers, which is the whole
point of WebRTC, and the reason a call keeps working at full quality however slow the signalling
server is.

**Signalling runs over `sua.ws`, not HTTP.** A WebSocket is the only transport here that can push:
the callee learns about a call because the server sends it a frame unprompted. It also gives
presence for free — `on("disconnect")` is how the roster stays honest and how the other side of a
call learns the peer is gone.

**Chat goes over the WebSocket too, not over an `RTCDataChannel`.** A data channel only exists
while a call is up, so a chat built on one cannot deliver a message to someone you are not already
calling — which is most of the time. Media peer-to-peer and text through the server is what
production apps do, and here it also means the messaging feature is a demonstration of `sua.ws`
rather than a second thing that breaks when a call does.

## 3. The signalling protocol

Every frame is one JSON object with a `type`. Server-bound frames name a recipient in `to`;
the server stamps `from` on what it forwards, so a client can never forge a sender.

| `type` | Direction | Payload | Meaning |
|---|---|---|---|
| `hello` | → server | `name` | announce yourself; answered with `welcome` |
| `welcome` | → client | `self`, `peers` | your assigned id, and who else is here |
| `roster` | → client | `peers` | sent to everyone whenever presence changes |
| `chat` | both | `text`, `to?` | room-wide when `to` is absent, direct when present |
| `offer` | both | `to`, `sdp`, `media` | a call is being placed (`media`: `audio` or `video`) |
| `answer` | both | `to`, `sdp` | the callee accepted |
| `ice` | both | `to`, `candidate` | one trickled ICE candidate |
| `decline` | both | `to`, `reason` | callee refused, or was busy |
| `bye` | both | `to` | hang up, from either side |
| `error` | → client | `message` | a frame the server refused, and why |

**`from` is assigned, never accepted.** The server knows which socket a frame arrived on and
stamps that id. A client that sends its own `from` has it overwritten. Without this, any peer could
impersonate another in chat or inject an `answer` into someone else's call.

**Unknown recipients are reported, not dropped.** A `to` that is not connected gets an `error`
frame back. Silence here reads exactly like a network failure and is the single most confusing
thing to debug in a signalling server.

**The relay is stateless about calls.** The server does not model "a call" — it has no idea whether
an `offer` was answered. Call state lives in the two browsers, which are the only parties who know
it. The server tracks exactly two things: who is connected, and what they are called.

## 4. The call, step by step

Caller presses **voice** or **video** on a roster entry:

1. `getUserMedia({audio: true, video: <video call>})`, **before** creating the connection, so a
   permission refusal costs nothing and reports cleanly.
2. `new RTCPeerConnection({iceServers})`, add every local track.
3. `createOffer` → `setLocalDescription` → send `offer` with `media`.
4. ICE candidates arrive asynchronously and are sent as `ice` frames as they come (trickle).

Callee receives `offer`:

5. Ring. Nothing is captured yet — a ringing phone must not hold the microphone.
6. On **accept**: `getUserMedia`, build the connection, add tracks,
   `setRemoteDescription(offer)` → `createAnswer` → `setLocalDescription` → send `answer`.
   On **decline**: send `decline`; no capture ever happens.
7. `ontrack` on both sides attaches the remote stream to a media element.

**ICE candidates that arrive before the description must be queued.** `addIceCandidate` throws if
there is no remote description yet, and with trickle ICE the first candidates routinely beat the
answer. They are buffered and flushed immediately after `setRemoteDescription`. This is the single
most common WebRTC defect and it is invisible on a fast loopback — it appears exactly once the two
browsers are on different machines, which is the test that matters here.

**Glare is refused, not negotiated.** If an `offer` arrives while a call is already up, the sample
sends `decline` with `reason: "busy"`. Perfect negotiation exists and is correct, but it is a
substantial state machine whose failure modes are subtle; a demonstration is better served by a
rule a reader can check by eye.

**Hangup is explicit and idempotent.** `bye` from either side tears down the connection, stops
every local track (so the camera light goes out — a leaked track is the defect users notice most)
and returns the UI to idle. Receiving `bye` for a call that is already gone does nothing.

## 5. The constraint that decides the phone test

`navigator.mediaDevices.getUserMedia` is only available in a **secure context**. `http://localhost`
counts as one; **`http://192.168.x.x:8080` does not**. On a phone pointed at the PC's LAN address,
`navigator.mediaDevices` is `undefined` — so the microphone cannot be opened at all, and the
failure surfaces as a `TypeError` on an undefined property, which looks like a bug in the app.

**sua cannot serve HTTPS.** `sua.server` has `get/post/put/.../static/listen/workers/limits` and no
TLS of any kind; `sua.http`'s TLS options are for the *client*. Server-side TLS is a real missing
feature — it also blocks service workers, and therefore the PWA sample, over anything but
localhost — but it is an interpreter change (a non-blocking OpenSSL handshake inside the kqueue/epoll
loop, plus certificate configuration, plus three build scripts), not something to bolt onto a
sample.

So the sample does two things instead:

1. **It detects the insecure context and says precisely what to do**, naming the two working
   routes, rather than letting `getUserMedia` fail as a null dereference. Chat still works —
   it needs no permission — so the app degrades instead of dying.
2. **It ships `tls-dev-proxy.py`**, a ~90-line TLS terminator using only the Python standard
   library. It generates a self-signed certificate whose `subjectAltName` carries the machine's LAN
   IP, listens on 8443, and pipes bytes to the Bantu server on 8080. Because it proxies raw TCP it
   is transparent to the WebSocket upgrade as well as to HTTP. The phone visits
   `https://192.168.x.x:8443`, accepts the certificate once, and `getUserMedia` works — on iOS
   Safari too, which has no flag to bypass the requirement.

The alternative, for Android Chrome only, is the browser's own allowlist:
`chrome://flags/#unsafely-treat-insecure-origin-as-secure`. Documented, no extra process, no
certificate — and no help on an iPhone.

**The WebSocket `Origin` check does not get in the way.** sua compares the `Origin` authority
against the request's own `Host`, so a phone loading `http://192.168.1.50:8080` sends a matching
pair and upgrades. No `ws_allowed_origins` configuration is needed; the same holds through the TLS
proxy, which preserves `Host`.

## 6. Failures this design has to survive

| Failure | Where it shows | What the sample does |
|---|---|---|
| Insecure origin | phone on LAN over plain HTTP | detected up front, actionable banner, chat still works |
| Permission denied | any browser | named error per device kind; call cancelled, not half-open |
| No camera, mic present | laptops with the lid shut, most phones in a drawer | video call retries as audio and says so |
| ICE before description | different machines, always | candidates queued, flushed after `setRemoteDescription` |
| Autoplay blocked | Safari, Chrome without a gesture | remote element is `playsinline`, `play()` failure caught and surfaced as a tap-to-listen control |
| Self-view feedback howl | two browsers on one machine | local preview is hard-muted in markup *and* in script |
| iOS fullscreen hijack | iPhone video call | `playsinline` on every media element |
| Peer vanishes mid-call | closing a tab | `disconnect` → roster update → `bye` to the other side |
| Reload during a call | constant while developing | socket closes, server drops the peer, both sides return to idle |
| Signalling socket drops | phone sleeping, Wi-Fi roam | exponential-backoff reconnect, state shown in the UI |

## 7. Interface

Dark-first, because a call UI is looked at for a long time and in the dark. Three zones: a roster
rail, a conversation column, and a call stage that is absent until a call exists rather than an
empty black box.

- **Colours are web-safe and tested for contrast.** Every foreground/background pair in the sheet
  is checked against WCAG AA (4.5:1 for text, 3:1 for large text and UI edges). The palette is
  defined once as custom properties on `:root`, with a light theme behind
  `prefers-color-scheme: light`, so there is a single place to change it.
- **Fonts are a system stack**, not a web font: a chat app that waits on a font download shows
  blank message bubbles on the slowest connections, which is when it matters most.
- **Responsive without a framework.** One column under 720px with the roster as a sheet, two
  columns above it. The composer is pinned with `env(safe-area-inset-bottom)` so an iPhone's home
  indicator does not sit on top of the send button, and `100dvh` so the mobile URL bar does not
  crop the stage.
- **Reachable by keyboard and by screen reader.** Visible focus rings, the message list as an
  `aria-live` log, the incoming-call dialog as a focus-trapped `role="dialog"`, and every
  icon-only control labelled.

## 8. How it is tested

Automated, in `tests/webrtc_chat_test.sh`: the server starts, serves its page, and the signalling
protocol is driven over a real WebSocket by a stdlib Python client — two peers connect, exchange
`hello`/`welcome`/`roster`, relay chat room-wide and directly, carry an `offer`/`answer`/`ice`
sequence between them, get `error` for an unknown recipient, fail to forge `from`, and see the
roster shrink when one disconnects. That covers everything the server is responsible for.

What it cannot cover is the browser half — `getUserMedia`, ICE, and whether audio is audible — so
the README carries an explicit manual script for the two-browser test and the phone test, with the
expected observation at each step.
