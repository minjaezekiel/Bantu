# Bantu Talk — messaging, voice and video over WebRTC

A working two-party call app. The **browsers** are the WebRTC peers; the **Bantu
server** relays their SDP and ICE over one WebSocket and never touches a frame
of audio or video.

```
bantu run server.b          # → http://localhost:8080
```

Open that in two browser windows, give each a name, and the two appear in each
other's list with a message button, a voice button and a video button.

Design and rationale: [`docs/webrtc-chat-design.md`](../../docs/webrtc-chat-design.md).

---

## What it does

| | |
|---|---|
| **Messaging** | Room-wide, or direct to one person. Works whether or not a call is up, because it goes over the signalling socket rather than a data channel. |
| **Voice calls** | `getUserMedia({audio})`, peer-to-peer. A level meter shows the remote talker's volume, so a silent call is distinguishable from a broken one. |
| **Video calls** | Camera plus microphone, with a mirrored self-view. Falls back to a voice call, and says so, when there is no camera. |
| **Presence** | Who is online, live. A peer who closes their tab disappears, and the other side of their call is torn down. |
| **Mute / camera off** | Toggles the track rather than the connection, so the call survives. |

Controls: **Esc** declines an incoming call. The composer sends on Enter. Every
icon-only button is labelled for a screen reader, the transcript is an
`aria-live` log, and both dialogs are real `<dialog>` elements, so focus is
trapped and returned for you.

---

## Live test 1 — two browsers on this machine

1. `bantu run server.b`
2. Open `http://localhost:8080` in **two** windows. Use two different browsers,
   or one normal and one private window — two tabs of the same profile share a
   camera and will fight over it.
3. Join as `alice` in one and `bob` in the other. Each should now show the other
   in **Online**, and the transcript should say `bob is online.`
4. **Messaging.** Send from alice. It appears in both, right-aligned for the
   sender. Click the message icon next to a name to switch to a direct message —
   the composer shows a blue pill with that name, and the message is tagged
   `direct` and arrives only there. Click the pill to go back to the room.
5. **Voice call.** Click the phone icon on bob's row. Bob's window rings with
   Accept / Decline. Accept, and grant the microphone.
   - Expected: both say `voice call — connected`, and the level bars move when
     you speak.
   - **Wear headphones, or mute one side.** Two windows on one machine with two
     live microphones is a feedback loop.
6. **Video call.** Click the camera icon. Accept and grant the camera.
   - Expected: each window shows the other's camera full-width with its own
     camera as a small mirrored inset.
   - Toggle **mute** and **camera off**: the other side keeps the call, the
     video freezes or the audio stops, and the button reads as pressed.
7. **Hang up** from either side. Both return to the list, and **the camera light
   goes out** — if it stays on, a track was leaked.
8. **Close one window mid-call.** The other should say the peer left, tear the
   call down, and drop them from Online within a second.

## Live test 2 — your phone calling the PC

The catch, stated plainly: **browsers only give out a microphone or camera in a
secure context.** `http://localhost` counts as one. `http://192.168.1.50:8080`
does not — on a phone, `navigator.mediaDevices` is simply `undefined`. sua
cannot serve HTTPS, so something has to terminate TLS in front of it. The app
detects the problem and tells you which of these to do rather than failing with
a null.

### The way that works everywhere, iPhone included

```bash
bantu run server.b                      # terminal 1
python3 tls-dev-proxy.py --target 8080  # terminal 2
```

The proxy prints the URL to use, e.g. `https://192.168.1.50:8443`. It is
standard-library Python only, and it generates a self-signed certificate whose
`subjectAltName` carries your LAN IP.

1. Phone and PC on the **same Wi-Fi**.
2. Open that `https://…:8443` URL on the phone. It will warn once, because the
   certificate is self-signed:
   - **iOS Safari** — Show Details → visit this website
   - **Android Chrome** — Advanced → Proceed
3. Join as `phone`. On the PC, open **either** `http://localhost:8080` or the
   same `https://…:8443` URL, and join as `demo`.

   Both reach the same room: the proxy forwards into the same Bantu process, so
   there is one peer registry regardless of which URL a peer arrived on.
   `localhost` is the easier choice on the PC — it is already a secure context,
   so there is no certificate warning there. (Verified: a peer on
   `http://localhost:8080` and a peer on `https://192.168.1.7:8443` appear in
   each other's roster and `/health` reports `online: 2`.)
4. From the phone, press the phone icon on `demo`. Accept on the PC.
5. **Speak into the phone.** You should hear yourself from the PC's speakers,
   and the PC's level bars should move with your voice.
   - If you hear a howl, that is the PC's microphone picking up its own speakers.
     Headphones on the PC, or mute the PC side — the call is still proving
     phone → PC audio.
6. Video from a phone: the camera icon uses the front camera, and the video stays
   inline rather than going fullscreen.

### The way with no extra process (Android only)

Chrome will treat one origin as secure on request:
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` → add
`http://192.168.1.50:8080` → Relaunch. There is no equivalent on iOS, which is
why the proxy exists.

### If a call connects but you hear nothing

- A **Tap to enable sound** button on the call stage means the browser blocked
  autoplay. Tap it.
- `could not connect` on the stage means no ICE candidate pair worked. On one
  Wi-Fi network that should not happen; across different networks it needs a
  TURN server, which a demo cannot provide for you.
- Check both ends are on the **same URL**. `localhost` on the PC and the LAN IP
  on the phone are different origins.

---

## Files

```
webrtc-chat/
├── server.b              # signalling relay + static files. No WebRTC peer here.
├── tls-dev-proxy.py      # TLS front end, so a phone can get a microphone
├── public/
│   ├── index.html        # app shell, both dialogs
│   ├── styles.css        # tokens, dark + light, WCAG AA checked
│   └── app.js            # the RTCPeerConnection half
└── bantu.json
```

## The protocol

One JSON object per frame, over the WebSocket at any path. `to` names a
recipient; the server stamps `from` itself, so a peer cannot post as another.

| `type` | Direction | Payload |
|---|---|---|
| `hello` | → server | `name` |
| `welcome` | → client | `self`, `peers` |
| `roster` | → client | `peers` |
| `chat` | both | `text`, `to?` |
| `offer` | both | `to`, `sdp`, `media` |
| `answer` | both | `to`, `sdp` |
| `ice` | both | `to`, `candidate` |
| `decline` | both | `to`, `reason` |
| `bye` | both | `to` |
| `error` | → client | `message` |

`GET /health` returns who is online, which is the quickest way to see whether
the server agrees with your browsers.

## Tests

```bash
BANTU=../../bantu-src/compiler/build/bantu bash ../../tests/webrtc_chat_test.sh
```

35 assertions drive the server with two real WebSocket clients: roster, chat
fan-out, direct messages, the relay of every call frame, that `from` cannot be
forged, that an unknown recipient is reported rather than dropped, and that a
peer who leaves actually leaves. The browser half — `getUserMedia`, ICE, whether
audio is audible — is what the two scripts above are for.

## Notes

- **`sua.webrtc.*` is deliberately unused.** It makes a *server-side* peer. In a
  browser-to-browser call the peers are the browsers, and the server's only job
  is to carry their SDP and ICE.
- **The server holds no call state.** It tracks who is connected and what they
  are called, nothing more. Whether an offer was answered is known only to the
  two browsers, which is where that belongs.
- **One call at a time per peer.** A second incoming offer is declined with
  `busy`. Perfect negotiation is the general answer and a much larger state
  machine; a sample is better served by a rule you can check by eye.
- **STUN is only for crossing NAT.** On one LAN, host candidates connect with no
  STUN at all, and the public server listed in `app.js` is not required.
- **Not production signalling.** No authentication, no rooms, no persistence,
  and the TLS proxy's certificate is self-signed by design.
