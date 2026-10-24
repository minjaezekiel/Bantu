/* ===========================================================================
   Bantu Talk — browser half of the sample.

   The server relays; this file is the WebRTC peer. Everything that makes a
   call actually work rather than nearly work is here, and the awkward parts
   are commented with WHY, because every one of them is a bug that shows up
   only under a specific test:

     * ICE candidates that arrive before the remote description  (two machines)
     * autoplay refusal                                          (Safari)
     * tracks left running after hangup                          (camera light)
     * getUserMedia on a non-secure origin                       (phone on LAN)
     * self-view feedback howl                                   (two tabs, one mic)

   Protocol: docs/webrtc-chat-design.md
   =========================================================================== */
'use strict';

const $ = (id) => document.getElementById(id);

const ui = {
  joinDialog: $('joinDialog'), joinForm: $('joinForm'), joinName: $('joinName'), joinHint: $('joinHint'),
  ringDialog: $('ringDialog'), ringName: $('ringName'), ringKind: $('ringKind'),
  ringAvatar: $('ringAvatar'), ringAccept: $('ringAccept'), ringDecline: $('ringDecline'),
  status: $('status'), statusText: $('statusText'), selfChip: $('selfChip'),
  banner: $('banner'),
  roster: $('roster'), rosterToggle: $('rosterToggle'), rosterClose: $('rosterClose'),
  scrim: $('scrim'), peerList: $('peerList'), rosterEmpty: $('rosterEmpty'),
  onlineCount: $('onlineCount'), shareUrl: $('shareUrl'),
  messages: $('messages'), composer: $('composer'), text: $('text'),
  dmBadge: $('dmBadge'), dmBadgeText: $('dmBadgeText'),
  stage: $('stage'), stagePeer: $('stagePeer'), stageState: $('stageState'),
  remoteVideo: $('remoteVideo'), localVideo: $('localVideo'),
  voiceFace: $('voiceFace'), voiceAvatar: $('voiceAvatar'), voicePeer: $('voicePeer'),
  level: $('level'), tapAudio: $('tapAudio'),
  btnMute: $('btnMute'), btnCam: $('btnCam'), btnHangup: $('btnHangup'),
};

/* Public STUN only helps across NAT. On one LAN — which is the test this
   sample is built for — host candidates connect with no STUN at all, so a
   lookup failure must never be fatal. */
const RTC_CONFIG = {
  iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }],
};

const RING_TIMEOUT_MS = 45000;

const state = {
  ws: null,
  self: null,              // { id, name }
  peers: [],               // [{ id, name }]
  dm: null,                // peer id for direct messages, null = whole room
  call: null,              // see startCall()
  incoming: null,          // pending offer frame
  retry: 0,
  closing: false,
  pingTimer: null,
  stableTimer: null,
  ringTimer: null,
  levelTimer: null,
};

/* ───────────────────────── small helpers ─────────────────────────────────── */

const initials = (name) => (name || '?').trim().slice(0, 1).toUpperCase() || '?';
const peerName = (id) => (state.peers.find((p) => p.id === id) || {}).name || id;

function clock(ts) {
  const d = ts ? new Date(ts) : new Date();
  return isNaN(d) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function setStatus(stateName, text) {
  ui.status.dataset.state = stateName;
  ui.statusText.textContent = text;
}

function showBanner(html, kind) {
  ui.banner.innerHTML = html;
  ui.banner.dataset.kind = kind || 'warn';
  ui.banner.hidden = false;
}

/* Messages are built with textContent, never innerHTML: a display name or a
   message body is someone else's input and must never be parsed as markup. */
function addMessage({ who, text, mine, system, at, direct }) {
  const li = document.createElement('li');
  li.className = 'msg' + (mine ? ' me' : '') + (system ? ' sys' : '');

  if (!system) {
    const head = document.createElement('p');
    head.className = 'msg-who';
    head.textContent = mine ? 'You' : who;
    if (direct) {
      const tag = document.createElement('span');
      tag.className = 'tag-dm';
      tag.textContent = 'direct';
      head.appendChild(tag);
    }
    const t = document.createElement('span');
    t.className = 'msg-time';
    t.textContent = clock(at);
    head.appendChild(t);
    li.appendChild(head);
  }

  const bubble = document.createElement('p');
  bubble.className = 'bubble';
  bubble.textContent = text;
  li.appendChild(bubble);

  const pinned = ui.messages.scrollTop + ui.messages.clientHeight >= ui.messages.scrollHeight - 48;
  ui.messages.appendChild(li);
  if (pinned || mine) ui.messages.scrollTop = ui.messages.scrollHeight;
}

const sysMessage = (text) => addMessage({ text, system: true });

/* ───────────────────────── secure-context gate ───────────────────────────── */
/*
 * getUserMedia exists only in a secure context. http://localhost is one;
 * http://192.168.x.x is NOT — so on a phone pointed at the PC's LAN address
 * navigator.mediaDevices is undefined and a naive call dies with
 * "Cannot read properties of undefined", which reads like an app bug.
 * Checked once, up front, and reported with the two routes that work.
 */
function mediaAvailable() {
  return !!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function gateMedia() {
  if (mediaAvailable()) return true;
  const host = location.hostname;
  showBanner(
    '<strong>Calling is unavailable on this address.</strong> Browsers only grant a microphone or ' +
    'camera in a secure context, and <code>' + location.origin + '</code> is not one. ' +
    'Messaging still works. To place calls from this device, either run the TLS front end on the ' +
    'server machine and open <code>https://' + host + ':8443</code>, or — Android Chrome only — add ' +
    '<code>' + location.origin + '</code> to <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code>.',
    'bad'
  );
  return false;
}

/* ───────────────────────── signalling socket ─────────────────────────────── */

let lastClose = null;

/* What a close code actually means for someone trying to use this. */
function describeClose(ev) {
  switch (ev.code) {
    case 1000: return 'The server closed it normally.';
    case 1001: return 'The page is going away.';
    case 1002: return 'Protocol error — the server broke the WebSocket framing.';
    case 1003: return 'The server rejected the kind of data sent.';
    case 1005: return 'Closed with no status.';
    case 1006:
      return 'The connection failed before a clean close. Either nothing is listening at that ' +
             'address, or the server\u2019s handshake reply was rejected by the browser.';
    case 1007: return 'The server sent text that was not valid UTF-8.';
    case 1008: return 'The server refused the connection by policy.';
    case 1009: return 'A message was too large.';
    case 1011: return 'The server hit an internal error.';
    case 1015: return 'TLS failed — the certificate was not accepted.';
    default:   return 'Closed with code ' + ev.code + '.';
  }
}

function wsUrl() {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return scheme + '//' + location.host + '/ws';
}

function send(obj) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return false;
  state.ws.send(JSON.stringify(obj));
  return true;
}

function connect() {
  setStatus('wait', 'connecting…');
  let ws;
  try {
    ws = new WebSocket(wsUrl());
  } catch (err) {
    setStatus('down', 'cannot connect');
    scheduleReconnect();
    return;
  }
  state.ws = ws;

  ws.onopen = () => {
    /* [found] Resetting the counter here made the backoff useless: a socket
       that opens and is dropped immediately went straight back to a 500ms
       retry, so a failing connection was hammered twice a second forever
       (58 reconnects from one phone in one session). The counter is only
       cleared once the socket has actually SURVIVED a few seconds. */
    if (state.stableTimer) clearTimeout(state.stableTimer);
    state.stableTimer = setTimeout(() => { state.retry = 0; }, 5000);
    lastClose = null;
    /* Only hide the banner if it was OUR failure notice -- the insecure-origin
       warning must survive, since it is still true. */
    if (mediaAvailable()) ui.banner.hidden = true;
    setStatus('live', 'connected');
    console.info('[bantu-talk] socket open at', wsUrl(), '- sending hello');
    send({ type: 'hello', name: state.self.name });

    /* Keepalive. The server reaps an idle socket after five minutes and the
       WebSocket API gives JavaScript no way to send a protocol PING, so the
       traffic has to be ours. 25s is comfortably inside every intermediary's
       idle window too -- a phone on mobile data behind a NAT often has 60s. */
    if (state.pingTimer) clearInterval(state.pingTimer);
    state.pingTimer = setInterval(() => {
      if (!send({ type: 'ping' }) && state.pingTimer) {
        clearInterval(state.pingTimer);
        state.pingTimer = null;
      }
    }, 25000);
  };

  ws.onmessage = (ev) => {
    let frame;
    try { frame = JSON.parse(ev.data); } catch (_) { return; }
    handle(frame);
  };

  ws.onclose = (ev) => {
    /* The close CODE is the only thing that says why, and throwing it away is
       why a failure here showed up as nothing but "Not connected". 1006 means
       the browser never got a usable handshake; 1002 a protocol error; 1015
       TLS. Surfaced in the UI as well as the console, because the console is
       not where someone testing a chat app is looking. */
    const why = describeClose(ev);
    console.warn('[bantu-talk] socket closed:', ev.code, ev.reason || '(no reason)',
                 'clean=' + ev.wasClean, '->', why, 'url=' + wsUrl());
    if (state.pingTimer) { clearInterval(state.pingTimer); state.pingTimer = null; }
    if (state.closing) return;
    lastClose = { code: ev.code, reason: ev.reason, why: why };
    setStatus('down', 'reconnecting…');

    /* [found] This used to end the call whenever the signalling socket
       dropped. That is wrong, and it was the reason calls "worked for a while
       and then hung up on their own": WebRTC media is peer-to-peer and keeps
       flowing with the server completely gone. Signalling is only needed to
       SET UP a call, and to carry `bye` at the end. So a reconnect is now
       invisible to a call in progress -- the media never notices.

       The banner is also only shown once the socket has been down long
       enough to matter, so a 200ms blip during a call does not throw a red
       alert over the stage. */
    if (state.retry >= 2) {
      showBanner('<strong>Signalling socket closed (code ' + ev.code + ').</strong> ' + why +
                 ' Reconnecting — a call already running is unaffected.', 'bad');
    }
    /* The roster is stale, not empty: clearing it would make the person you
       are calling vanish from the list mid-call. */
    scheduleReconnect();
  };

  ws.onerror = () => {
    /* The error event carries no detail by design; onclose follows with the
       code, which is where this is reported. */
    console.warn('[bantu-talk] socket error on', wsUrl());
  };
}

/* Backoff, capped. A phone that sleeps and wakes reconnects on the first
   attempt; a server that is actually down is not hammered. */
function scheduleReconnect() {
  const wait = Math.min(500 * Math.pow(2, state.retry), 10000);
  state.retry += 1;
  setTimeout(() => { if (!state.closing) connect(); }, wait);
}

/* ───────────────────────── frame handling ────────────────────────────────── */

function handle(f) {
  switch (f.type) {
    case 'welcome':
      state.self = f.self;
      ui.selfChip.textContent = f.self.name;
      ui.selfChip.hidden = false;
      state.peers = f.peers || [];
      rememberNames();
      renderRoster();
      sysMessage('You joined as ' + f.self.name + '.');
      break;

    /* Presence arrives as a DELTA. The server sends the whole roster only in
       the welcome: broadcasting it on every join cost O(n^2) bytes and, in a
       room of a few hundred, filled the server's per-connection send queue
       until it started dropping peers to defend itself. See the note in
       server.b.

       All three frames end in the same applyRoster(), so the call re-binding
       below -- which is the subtle part -- has exactly one implementation. */
    case 'join':
      if (!f.peer) break;
      applyRoster(
        state.peers.some((p) => p.id === f.peer.id)
          ? state.peers.map((p) => (p.id === f.peer.id ? f.peer : p))   /* rename */
          : state.peers.concat([f.peer])
      );
      break;

    case 'leave':
      applyRoster(state.peers.filter((p) => p.id !== f.id));
      break;

    /* A full roster is still accepted: it is the cheapest possible resync and
       an older server sends nothing else. */
    case 'roster':
      applyRoster(f.peers || []);
      break;

    case 'chat':
      addMessage({
        who: f.fromName || f.from,
        text: f.text,
        mine: f.from === state.self.id,
        at: f.at,
        direct: !!f.direct,
      });
      break;

    case 'offer':   onOffer(f);   break;
    case 'answer':  onAnswer(f);  break;
    case 'ice':     onIce(f);     break;
    case 'decline': onDecline(f); break;
    case 'bye':
      if (state.call && state.call.peerId === f.from) endCall(state.call.peerName + ' hung up', { notify: false });
      else if (state.incoming && state.incoming.from === f.from) dismissRing();
      break;

    case 'pong':
      break;                       /* keepalive answered; nothing to show */

    case 'error':
      sysMessage('Server: ' + f.message);
      break;
  }
}

/* Replace the roster with `next` and react to what changed: who arrived, who
   left, and whether a call or DM was pointing at any of them. */
function applyRoster(next) {
  const selfId = state.self ? state.self.id : null;
  const before = new Set(state.peers.map((p) => p.id));
  state.peers = next;
  const now = new Set(state.peers.map((p) => p.id));
  for (const p of state.peers) if (!before.has(p.id) && p.id !== selfId) sysMessage(p.name + ' is online.');
  for (const id of before) if (!now.has(id) && id !== selfId) sysMessage(peerNameCache(id) + ' went offline.');
  rememberNames();
  /* The server does not model calls, so nobody sends `bye` for a peer that
     simply vanished. Presence is the signal -- but an id is only stable for
     the lifetime of a socket.
     [found] After either side reconnected, the peer came back under a NEW id,
     the old id was gone, and this ended a call whose audio and video were
     still flowing perfectly. A call is now re-bound to the peer's new id by
     name, and only ended when nobody by that name is left. */
  if (state.call && !now.has(state.call.peerId)) {
    const same = state.peers.find((p) => p.name === state.call.peerName);
    if (same) {
      state.call.peerId = same.id;            // reconnected under a new id
    } else {
      endCall(state.call.peerName + ' left', { notify: false });
    }
  }
  if (state.dm && !now.has(state.dm)) setDm(null);
  renderRoster();
}

/* Names of peers who have left are still needed for one message, so the last
   known name of every id seen is kept. */
const nameCache = new Map();
function rememberNames() { for (const p of state.peers) nameCache.set(p.id, p.name); }
function peerNameCache(id) { return nameCache.get(id) || id; }

/* ───────────────────────── roster rendering ──────────────────────────────── */

function renderRoster() {
  const others = state.peers.filter((p) => !state.self || p.id !== state.self.id);
  ui.onlineCount.textContent = String(state.peers.length);
  ui.rosterEmpty.hidden = others.length > 0;
  ui.peerList.replaceChildren();

  for (const p of others) {
    const li = document.createElement('li');
    li.className = 'peer';

    const av = document.createElement('div');
    av.className = 'avatar';
    av.textContent = initials(p.name);
    av.setAttribute('aria-hidden', 'true');

    const nm = document.createElement('span');
    nm.className = 'peer-name';
    nm.textContent = p.name;

    const acts = document.createElement('div');
    acts.className = 'peer-acts';
    acts.appendChild(peerButton('Message ' + p.name, ICON.chat, () => { setDm(p.id); closeRoster(); ui.text.focus(); }));
    acts.appendChild(peerButton('Voice call ' + p.name, ICON.phone, () => placeCall(p, 'audio')));
    acts.appendChild(peerButton('Video call ' + p.name, ICON.cam, () => placeCall(p, 'video')));

    li.append(av, nm, acts);
    ui.peerList.appendChild(li);
  }
}

const ICON = {
  chat:  'M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z',
  phone: 'M21 15.5v3a2 2 0 01-2.2 2 19 19 0 01-8.3-3 19 19 0 01-3-3A19 19 0 014.5 5.2 2 2 0 016.5 3h3a2 2 0 012 1.7c.1.9.4 1.8.7 2.6a2 2 0 01-.5 2.1l-1.3 1.3a15 15 0 005.9 5.9l1.3-1.3a2 2 0 012.1-.5c.8.3 1.7.6 2.6.7a2 2 0 011.7 2z',
  cam:   'M2 7h11v10H2zM13 11l7-4v10l-7-4z',
};

function peerButton(label, path, onClick) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'icon-btn call';
  b.setAttribute('aria-label', label);
  b.title = label;
  b.innerHTML = '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="' + path + '"/></svg>';
  b.addEventListener('click', onClick);
  return b;
}

function setDm(id) {
  state.dm = id;
  if (id) {
    ui.dmBadgeText.textContent = peerName(id);
    ui.dmBadge.hidden = false;
    ui.text.placeholder = 'Direct message to ' + peerName(id) + '…';
  } else {
    ui.dmBadge.hidden = true;
    ui.text.placeholder = 'Write a message…';
  }
}

/* ───────────────────────── media capture ─────────────────────────────────── */
/*
 * Capture happens BEFORE any RTCPeerConnection is built, so a refusal costs
 * nothing and leaves no half-open connection behind.
 *
 * A video call on a device with a microphone but no camera is common (a
 * laptop with the lid shut, a desktop without a webcam). Rather than fail,
 * it retries as audio and says so — a voice call is what the user wanted
 * most of.
 */
async function capture(wantVideo) {
  const describe = (err) => {
    const n = err && err.name;
    if (n === 'NotAllowedError' || n === 'SecurityError') return 'permission was refused in the browser';
    if (n === 'NotFoundError' || n === 'OverconstrainedError') return 'no suitable microphone or camera was found';
    if (n === 'NotReadableError') return 'the microphone or camera is already in use by another app';
    return (err && err.message) || 'the device could not be opened';
  };

  try {
    return {
      stream: await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        video: wantVideo ? { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } : false,
      }),
      video: wantVideo,
    };
  } catch (err) {
    if (wantVideo && (err.name === 'NotFoundError' || err.name === 'OverconstrainedError' || err.name === 'NotReadableError')) {
      sysMessage('No camera available — continuing as a voice call.');
      try {
        return { stream: await navigator.mediaDevices.getUserMedia({ audio: true, video: false }), video: false };
      } catch (err2) {
        throw new Error(describe(err2));
      }
    }
    throw new Error(describe(err));
  }
}

/* ───────────────────────── the call ─────────────────────────────────────── */

function newCall(peer, media, role) {
  return {
    peerId: peer.id, peerName: peer.name, media, role,
    pc: null, local: null,
    pendingIce: [],      // candidates that arrived before setRemoteDescription
    haveRemote: false,
    connected: false,
  };
}

function addLocalTracks(pc, call, stream) {
  for (const track of stream.getTracks()) pc.addTrack(track, stream);

  /* [found] "I cannot see the other user" on a device with no usable camera.
     addTrack() only negotiates what we SEND. A video call from a phone that
     fell back to audio (no camera, camera busy, mic-only permission) produced
     an SDP with no video m-section at all, so the other side's video was
     never transmitted -- there was nothing to carry it. The call reported
     itself connected and the stage stayed black.

     A recvonly transceiver adds that m-section explicitly, so video flows IN
     even when none flows out. */
  const want = (kind) => {
    if (stream.getTracks().some((t) => t.kind === kind)) return;   // already sending
    if (kind === 'video' && call.media !== 'video') return;        // voice call: no video line
    try { pc.addTransceiver(kind, { direction: 'recvonly' }); }
    catch (err) { console.warn('addTransceiver(' + kind + ') failed', err); }
  };
  want('audio');
  want('video');
}

function buildPeerConnection(call) {
  const pc = new RTCPeerConnection(RTC_CONFIG);
  call.pc = pc;

  pc.onicecandidate = (ev) => {
    if (ev.candidate) send({ type: 'ice', to: call.peerId, candidate: ev.candidate.toJSON() });
  };

  /* ONE stream for every remote track, built here rather than taken from
     ev.streams.
     [found] This used to assign remoteVideo.srcObject = ev.streams[0] on each
     ontrack. ontrack fires once PER TRACK, and a browser may deliver audio and
     video in SEPARATE streams -- so the second assignment detached the first,
     and whichever track arrived last was the only one attached. Accumulating
     into one stream is the canonical pattern: it cannot drop a track, it works
     whether or not the peer groups them, and because we call addTrack
     ourselves the stream's addtrack event fires, which is what re-runs the
     video/overlay decision. */
  call.remoteStream = new MediaStream();

  pc.ontrack = (ev) => {
    console.info('[bantu-talk] remote ' + ev.track.kind + ' track'
               + ' muted=' + ev.track.muted + ' state=' + ev.track.readyState);
    if (!call.remoteStream.getTracks().includes(ev.track)) {
      call.remoteStream.addTrack(ev.track);
      /* Said out loud in the transcript, not just the console: a phone has no
         console, and "did video actually arrive?" is the only question that
         matters when a call shows an avatar instead of a face. */
      sysMessage('Receiving ' + ev.track.kind + ' from ' + call.peerName + '.');
    }
    /* A track arrives muted and unmutes when the first frame lands -- that is
       the moment video becomes real, so the decision is made again then. */
    ev.track.onunmute = () => attachRemote(call.remoteStream, call);
    ev.track.onended  = () => attachRemote(call.remoteStream, call);
    attachRemote(call.remoteStream, call);
  };

  pc.onconnectionstatechange = () => {
    if (!state.call || state.call !== call) return;
    const s = pc.connectionState;
    if (s === 'connected') {
      call.connected = true;
      ui.stageState.textContent = call.media === 'video' ? 'video call — connected' : 'voice call — connected';
      /* What was actually negotiated, in one line. If this says the video
         transceiver is "inactive" or missing, the problem is the offer/answer;
         if it says sendrecv but no video track arrived, the problem is the
         other end's camera. */
      try {
        const lines = pc.getTransceivers().map((t) => {
          const kind = (t.receiver && t.receiver.track && t.receiver.track.kind)
                    || (t.sender && t.sender.track && t.sender.track.kind) || t.mid;
          return kind + ':' + (t.currentDirection || t.direction || '?');
        });
        const local = (call.local ? call.local.getTracks().map((t) => t.kind).join('+') : 'none');
        sysMessage('Negotiated ' + lines.join(', ') + ' — sending ' + local + '.');
      } catch (_) {}
    } else if (s === 'connecting') {
      ui.stageState.textContent = 'connecting…';
    } else if (s === 'failed') {
      /* Reached when no candidate pair works — different networks with no
         TURN server. Saying so beats a stage that sits on "connecting". */
      ui.stageState.textContent = 'could not connect';
      sysMessage('The media connection failed. On different networks this needs a TURN server; on one Wi-Fi network it should not happen.');
    } else if (s === 'disconnected') {
      ui.stageState.textContent = 'reconnecting…';
    }
  };

  return pc;
}

function attachRemote(stream, call) {
  if (ui.remoteVideo.srcObject !== stream) ui.remoteVideo.srcObject = stream;
  /* The remote <video> is the audio sink for a VOICE call too, so it is never
     hidden -- a display:none media element is not a dependable place to play
     audio, and some mobile browsers pause it. For a voice call it is left
     playing and covered by .voice-face, which is opaque. */
  /* [found] Whether the call "has video" was decided ONCE, on the first
     ontrack -- and ontrack fires once PER TRACK. When the audio track arrived
     first, the opaque .voice-face overlay was drawn over the video element and
     never taken down, so a perfectly good video call showed an avatar instead
     of the other person. It is now re-decided whenever the stream's tracks
     change, and again when the element reports real frame dimensions, which
     is the only proof that video is actually arriving. */
  const sync = () => {
    /* `muted` is NOT a disqualifier: a remote video track is muted until the
       first frame decodes, and waiting for unmute leaves the avatar up over a
       stream that is already arriving. readyState is the real signal. */
    const hasVideo = stream.getVideoTracks().some((t) => t.readyState !== 'ended');
    ui.voiceFace.hidden = hasVideo;
    if (hasVideo) stopLevelMeter(); else startLevelMeter(stream);
  };
  sync();
  stream.onaddtrack = sync;
  stream.onremovetrack = sync;
  ui.remoteVideo.onloadedmetadata = sync;
  ui.remoteVideo.onresize = sync;

  /* Autoplay: a call always starts from a click, so this usually succeeds.
     It still has to be handled — Safari refuses when the gesture is judged
     too old, and an unhandled rejection means a connected call with silence
     and no explanation. */
  /* play() rejects with AbortError whenever a new track interrupts a pending
     play request, which happens routinely here -- so the button is driven by
     whether the element is ACTUALLY paused a moment later, not by the promise.
     Otherwise "Tap to enable sound" sits there during a call whose audio is
     already playing, which is what it did. */
  const tryPlay = () => ui.remoteVideo.play().catch(() => {}).finally(() => {
    setTimeout(() => { ui.tapAudio.hidden = !ui.remoteVideo.paused; }, 250);
  });
  tryPlay();
  /* A track arriving after the element was already playing can leave it
     paused on iOS, so playback is nudged again when that happens. */
  ui.remoteVideo.onpause = () => { if (state.call) tryPlay(); };
}

/* The five bars under a voice call. Without them a silent call gives no clue
   whether audio is flowing at all, which is the first question during a test. */
function startLevelMeter(stream) {
  stopLevelMeter();
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return;
  let ctx;
  try { ctx = new AC(); } catch (_) { return; }
  const src = ctx.createMediaStreamSource(stream);
  const node = ctx.createAnalyser();
  node.fftSize = 512;
  src.connect(node);
  const buf = new Uint8Array(node.frequencyBinCount);
  const bars = [...ui.level.querySelectorAll('i')];

  state.levelTimer = setInterval(() => {
    node.getByteFrequencyData(buf);
    let sum = 0;
    for (const v of buf) sum += v;
    const avg = sum / buf.length / 255;
    bars.forEach((bar, i) => {
      const reach = Math.max(0, Math.min(1, avg * 3 - i * 0.14));
      bar.style.height = (5 + reach * 17).toFixed(1) + 'px';
    });
  }, 90);
  state.levelCtx = ctx;
}

function stopLevelMeter() {
  if (state.levelTimer) { clearInterval(state.levelTimer); state.levelTimer = null; }
  if (state.levelCtx) { state.levelCtx.close().catch(() => {}); state.levelCtx = null; }
}

async function placeCall(peer, media) {
  if (state.call) { sysMessage('Finish the current call first.'); return; }
  if (state.incoming) { sysMessage('Answer or decline the incoming call first.'); return; }
  if (!gateMedia()) return;

  let got;
  try {
    got = await capture(media === 'video');
  } catch (err) {
    sysMessage('Could not start the call: ' + err.message + '.');
    return;
  }

  const call = newCall(peer, got.video ? 'video' : 'audio', 'caller');
  call.local = got.stream;
  state.call = call;

  openStage(call);
  ui.stageState.textContent = 'ringing…';

  const pc = buildPeerConnection(call);
  addLocalTracks(pc, call, got.stream);

  try {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    send({ type: 'offer', to: peer.id, media: call.media, sdp: pc.localDescription });
  } catch (err) {
    sysMessage('Could not create the call: ' + err.message);
    endCall('failed to start', { notify: false });
    return;
  }

  /* An offer nobody answers must not leave the camera on forever. */
  call.timeout = setTimeout(() => {
    if (state.call === call && !call.connected) {
      endCall('no answer', { notify: true });
    }
  }, RING_TIMEOUT_MS);
}

function onOffer(f) {
  /* Glare, and the busy case. Declining with a reason is a rule a reader can
     check by eye; perfect negotiation is correct but a large state machine
     whose failure modes are subtle. */
  if (state.call || state.incoming) {
    send({ type: 'decline', to: f.from, reason: 'busy' });
    return;
  }
  state.incoming = f;
  const name = f.fromName || f.from;
  ui.ringName.textContent = name;
  ui.ringAvatar.textContent = initials(name);
  ui.ringKind.textContent = (f.media === 'video' ? 'is starting a video call' : 'is calling you');
  if (!ui.ringDialog.open) ui.ringDialog.showModal();
  ui.ringAccept.focus();

  state.ringTimer = setTimeout(() => {
    if (state.incoming === f) {
      send({ type: 'decline', to: f.from, reason: 'no answer' });
      dismissRing();
      sysMessage('Missed a call from ' + name + '.');
    }
  }, RING_TIMEOUT_MS);
}

function dismissRing() {
  state.incoming = null;
  if (state.ringTimer) { clearTimeout(state.ringTimer); state.ringTimer = null; }
  if (ui.ringDialog.open) ui.ringDialog.close();
}

async function acceptCall() {
  const f = state.incoming;
  if (!f) return;
  dismissRing();

  if (!gateMedia()) { send({ type: 'decline', to: f.from, reason: 'no media access' }); return; }

  let got;
  try {
    got = await capture(f.media === 'video');
  } catch (err) {
    sysMessage('Could not answer: ' + err.message + '.');
    send({ type: 'decline', to: f.from, reason: err.message });
    return;
  }

  const call = newCall({ id: f.from, name: f.fromName || f.from }, f.media === 'video' ? 'video' : 'audio', 'callee');
  call.local = got.stream;
  /* Candidates that arrived while the phone was ringing. They are queued, not
     applied: there is still no remote description until setRemoteDescription
     below, and addIceCandidate would throw. flushIce() drains them. */
  call.pendingIce = (f.earlyIce || []).slice();
  state.call = call;
  openStage(call);

  const pc = buildPeerConnection(call);
  addLocalTracks(pc, call, got.stream);

  try {
    await pc.setRemoteDescription(new RTCSessionDescription(f.sdp));
    call.haveRemote = true;
    await flushIce(call);
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    send({ type: 'answer', to: f.from, sdp: pc.localDescription });
  } catch (err) {
    sysMessage('Could not answer: ' + err.message);
    endCall('failed to answer', { notify: true });
  }
}

async function onAnswer(f) {
  const call = state.call;
  if (!call || call.peerId !== f.from || !call.pc) return;
  try {
    await call.pc.setRemoteDescription(new RTCSessionDescription(f.sdp));
    call.haveRemote = true;
    await flushIce(call);
  } catch (err) {
    sysMessage('The answer could not be applied: ' + err.message);
    endCall('negotiation failed', { notify: true });
  }
}

/*
 * THE defect that only appears on two machines.
 *
 * With trickle ICE the first candidates routinely arrive before the
 * answer does, and addIceCandidate() throws if there is no remote
 * description yet. On loopback the answer is so fast it never happens; put
 * the two browsers on different devices and it happens every time. So
 * candidates are queued until the description lands, then flushed in order.
 */
function onIce(f) {
  if (!f.candidate) return;

  /* The caller starts trickling the moment it has a local description --
     which is BEFORE the callee has pressed accept. At that point the callee
     has no `state.call` at all, only a ringing offer, so these candidates
     were being dropped on the floor and the call then had to fall back to
     whatever candidates happened to arrive after the answer. Buffer them
     against the pending offer and hand them to the connection on accept. */
  if (!state.call && state.incoming && state.incoming.from === f.from) {
    (state.incoming.earlyIce = state.incoming.earlyIce || []).push(f.candidate);
    return;
  }

  const call = state.call;
  if (!call || call.peerId !== f.from) return;
  if (!call.haveRemote) { call.pendingIce.push(f.candidate); return; }
  call.pc.addIceCandidate(new RTCIceCandidate(f.candidate)).catch((err) => {
    console.warn('addIceCandidate failed', err);
  });
}

async function flushIce(call) {
  const queued = call.pendingIce;
  call.pendingIce = [];
  for (const c of queued) {
    try { await call.pc.addIceCandidate(new RTCIceCandidate(c)); }
    catch (err) { console.warn('queued addIceCandidate failed', err); }
  }
}

function onDecline(f) {
  if (!state.call || state.call.peerId !== f.from) return;
  const why = f.reason === 'busy' ? ' is already on a call' : (f.reason ? ' declined (' + f.reason + ')' : ' declined');
  sysMessage(state.call.peerName + why + '.');
  endCall('declined', { notify: false });
}

function openStage(call) {
  ui.stage.hidden = false;
  ui.stagePeer.textContent = call.peerName;
  ui.stageState.textContent = 'connecting…';
  ui.voicePeer.textContent = call.peerName;
  ui.voiceAvatar.textContent = initials(call.peerName);
  ui.voiceFace.hidden = call.media === 'video';
  ui.tapAudio.hidden = true;

  const hasLocalVideo = call.local && call.local.getVideoTracks().length > 0;
  /* muted is set in markup AND here. Two browsers on one machine with a live
     self-view is an instant feedback howl, and markup alone has been known to
     be lost when srcObject is reassigned. */
  ui.localVideo.muted = true;
  ui.localVideo.srcObject = hasLocalVideo ? call.local : null;
  ui.localVideo.hidden = !hasLocalVideo;
  if (hasLocalVideo) ui.localVideo.play().catch(() => {});

  /* Driven by the tracks actually captured, not by what the call was called:
     a video call that fell back to audio has no camera to toggle. */
  ui.btnCam.hidden = !hasLocalVideo;
  ui.btnMute.setAttribute('aria-pressed', 'false');
  ui.btnMute.setAttribute('aria-label', 'Mute microphone');
  ui.btnCam.setAttribute('aria-pressed', 'false');
  ui.btnCam.setAttribute('aria-label', 'Turn camera off');
  sysMessage((call.role === 'caller' ? 'Calling ' : 'In a call with ') + call.peerName +
             ' (' + call.media + ').');
}

function endCall(reason, opts) {
  const call = state.call;
  if (!call) return;
  state.call = null;

  if (call.timeout) clearTimeout(call.timeout);
  if (opts && opts.notify) send({ type: 'bye', to: call.peerId });

  /* Every track is stopped explicitly. Closing the RTCPeerConnection does
     NOT release the camera — the light stays on, which is the single most
     noticeable bug a call app can ship. */
  if (call.local) for (const t of call.local.getTracks()) t.stop();
  if (call.pc) {
    call.pc.onicecandidate = null;
    call.pc.ontrack = null;
    call.pc.onconnectionstatechange = null;
    try { call.pc.close(); } catch (_) {}
  }

  stopLevelMeter();
  if (call.remoteStream) { for (const t of call.remoteStream.getTracks()) call.remoteStream.removeTrack(t); }
  ui.remoteVideo.srcObject = null;
  ui.localVideo.srcObject = null;
  ui.stage.hidden = true;
  ui.tapAudio.hidden = true;

  sysMessage('Call with ' + call.peerName + ' ended' + (reason ? ' — ' + reason : '') + '.');
}

/* ───────────────────────── wiring ───────────────────────────────────────── */

ui.joinForm.addEventListener('submit', (ev) => {
  ev.preventDefault();
  const name = ui.joinName.value.trim().slice(0, 24);
  if (!name) return;
  state.self = { id: null, name };
  try { localStorage.setItem('bantu-talk-name', name); } catch (_) {}
  ui.joinDialog.close();
  gateMedia();
  connect();
  ui.text.focus();
});

ui.composer.addEventListener('submit', (ev) => {
  ev.preventDefault();
  const text = ui.text.value.trim();
  if (!text) return;
  const frame = { type: 'chat', text };
  if (state.dm) frame.to = state.dm;
  if (!send(frame)) { sysMessage('Not connected — message not sent.'); return; }
  ui.text.value = '';
});

ui.dmBadge.addEventListener('click', () => setDm(null));
ui.ringAccept.addEventListener('click', () => { acceptCall(); });
ui.ringDecline.addEventListener('click', () => {
  if (state.incoming) send({ type: 'decline', to: state.incoming.from, reason: 'declined' });
  dismissRing();
});
/* Escape on the ring dialog must decline, not silently drop the call. */
ui.ringDialog.addEventListener('cancel', (ev) => {
  ev.preventDefault();
  if (state.incoming) send({ type: 'decline', to: state.incoming.from, reason: 'declined' });
  dismissRing();
});

ui.btnHangup.addEventListener('click', () => endCall('you hung up', { notify: true }));

ui.btnMute.addEventListener('click', () => {
  const call = state.call;
  if (!call || !call.local) return;
  const tracks = call.local.getAudioTracks();
  const muted = tracks.length > 0 && tracks[0].enabled;   // about to become muted
  for (const t of tracks) t.enabled = !muted;
  ui.btnMute.setAttribute('aria-pressed', String(muted));
  ui.btnMute.setAttribute('aria-label', muted ? 'Unmute microphone' : 'Mute microphone');
});

ui.btnCam.addEventListener('click', () => {
  const call = state.call;
  if (!call || !call.local) return;
  const tracks = call.local.getVideoTracks();
  const off = tracks.length > 0 && tracks[0].enabled;
  for (const t of tracks) t.enabled = !off;
  ui.btnCam.setAttribute('aria-pressed', String(off));
  ui.btnCam.setAttribute('aria-label', off ? 'Turn camera on' : 'Turn camera off');
  ui.localVideo.hidden = off;
});

ui.tapAudio.addEventListener('click', () => {
  ui.remoteVideo.play().then(() => { ui.tapAudio.hidden = true; }).catch(() => {});
});

const openRoster = () => {
  ui.roster.classList.add('open');
  ui.scrim.hidden = false;
  ui.rosterToggle.setAttribute('aria-expanded', 'true');
};
const closeRoster = () => {
  ui.roster.classList.remove('open');
  ui.scrim.hidden = true;
  ui.rosterToggle.setAttribute('aria-expanded', 'false');
};
ui.rosterToggle.addEventListener('click', () => {
  ui.roster.classList.contains('open') ? closeRoster() : openRoster();
});
ui.rosterClose.addEventListener('click', closeRoster);
ui.scrim.addEventListener('click', closeRoster);

/* A reload mid-call would otherwise leave the other side ringing until the
   socket's own timeout. */
window.addEventListener('pagehide', () => {
  state.closing = true;
  if (state.call) send({ type: 'bye', to: state.call.peerId });
  if (state.call && state.call.local) for (const t of state.call.local.getTracks()) t.stop();
});

/* ───────────────────────── boot ─────────────────────────────────────────── */

ui.shareUrl.textContent = location.origin;

if (!window.RTCPeerConnection) {
  showBanner('<strong>This browser has no WebRTC support,</strong> so calls are not possible. Messaging will still work.', 'bad');
}

try {
  const saved = localStorage.getItem('bantu-talk-name');
  if (saved) ui.joinName.value = saved;
} catch (_) {}

if (!mediaAvailable()) {
  ui.joinHint.innerHTML = 'Heads up: this address cannot grant a microphone, so only messaging will work here. ' +
                          'See the note after you join.';
}

ui.joinDialog.showModal();
ui.joinName.focus();
ui.joinName.select();
