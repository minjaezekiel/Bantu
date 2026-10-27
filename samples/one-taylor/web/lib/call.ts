"use client";

/**
 * WebRTC, from the browser's side.
 *
 * The Bantu server is not a peer: it relays the offer, the answer and the ICE
 * candidates and never sees a frame of media. Everything below runs here.
 *
 * Three things that are easy to get wrong and expensive to debug:
 *
 *   Capture happens BEFORE the RTCPeerConnection is built. If the user
 *   refuses the camera, we have spent nothing and can say so plainly instead
 *   of leaving a half-built connection and a confusing failure.
 *
 *   ICE candidates are trickled, and ones that arrive before the remote
 *   description is set are QUEUED. addIceCandidate throws if there is no
 *   remote description yet, and the answerer always receives candidates
 *   before it has finished answering.
 *
 *   Every track is stopped explicitly on hang-up. Closing the peer connection
 *   does not release the camera, and a camera light that stays on after a
 *   call is the single most alarming bug this feature can have.
 */

export type CallState =
  | "idle" | "requesting-media" | "ringing" | "incoming"
  | "connecting" | "connected" | "ended" | "failed";

export interface MediaRefusal {
  kind: "denied" | "missing" | "insecure" | "downgraded";
  message: string;
}

/**
 * getUserMedia exists only in a secure context. http://localhost is one; a LAN
 * address is not, which is why a phone pointed at this machine can message but
 * cannot call.
 */
export function canCall(): boolean {
  return Boolean(
    typeof window !== "undefined" &&
    window.isSecureContext &&
    window.RTCPeerConnection &&
    navigator.mediaDevices?.getUserMedia,
  );
}

export function insecureReason(): string | null {
  if (typeof window === "undefined") return null;
  if (!window.RTCPeerConnection) return "This browser does not support calls.";
  if (!window.isSecureContext) {
    return "Calls need a secure connection. This page is not on https:// or localhost, so the browser will not give it a camera or microphone.";
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return "This browser will not give the page a camera or microphone.";
  }
  return null;
}

export interface Captured {
  stream: MediaStream;
  video: boolean;
  downgraded: boolean;
}

/**
 * Asks for what was requested, and falls back to audio when a camera is asked
 * for and refused or absent. A video call that becomes a voice call is a far
 * better outcome than a call that does not happen.
 */
export async function capture(wantVideo: boolean): Promise<Captured> {
  const audio = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };
  if (!wantVideo) {
    return { stream: await navigator.mediaDevices.getUserMedia({ audio, video: false }),
             video: false, downgraded: false };
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio,
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    });
    return { stream, video: true, downgraded: false };
  } catch {
    const stream = await navigator.mediaDevices.getUserMedia({ audio, video: false });
    return { stream, video: false, downgraded: true };
  }
}

export interface PeerHandle {
  pc: RTCPeerConnection;
  /** Queue a candidate until the remote description exists, then flush. */
  addCandidate: (c: RTCIceCandidateInit) => Promise<void>;
  setRemote: (d: RTCSessionDescriptionInit) => Promise<void>;
  close: () => void;
}

export function createPeer(
  iceServers: RTCIceServer[],
  local: MediaStream,
  onIce: (c: RTCIceCandidateInit) => void,
  onRemoteStream: (s: MediaStream) => void,
  onState: (s: RTCPeerConnectionState) => void,
): PeerHandle {
  const pc = new RTCPeerConnection({ iceServers });

  for (const track of local.getTracks()) pc.addTrack(track, local);

  const remote = new MediaStream();
  pc.ontrack = (ev) => {
    for (const track of ev.streams[0]?.getTracks() ?? [ev.track]) {
      // Guard against adding the same track twice: ontrack fires per track and
      // a renegotiation can repeat one.
      if (!remote.getTracks().some((t) => t.id === track.id)) remote.addTrack(track);
    }
    onRemoteStream(remote);
  };

  pc.onicecandidate = (ev) => {
    if (ev.candidate) onIce(ev.candidate.toJSON());
  };
  pc.onconnectionstatechange = () => onState(pc.connectionState);

  const pending: RTCIceCandidateInit[] = [];
  let haveRemote = false;

  async function flush() {
    while (pending.length) {
      const c = pending.shift();
      if (!c) break;
      try { await pc.addIceCandidate(c); } catch { /* a stale candidate is not fatal */ }
    }
  }

  return {
    pc,
    async addCandidate(c) {
      if (!haveRemote) { pending.push(c); return; }
      try { await pc.addIceCandidate(c); } catch { /* as above */ }
    },
    async setRemote(d) {
      await pc.setRemoteDescription(d);
      haveRemote = true;
      await flush();
    },
    close() {
      pc.ontrack = null;
      pc.onicecandidate = null;
      pc.onconnectionstatechange = null;
      try { pc.close(); } catch { /* already closed */ }
    },
  };
}

/** Stops every track. Closing the peer connection does NOT free the camera. */
export function release(...streams: (MediaStream | null | undefined)[]) {
  for (const s of streams) {
    if (!s) continue;
    for (const t of s.getTracks()) {
      try { t.stop(); } catch { /* already stopped */ }
    }
  }
}
