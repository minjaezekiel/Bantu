"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Avatar } from "./Avatar";
import { CallOverlay } from "./CallOverlay";
import { useRealtime } from "@/lib/realtime";
import { capture, createPeer, insecureReason, release } from "@/lib/call";
import type { CallState, PeerHandle } from "@/lib/call";
import { timeAgo } from "@/lib/format";
import type {
  ChatMessage, HistoryResponse, InboundFrame, PublicUser, ThreadSummary,
} from "@/lib/types";

/**
 * Messaging and calls, in one component.
 *
 * They are together because they share one WebSocket and one notion of "who
 * am I talking to". Splitting them would mean two sockets or a context whose
 * only consumer is the other half.
 *
 * The WebRTC objects live in refs, not state: a peer connection is not
 * renderable data, and putting it in state means a re-render can replace a
 * live connection. What goes in state is only what the UI draws.
 */
export function Messenger({
  me, initialThreads, openWith,
}: {
  me: PublicUser;
  initialThreads: ThreadSummary[];
  openWith: string | null;
}) {
  const [threads, setThreads] = useState<ThreadSummary[]>(initialThreads);
  const [active, setActive] = useState<string | null>(
    openWith ?? initialThreads[0]?.with.username ?? null,
  );
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [peerUser, setPeerUser] = useState<PublicUser | null>(null);
  const [online, setOnline] = useState<Record<string, boolean>>({});
  const [typingFrom, setTypingFrom] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // ── call state ────────────────────────────────────────────────────────────
  const [callState, setCallState] = useState<CallState>("idle");
  const [callPeer, setCallPeer] = useState<PublicUser | null>(null);
  const [callVideo, setCallVideo] = useState(false);
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [muted, setMuted] = useState(false);
  const [cameraOff, setCameraOff] = useState(false);
  const [callNote, setCallNote] = useState<string | null>(null);

  const peerRef = useRef<PeerHandle | null>(null);
  const localRef = useRef<MediaStream | null>(null);
  const callWith = useRef<string | null>(null);
  const pendingOffer = useRef<RTCSessionDescriptionInit | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const typingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const iceServers = useRef<RTCIceServer[]>([]);

  const activeRef = useRef<string | null>(active);
  useEffect(() => { activeRef.current = active; }, [active]);

  // ── teardown, used by every exit from a call ──────────────────────────────
  const teardown = useCallback(() => {
    peerRef.current?.close();
    peerRef.current = null;
    release(localRef.current);
    localRef.current = null;
    setLocalStream(null);
    setRemoteStream(null);
    setMuted(false);
    setCameraOff(false);
    callWith.current = null;
    pendingOffer.current = null;
  }, []);

  // Any state the call machine can be in where a NEW incoming call should be
  // refused as busy. Used by the button's disabled state too, so the UI and
  // the logic cannot disagree about whether a call is in progress.


  // ── frames in ─────────────────────────────────────────────────────────────
  const onFrame = useCallback((f: InboundFrame) => {
    switch (f.type) {
      case "welcome":
        setThreads(f.conversations);
        setOnline({});
        return;

      case "presence":
        setOnline((cur) => ({ ...cur, [f.username]: f.online }));
        return;

      case "message": {
        const who = f.with;
        if (who === activeRef.current) {
          setMessages((cur) => (cur.some((m) => m.id === f.message.id)
            ? cur : [...cur, f.message]));
          sendRef.current?.({ type: "read", to: who });
        }
        setThreads((cur) => bumpThread(cur, who, f.message.body, f.from,
                                       who === activeRef.current));
        return;
      }

      case "message-sent":
        setMessages((cur) => (cur.some((m) => m.id === f.message.id)
          ? cur : [...cur, f.message]));
        setNote(f.note);
        if (f.message.status === "live") {
          setThreads((cur) => bumpThread(cur, f.with, f.message.body, null, true));
        }
        return;

      case "read":
        setMessages((cur) => cur.map((m) => (m.mine && !m.readAt
          ? { ...m, readAt: new Date().toISOString() } : m)));
        return;

      case "typing":
        setTypingFrom(f.on ? f.with : null);
        if (f.on) {
          if (typingTimer.current) clearTimeout(typingTimer.current);
          // Cleared on a timer as well as on the "off" frame: the sender may
          // close the tab mid-sentence and never send one.
          typingTimer.current = setTimeout(() => setTypingFrom(null), 6000);
        }
        return;

      case "call-ringing":
        // A repeat of the SAME ringing is ignored, not rejected. The frame can
        // legitimately arrive more than once — the hub sends it to every
        // socket the user has open, so a second tab gets a copy — and the
        // first version of this branch answered the duplicate with a reject,
        // which cancelled the call the user was in the middle of accepting.
        if (callWith.current === f.with) return;
        // A different caller while one call is already up is genuinely busy.
        if (callWith.current) { sendRef.current?.({ type: "call-reject", to: f.with }); return; }
        callWith.current = f.with;
        pendingOffer.current = f.sdp;
        setCallPeer(f.from);
        setCallVideo(f.video);
        setCallState("incoming");
        setCallNote(null);
        return;

      case "call-answer":
        void (async () => {
          const handle = peerRef.current;
          if (!handle) return;
          // Only in have-local-offer. setRemoteDescription(answer) throws in
          // any other signalling state, and a duplicate answer frame would
          // otherwise tear down a connection that was already working.
          if (handle.pc.signalingState !== "have-local-offer") return;
          try {
            await handle.setRemote(f.sdp);
          } catch {
            setCallNote("The other side's answer could not be applied.");
            setCallState("failed");
            return;
          }
          setCallState("connecting");
        })();
        return;

      case "call-ice":
        void peerRef.current?.addCandidate(f.candidate);
        return;

      case "call-reject":
        setCallNote(f.reason === "offline"
          ? "They are not online right now."
          : f.reason === "busy" ? "They are already on a call."
          : "Call declined.");
        setCallState("ended");
        teardown();
        setTimeout(() => { setCallState("idle"); setCallNote(null); }, 2600);
        return;

      case "call-end":
        setCallState("ended");
        teardown();
        setTimeout(() => setCallState("idle"), 900);
        return;

      case "error":
        setNote(f.message);
        return;
    }
  }, [teardown]);

  const wire = useRealtime(onFrame);
  const sendRef = useRef(wire.send);
  useEffect(() => { sendRef.current = wire.send; }, [wire.send]);
  useEffect(() => {
    if (wire.config) iceServers.current = wire.config.iceServers;
  }, [wire.config]);

  // ── history for the active thread ─────────────────────────────────────────
  useEffect(() => {
    if (!active) { setMessages([]); setPeerUser(null); return; }
    let cancelled = false;
    setLoading(true);
    setNote(null);
    (async () => {
      const res = await fetch(
        `/api/proxy/conversations/${encodeURIComponent(active)}/messages?limit=60`);
      const data = (await res.json()) as HistoryResponse | { ok: false; error: string };
      if (cancelled) return;
      setLoading(false);
      if (!("messages" in data)) { setNote(data.error); setMessages([]); return; }
      setMessages(data.messages);
      setPeerUser(data.with);
      sendRef.current({ type: "read", to: active });
      setThreads((cur) => cur.map((t) =>
        (t.with.username === active ? { ...t, unread: 0 } : t)));
    })();
    return () => { cancelled = true; };
  }, [active]);

  // Pinned to the bottom on every new message, which is what a chat should do.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, typingFrom]);

  // The camera must be released if this component ever unmounts mid-call.
  useEffect(() => () => teardown(), [teardown]);

  // ── sending ───────────────────────────────────────────────────────────────
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const body = draft.trim();
    if (!body || !active) return;
    setDraft("");
    setNote(null);
    sendRef.current({ type: "typing", to: active, on: false });

    // The socket first; HTTP if it is not open. The same gate and the same
    // moderation run either way, so the fallback is not a weaker path.
    if (wire.send({ type: "message", to: active, body })) return;

    const res = await fetch(
      `/api/proxy/conversations/${encodeURIComponent(active)}/messages`,
      { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ body }) });
    const data = (await res.json()) as
      { ok?: boolean; message?: ChatMessage; note?: string | null; error?: string };
    if (!data.ok || !data.message) { setNote(data.error ?? "That did not send."); return; }
    setMessages((cur) => [...cur, data.message as ChatMessage]);
    setNote(data.note ?? null);
  }

  function onDraft(v: string) {
    setDraft(v);
    if (!active) return;
    sendRef.current({ type: "typing", to: active, on: v.length > 0 });
  }

  // ── placing a call ────────────────────────────────────────────────────────
  async function place(video: boolean) {
    if (!active) return;
    const blocked = insecureReason();
    if (blocked) { setCallNote(blocked); setCallState("failed");
                   setTimeout(() => setCallState("idle"), 5000); return; }

    setCallPeer(peerUser);
    setCallVideo(video);
    setCallNote(null);
    setCallState("requesting-media");
    let got;
    try {
      got = await capture(video);
    } catch {
      setCallNote("Your browser would not give this page a microphone. Check the address-bar permissions.");
      setCallState("failed");
      setTimeout(() => setCallState("idle"), 5000);
      return;
    }
    localRef.current = got.stream;
    setLocalStream(got.stream);
    setCallVideo(got.video);
    if (got.downgraded) setCallNote("No camera available — placing a voice call instead.");

    callWith.current = active;
    const handle = createPeer(
      iceServers.current, got.stream,
      (c) => sendRef.current({ type: "call-ice", to: active, candidate: c }),
      (s) => setRemoteStream(s),
      (st) => {
        if (st === "connected") setCallState("connected");
        else if (st === "failed") {
          setCallNote(wire.config?.hasTurn
            ? "The connection failed."
            : "The connection failed. Both of you may be behind a NAT that needs a TURN relay — see the README.");
          setCallState("failed");
        }
      });
    peerRef.current = handle;

    const offer = await handle.pc.createOffer();
    await handle.pc.setLocalDescription(offer);
    setCallState("ringing");
    sendRef.current({ type: "call-offer", to: active, sdp: offer, video: got.video });
  }

  async function accept() {
    const who = callWith.current;
    const offer = pendingOffer.current;
    if (!who || !offer) return;
    // Consumed up front, so a double tap on Answer cannot build two peer
    // connections for one call.
    pendingOffer.current = null;
    setCallState("requesting-media");
    let got;
    try {
      got = await capture(callVideo);
    } catch {
      sendRef.current({ type: "call-reject", to: who });
      setCallNote("Your browser would not give this page a microphone.");
      setCallState("failed");
      teardown();
      setTimeout(() => setCallState("idle"), 4000);
      return;
    }
    localRef.current = got.stream;
    setLocalStream(got.stream);

    const handle = createPeer(
      iceServers.current, got.stream,
      (c) => sendRef.current({ type: "call-ice", to: who, candidate: c }),
      (s) => setRemoteStream(s),
      (st) => {
        if (st === "connected") setCallState("connected");
        else if (st === "failed") { setCallNote("The connection failed."); setCallState("failed"); }
      });
    peerRef.current = handle;

    await handle.setRemote(offer);
    const answer = await handle.pc.createAnswer();
    await handle.pc.setLocalDescription(answer);
    setCallState("connecting");
    sendRef.current({ type: "call-answer", to: who, sdp: answer });
  }

  function decline() {
    const who = callWith.current;
    if (who) sendRef.current({ type: "call-reject", to: who });
    setCallState("idle");
    teardown();
  }

  function hangUp() {
    const who = callWith.current;
    if (who) sendRef.current({ type: "call-end", to: who });
    setCallState("idle");
    setCallNote(null);
    teardown();
  }

  function toggleMute() {
    const s = localRef.current;
    if (!s) return;
    const next = !muted;
    for (const t of s.getAudioTracks()) t.enabled = !next;
    setMuted(next);
  }

  function toggleCamera() {
    const s = localRef.current;
    if (!s) return;
    const next = !cameraOff;
    for (const t of s.getVideoTracks()) t.enabled = !next;
    setCameraOff(next);
  }

  const inCall = callState !== "idle" && callState !== "ended";

  return (
    <div className="msg-layout">
      {/* ── threads ───────────────────────────────────────────────────── */}
      <aside className="card card-tight">
        <div className="row" style={{ marginBottom: "var(--s-3)" }}>
          <h2 style={{ margin: 0, fontSize: "var(--step-1)" }}>Messages</h2>
          <div className="spacer" />
          <span className="wire" title={`Connection: ${wire.state}`}>
            <span className="wire-dot" data-state={wire.state} />
            <span className="faint">
              {wire.state === "open" ? "live"
                : wire.state === "connecting" ? "connecting" : "offline"}
            </span>
          </span>
        </div>

        {threads.length === 0 ? (
          <div className="empty" style={{ padding: "var(--s-5) var(--s-3)" }}>
            <p style={{ margin: 0 }}><strong>No conversations yet.</strong></p>
            <p className="faint" style={{ margin: "var(--s-2) 0 0" }}>
              You can message someone once you both follow each other. Find
              people on Discover.
            </p>
          </div>
        ) : (
          <div className="thread-list">
            {threads.map((t) => (
              <button
                key={t.id}
                type="button"
                className="thread"
                aria-current={t.with.username === active}
                onClick={() => setActive(t.with.username)}
              >
                <Avatar user={t.with} online={online[t.with.username]} />

                <span className="thread-body">
                  <span className="thread-name">
                    {t.with.displayName || t.with.username}
                  </span>
                  <span className="thread-preview">
                    {t.preview || "Say hello"}
                  </span>
                </span>
                {t.unread > 0 && <span className="thread-unread">{t.unread}</span>}
              </button>
            ))}
          </div>
        )}
      </aside>

      {/* ── the active thread ─────────────────────────────────────────── */}
      {active ? (
        <section className="chat">
          <header className="chat-head">
            <Avatar user={peerUser} online={online[active]} />
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: 600 }}>
                {peerUser?.displayName || active}
              </div>
              <div className="faint">
                {online[active] ? "Online now"
                  : peerUser ? `${peerUser.age} · ${peerUser.gender}` : ""}
              </div>
            </div>
            <div className="spacer" />
            <button type="button" className="icon-btn" title="Voice call"
                    aria-label="Start a voice call"
                    disabled={inCall || wire.state !== "open"}
                    onClick={() => void place(false)}>☎</button>
            <button type="button" className="icon-btn icon-btn-accent" title="Video call"
                    aria-label="Start a video call"
                    disabled={inCall || wire.state !== "open"}
                    onClick={() => void place(true)}>🎥</button>
          </header>

          <div className="chat-scroll" ref={scrollRef}>
            {loading && <div className="skeleton" style={{ height: "3rem" }} />}
            {!loading && messages.length === 0 && (
              <p className="faint" style={{ textAlign: "center", margin: "auto" }}>
                No messages yet. Say something.
              </p>
            )}
            {messages.map((m) => (
              <div
                key={m.id}
                className={
                  m.kind === "call" ? "bubble bubble-call"
                  : m.status === "blocked" ? "bubble bubble-blocked"
                  : m.status === "held" ? "bubble bubble-held"
                  : m.mine ? "bubble bubble-me" : "bubble bubble-them"
                }
              >
                {m.body}
                <span className="bubble-meta">
                  {timeAgo(m.createdAt)}
                  {m.status === "blocked" && " · not sent"}
                  {m.status === "held" && " · held for review"}
                  {m.mine && m.status === "live" && m.readAt ? " · read" : ""}
                </span>
              </div>
            ))}
          </div>

          {typingFrom === active && (
            <div className="typing">
              {peerUser?.displayName || active} is typing…
            </div>
          )}

          {note && (
            <div style={{ padding: "0 var(--s-3)" }}>
              <div className="notice notice-warn">{note}</div>
            </div>
          )}

          <form className="chat-compose" onSubmit={submit}>
            <label className="sr-only" htmlFor="draft">Message</label>
            <textarea
              id="draft"
              className="textarea"
              placeholder="Write a message…"
              rows={1}
              maxLength={2000}
              value={draft}
              onChange={(e) => onDraft(e.target.value)}
              onKeyDown={(e) => {
                // Enter sends, Shift+Enter is a newline — what every
                // messaging app does, and what fingers expect.
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void submit(e as unknown as React.FormEvent);
                }
              }}
            />
            <button type="submit" className="icon-btn icon-btn-accent"
                    aria-label="Send" disabled={!draft.trim()}>➤</button>
          </form>
        </section>
      ) : (
        <section className="empty">
          <p><strong>Pick a conversation.</strong></p>
          <p className="faint">
            Messaging opens once you and the other person both follow each other.
          </p>
        </section>
      )}

      <CallOverlay
        state={callState}
        peer={callPeer}
        video={callVideo}
        localStream={localStream}
        remoteStream={remoteStream}
        muted={muted}
        cameraOff={cameraOff}
        note={callNote}
        onAccept={() => void accept()}
        onDecline={decline}
        onHangUp={hangUp}
        onToggleMute={toggleMute}
        onToggleCamera={toggleCamera}
      />
    </div>
  );
}

/** Move a thread to the top with a new preview, creating it if it is new. */
function bumpThread(
  cur: ThreadSummary[], username: string, preview: string,
  who: PublicUser | null, isActive: boolean,
): ThreadSummary[] {
  const idx = cur.findIndex((t) => t.with.username === username);
  const now = new Date().toISOString();
  if (idx === -1) {
    if (!who) return cur;
    return [{ id: -Date.now(), with: who, lastMessageAt: now,
              preview, unread: isActive ? 0 : 1 }, ...cur];
  }
  const existing = cur[idx];
  if (!existing) return cur;
  const updated: ThreadSummary = {
    ...existing,
    preview,
    lastMessageAt: now,
    unread: isActive ? 0 : existing.unread + 1,
  };
  return [updated, ...cur.filter((_, i) => i !== idx)];
}
