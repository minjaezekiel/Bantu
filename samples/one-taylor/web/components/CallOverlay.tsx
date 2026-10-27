"use client";

import { useEffect, useRef, useState } from "react";
import { Avatar } from "./Avatar";
import { release } from "@/lib/call";
import type { CallState } from "@/lib/call";
import type { PublicUser } from "@/lib/types";

/**
 * The call UI. It owns no WebRTC state — Messenger does — so that a
 * re-render here cannot tear down a live connection.
 */
export function CallOverlay({
  state, peer, video, localStream, remoteStream, muted, cameraOff, note,
  onAccept, onDecline, onHangUp, onToggleMute, onToggleCamera,
}: {
  state: CallState;
  peer: PublicUser | null;
  video: boolean;
  localStream: MediaStream | null;
  remoteStream: MediaStream | null;
  muted: boolean;
  cameraOff: boolean;
  note: string | null;
  onAccept: () => void;
  onDecline: () => void;
  onHangUp: () => void;
  onToggleMute: () => void;
  onToggleCamera: () => void;
}) {
  const localRef = useRef<HTMLVideoElement | null>(null);
  const remoteRef = useRef<HTMLVideoElement | null>(null);
  const [seconds, setSeconds] = useState(0);

  // srcObject is a property, not an attribute, so it cannot be set in JSX.
  useEffect(() => {
    if (localRef.current && localRef.current.srcObject !== localStream) {
      localRef.current.srcObject = localStream;
    }
  }, [localStream]);
  useEffect(() => {
    if (remoteRef.current && remoteRef.current.srcObject !== remoteStream) {
      remoteRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  useEffect(() => {
    if (state !== "connected") { setSeconds(0); return; }
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [state]);

  // Escape hangs up. A modal with no keyboard exit is a trap, and a call is
  // the worst possible thing to be trapped in.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        if (state === "incoming") onDecline();
        else onHangUp();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state, onDecline, onHangUp]);

  if (state === "idle" || state === "ended") return null;

  const label: Record<CallState, string> = {
    idle: "", ended: "",
    "requesting-media": "Asking for your camera and microphone…",
    ringing: "Ringing…",
    incoming: video ? "Incoming video call" : "Incoming voice call",
    connecting: "Connecting…",
    connected: formatDuration(seconds),
    failed: "Could not connect",
  };
  const dot: Record<CallState, string> = {
    idle: "", ended: "",
    "requesting-media": "connecting", ringing: "ringing", incoming: "ringing",
    connecting: "connecting", connected: "connected", failed: "failed",
  };

  const showRemoteVideo = video && state === "connected" && remoteStream !== null;

  return (
    <div
      className="call-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label={video ? "Video call" : "Voice call"}
    >
      <div className="call-panel">
        <div className="call-stage">
          {showRemoteVideo ? (
            <video ref={remoteRef} autoPlay playsInline />
          ) : (
            <div className="call-voice">
              <div className={state === "ringing" || state === "incoming" ? "call-pulse" : ""}
                   style={{ borderRadius: "999px" }}>
                <Avatar user={peer} large />
              </div>
              <div>
                <div className="call-name">
                  {peer?.displayName || peer?.username || "Calling"}
                </div>
                <div style={{ opacity: 0.75, fontSize: "var(--step--1)" }}>
                  {video ? "Video call" : "Voice call"}
                </div>
              </div>
            </div>
          )}

          {/* The local preview only appears once there is something to show. */}
          {video && !cameraOff && localStream && (
            <div className="call-self">
              <video ref={localRef} autoPlay playsInline muted />
            </div>
          )}
        </div>

        <div className="call-status">
          <span className="call-dot" data-state={dot[state]} />
          <span>{label[state]}</span>
        </div>

        {note && (
          <div style={{ padding: "0 var(--s-4)" }}>
            <div className="notice notice-warn" style={{ marginTop: "var(--s-2)" }}>
              {note}
            </div>
          </div>
        )}

        <div className="call-bar">
          {state === "incoming" ? (
            <>
              <button type="button" className="icon-btn icon-btn-danger icon-btn-lg"
                      onClick={onDecline} aria-label="Decline">✕</button>
              <button type="button" className="icon-btn icon-btn-accent icon-btn-lg"
                      onClick={onAccept} aria-label="Answer">✓</button>
            </>
          ) : (
            <>
              <button type="button" className="icon-btn icon-btn-lg"
                      onClick={onToggleMute} aria-pressed={muted}
                      aria-label={muted ? "Unmute" : "Mute"}>
                {muted ? "🔇" : "🎙"}
              </button>
              {video && (
                <button type="button" className="icon-btn icon-btn-lg"
                        onClick={onToggleCamera} aria-pressed={cameraOff}
                        aria-label={cameraOff ? "Turn camera on" : "Turn camera off"}>
                  {cameraOff ? "📷" : "🎥"}
                </button>
              )}
              <button type="button" className="icon-btn icon-btn-danger icon-btn-lg"
                      onClick={onHangUp} aria-label="Hang up">✕</button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function formatDuration(total: number): string {
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Re-exported so Messenger can release tracks without importing both files. */
export { release };
