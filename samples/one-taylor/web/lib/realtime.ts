"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  InboundFrame, OutboundFrame, RealtimeConfig, TicketResponse, WireState,
} from "./types";

/**
 * One WebSocket for the whole session: presence, messages and call signalling.
 *
 * Authentication is a TICKET, not the session token. The token lives in an
 * httpOnly cookie on the Next.js origin and the socket connects to the Bantu
 * origin, so the browser would not send it anyway — and a token in a WebSocket
 * URL is written to every proxy log on the path. The page asks its own server
 * for a ticket good for sixty seconds and one use, and spends it in the first
 * frame.
 *
 * Reconnection backs off exponentially with jitter. Without the jitter, every
 * client that was connected when the server restarted comes back at the same
 * moment and knocks it over again.
 *
 * ── Why there is a generation counter ──────────────────────────────────────
 *
 * connect() awaits two fetches before it creates the socket, and that await is
 * a window in which the component can unmount and remount — which React
 * StrictMode does on every mount in development, and a navigation does in
 * production. The first version checked `alive` only at the top, so the
 * abandoned connect() woke up, built a WebSocket anyway and installed an
 * onmessage that still called the handler.
 *
 * The result was TWO live sockets and every frame processed twice. Messages
 * were deduplicated by id so nothing looked wrong, typing indicators were
 * idempotent, and the pings were invisible — but the second copy of a
 * `call-ringing` frame hit the "already in a call" branch and auto-rejected
 * the call. Voice and video simply never connected, and the only trace was a
 * doubled frame count in the server log.
 *
 * So every connect() takes a generation number, re-checks it after each await,
 * and a socket whose generation has been superseded is closed and its frames
 * dropped. The teardown bumps the generation, which is what makes an
 * in-flight connect abandon itself.
 */

const MAX_BACKOFF_MS = 20_000;
const PING_MS = 25_000;

export interface Realtime {
  state: WireState;
  /** Returns false when the socket is not open, so callers can fall back to REST. */
  send: (frame: OutboundFrame) => boolean;
  config: RealtimeConfig | null;
  lastError: string | null;
}

export function useRealtime(onFrame: (f: InboundFrame) => void): Realtime {
  const [state, setState] = useState<WireState>("connecting");
  const [config, setConfig] = useState<RealtimeConfig | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);

  const sock = useRef<WebSocket | null>(null);
  const attempt = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pinger = useRef<ReturnType<typeof setInterval> | null>(null);
  const alive = useRef(true);
  // Bumped by the teardown and by every new connect, so an older attempt that
  // is still awaiting a fetch knows it has been superseded.
  const generation = useRef(0);

  // The handler is held in a ref so that reconnection does not depend on the
  // caller passing a stable callback. Otherwise every parent re-render tears
  // the socket down and builds a new one.
  const handler = useRef(onFrame);
  useEffect(() => { handler.current = onFrame; }, [onFrame]);

  const connect = useCallback(async () => {
    if (!alive.current) return;
    const gen = ++generation.current;
    const current = () => alive.current && gen === generation.current;
    setState("connecting");

    let cfg: RealtimeConfig | null = null;
    let ticket: string | null = null;
    try {
      const [cRes, tRes] = await Promise.all([
        fetch("/api/proxy/realtime/config"),
        fetch("/api/proxy/realtime/ticket", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: "{}",
        }),
      ]);
      const cJson = (await cRes.json()) as RealtimeConfig | { ok: false; error: string };
      const tJson = (await tRes.json()) as TicketResponse | { ok: false; error: string };
      if ("wsUrl" in cJson) { cfg = cJson; setConfig(cJson); }
      if ("ticket" in tJson) ticket = tJson.ticket;
      if (!cfg || !ticket) {
        setLastError("Could not get a connection ticket. Are you still signed in?");
        throw new Error("no ticket");
      }
    } catch {
      // Retry rather than give up: the usual cause is the backend restarting.
      if (current()) schedule();
      return;
    }

    // The await above is the window this guard exists for. A ticket fetched by
    // an abandoned attempt is simply left to expire — it is worth sixty
    // seconds and one use, and spending it would register a second socket.
    if (!current()) return;

    let ws: WebSocket;
    try {
      ws = new WebSocket(cfg.wsUrl);
    } catch {
      if (current()) schedule();
      return;
    }
    // Checked once more, because `new WebSocket` is synchronous but the
    // scheduling around it is not.
    if (!current()) { try { ws.close(); } catch { /* never opened */ } return; }
    sock.current = ws;

    ws.onopen = () => {
      if (!current()) { try { ws.close(); } catch { /* ignore */ } return; }
      attempt.current = 0;
      setLastError(null);
      ws.send(JSON.stringify({ type: "hello", ticket } satisfies OutboundFrame));
      setState("open");
      if (pinger.current) clearInterval(pinger.current);
      // A keepalive, so an idle socket is not reaped by something in the middle.
      pinger.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" } satisfies OutboundFrame));
        }
      }, PING_MS);
    };

    ws.onmessage = (ev) => {
      // A superseded socket's frames are dropped rather than handled. This is
      // the line whose absence broke every call.
      if (!current()) return;
      let frame: InboundFrame;
      try {
        frame = JSON.parse(String(ev.data)) as InboundFrame;
      } catch {
        return;
      }
      if (frame.type === "error") setLastError(frame.message);
      if (frame.type === "pong") return;
      handler.current(frame);
    };

    ws.onclose = () => {
      if (pinger.current) { clearInterval(pinger.current); pinger.current = null; }
      if (sock.current === ws) sock.current = null;
      if (!current()) return;
      setState("closed");
      schedule();
    };

    ws.onerror = () => { /* onclose always follows; handled there */ };
  }, []);

  const schedule = useCallback(() => {
    if (!alive.current) return;
    if (timer.current) clearTimeout(timer.current);
    const n = Math.min(attempt.current++, 6);
    // Exponential, with jitter: without the jitter every client that was
    // connected when the server restarted comes back at the same instant.
    const base = Math.min(500 * 2 ** n, MAX_BACKOFF_MS);
    const delay = base / 2 + Math.random() * (base / 2);
    timer.current = setTimeout(() => { void connect(); }, delay);
  }, [connect]);

  useEffect(() => {
    alive.current = true;
    void connect();
    return () => {
      alive.current = false;
      // Bumping the generation is what makes an in-flight connect() abandon
      // itself rather than installing a second socket behind our back.
      generation.current += 1;
      if (timer.current) clearTimeout(timer.current);
      if (pinger.current) clearInterval(pinger.current);
      const s = sock.current;
      sock.current = null;
      if (s) { s.onclose = null; s.onmessage = null; s.close(); }
    };
  }, [connect]);

  const send = useCallback((frame: OutboundFrame) => {
    const s = sock.current;
    if (!s || s.readyState !== WebSocket.OPEN) return false;
    s.send(JSON.stringify(frame));
    return true;
  }, []);

  return useMemo(() => ({ state, send, config, lastError }),
                 [state, send, config, lastError]);
}
