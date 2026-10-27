"use client";

import { useState } from "react";
import { PostCard } from "./PostCard";
import type { FeedResponse, Post } from "@/lib/types";

export function FeedList({
  initial,
  initialBefore,
}: {
  initial: Post[];
  initialBefore: number | null;
}) {
  const [posts, setPosts] = useState(initial);
  const [before, setBefore] = useState(initialBefore);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(initialBefore === null);

  async function more() {
    if (busy || before === null) return;
    setBusy(true);
    const res = await fetch(`/api/proxy/feed?limit=20&before=${before}`);
    const data = (await res.json()) as FeedResponse | { ok: false; error: string };
    setBusy(false);
    if (!("posts" in data)) return;
    if (data.posts.length === 0) {
      setDone(true);
      return;
    }
    // Deduplicate by id: a cursor page can overlap when something is posted
    // between requests, and React keys must stay unique.
    const seen = new Set(posts.map((p) => p.id));
    const fresh = data.posts.filter((p) => !seen.has(p.id));
    setPosts([...posts, ...fresh]);
    setBefore(data.nextBefore);
    if (data.nextBefore === null || fresh.length === 0) setDone(true);
  }

  function drop(id: number) {
    setPosts((cur) => cur.filter((p) => p.id !== id));
  }

  if (posts.length === 0) {
    return (
      <div className="empty">
        <p><strong>Nothing here yet.</strong></p>
        <p className="faint">
          Post something, or find people on Discover. The feed only ever shows
          accounts in your own age band.
        </p>
      </div>
    );
  }

  return (
    <div className="stack">
      {posts.map((p) => (
        <PostCard key={p.id} post={p} onGone={drop} />
      ))}
      {!done && (
        <button type="button" className="btn btn-ghost" onClick={more} disabled={busy}>
          {busy ? "Loading…" : "Show more"}
        </button>
      )}
      {done && posts.length > 0 && <p className="faint">That is everything for now.</p>}
    </div>
  );
}
