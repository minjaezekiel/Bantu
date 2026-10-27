"use client";

import { useEffect, useState } from "react";

interface QueueEvent {
  id: number;
  decision: string;
  detail: string;
  scores: string;
  created_at: string;
  actor: string;
}

interface QueuePost {
  id: number;
  body: string;
  authorBand: string;
  authorAge: number;
  status: string;
  createdAt: string;
  author: { username: string } | null;
  events: QueueEvent[];
}

interface Brigaded {
  id: number;
  body: string;
  author_band: string;
  status: string;
  n: number;
}

interface Queue {
  ok: true;
  blocked: QueuePost[];
  review: QueuePost[];
  brigaded: Brigaded[];
  counts: { blocked: number; review: number; brigaded: number };
}

export function ModerationQueue() {
  const [q, setQ] = useState<Queue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<number | null>(null);

  async function load() {
    const res = await fetch("/api/proxy/admin/moderation/queue");
    const data = (await res.json()) as Queue | { ok: false; error: string };
    if (!("blocked" in data)) {
      setError(
        data.error === "not found"
          ? "This queue needs ONE_TAYLOR_ADMIN_TOKEN set on the Next.js server as well as the backend."
          : data.error,
      );
      return;
    }
    setQ(data);
  }

  useEffect(() => { void load(); }, []);

  async function decide(id: number, action: "release" | "remove") {
    setBusy(id);
    await fetch(`/api/proxy/admin/moderation/${id}/${action}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ note: `${action} from the web queue` }),
    });
    setBusy(null);
    await load();
  }

  if (error) return <div className="notice notice-danger">{error}</div>;
  if (!q) return <div className="skeleton" style={{ height: "8rem" }} />;

  const section = (title: string, posts: QueuePost[], note: string) => (
    <section className="stack">
      <h2>{title} <span className="badge">{posts.length}</span></h2>
      <p className="faint">{note}</p>
      {posts.length === 0 ? (
        <div className="empty">Nothing waiting.</div>
      ) : (
        posts.map((p) => (
          <article className="card stack-tight" key={p.id}>
            <div className="row">
              <span className={p.authorBand === "minor" ? "badge badge-minor" : "badge badge-adult"}>
                {p.authorBand === "minor" ? "under 18" : "18+"} · {p.authorAge}
              </span>
              <span className="faint">@{p.author?.username ?? "unknown"}</span>
              <span className="faint">#{p.id}</span>
              <div className="spacer" />
              <span className="badge">{p.status}</span>
            </div>
            <p className="post-body">{p.body || <em className="faint">no text</em>}</p>
            {p.events.length > 0 && (
              <div className="stack-tight">
                {p.events.map((e) => (
                  <div className="mono faint" key={e.id}>
                    {e.decision}
                    {e.detail ? ` — ${e.detail}` : ""}
                    {e.scores ? ` · ${e.scores}` : ""}
                  </div>
                ))}
              </div>
            )}
            <div className="row">
              <button className="btn btn-sm" disabled={busy === p.id}
                      onClick={() => decide(p.id, "release")}>
                Publish it
              </button>
              <button className="btn btn-danger btn-sm" disabled={busy === p.id}
                      onClick={() => decide(p.id, "remove")}>
                Keep it down
              </button>
            </div>
          </article>
        ))
      )}
    </section>
  );

  return (
    <div className="stack">
      {section("Blocked", q.blocked,
        "A lexicon term matched exactly, so these were refused outright. The text was kept so the decision can be reviewed and appealed.")}
      {section("Held for review", q.review,
        "The classifier flagged these but is not precise enough to delete a post on its own. Only the author can see them until you decide.")}
      <section className="stack">
        <h2>Possible brigades <span className="badge">{q.brigaded.length}</span></h2>
        <p className="faint">
          Posts with reports that were recorded but not counted toward the
          takedown threshold — a new empty account, or a reporter the author had
          blocked. A brigade is itself worth seeing.
        </p>
        {q.brigaded.length === 0 ? (
          <div className="empty">Nothing waiting.</div>
        ) : (
          q.brigaded.map((b) => (
            <article className="card card-tight" key={b.id}>
              <div className="row">
                <span className="badge">#{b.id}</span>
                <span className="badge">{b.status}</span>
                <span className="faint">{b.n} uncounted report{b.n === 1 ? "" : "s"}</span>
              </div>
              <p className="post-body">{b.body}</p>
            </article>
          ))
        )}
      </section>
    </div>
  );
}
