"use client";

import Link from "next/link";
import { useState } from "react";
import { Avatar } from "./Avatar";
import type { Band, SearchResponse } from "@/lib/types";

export function SearchPanel({ band }: { band: Band }) {
  const [q, setQ] = useState("");
  const [minAge, setMinAge] = useState("");
  const [maxAge, setMaxAge] = useState("");
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const params = new URLSearchParams({ q, limit: "30" });
    if (minAge) params.set("minAge", minAge);
    if (maxAge) params.set("maxAge", maxAge);
    const r = await fetch(`/api/proxy/search/users?${params.toString()}`);
    const data = (await r.json()) as SearchResponse | { ok: false; error: string };
    setBusy(false);
    if (!("results" in data)) {
      setError(data.error);
      return;
    }
    setRes(data);
  }

  return (
    <div className="stack">
      <form className="card stack" onSubmit={run}>
        <div>
          <label className="label" htmlFor="q">Username</label>
          <input
            id="q"
            className="input"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="part of a username"
          />
        </div>

        <div className="grid-2">
          <div>
            <label className="label" htmlFor="minAge">Age from</label>
            <input
              id="minAge"
              className="input"
              type="number"
              inputMode="numeric"
              value={minAge}
              onChange={(e) => setMinAge(e.target.value)}
            />
          </div>
          <div>
            <label className="label" htmlFor="maxAge">Age to</label>
            <input
              id="maxAge"
              className="input"
              type="number"
              inputMode="numeric"
              value={maxAge}
              onChange={(e) => setMaxAge(e.target.value)}
            />
          </div>
        </div>
        <span className="hint">
          {/* The filter narrows WITHIN the band. It cannot widen past it: the
              band comes from the session, never from these fields. */}
          This narrows your results. It cannot reach outside your age band —
          {band === "minor" ? " 13 to 17" : " 18 and over"} — whatever you type.
        </span>

        <button className="btn" type="submit" disabled={busy}>
          {busy ? "Searching…" : "Search"}
        </button>
      </form>

      {error && <div className="notice notice-danger">{error}</div>}

      {res && (
        <div className="stack">
          <p className="faint">
            {res.results.length} result{res.results.length === 1 ? "" : "s"}, ages{" "}
            {res.ageWindow.min}–{res.ageWindow.max}.
          </p>
          {res.results.length === 0 ? (
            <div className="empty">Nobody matched that.</div>
          ) : (
            res.results.map((u) => (
              <article className="card card-tight" key={u.id}>
                <div className="post-head">
                  <Avatar user={u} />
                  <div className="stack-tight" style={{ minWidth: 0 }}>
                    <Link href={`/u/${u.username}`} className="post-author">
                      {u.displayName || u.username}
                    </Link>
                    <span className="faint">{u.age} · {u.gender}</span>
                  </div>
                </div>
                {u.bio && <p className="muted" style={{ margin: "var(--s-2) 0 0" }}>{u.bio}</p>}
              </article>
            ))
          )}
        </div>
      )}
    </div>
  );
}
