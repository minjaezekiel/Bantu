"use client";

import Link from "next/link";
import { useState } from "react";
import { Avatar } from "./Avatar";
import { categoryLabel, timeAgo } from "@/lib/format";
import type { Post } from "@/lib/types";

const REASONS = [
  { value: "explicit", label: "Explicit content" },
  { value: "harassment", label: "Harassment or bullying" },
  { value: "violence", label: "Violence or threats" },
  { value: "self_harm", label: "Self-harm" },
  { value: "underage", label: "Involves someone under 18" },
  { value: "spam", label: "Spam" },
  { value: "impersonation", label: "Impersonation" },
  { value: "other", label: "Something else" },
];

export function PostCard({ post, onGone }: { post: Post; onGone?: (id: number) => void }) {
  const [likes, setLikes] = useState(post.likeCount);
  const [liked, setLiked] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [reporting, setReporting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const author = post.author;

  async function like() {
    const next = !liked;
    setLiked(next);
    const res = await fetch(`/api/proxy/posts/${post.id}/${next ? "like" : "unlike"}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    const data = (await res.json()) as { ok?: boolean; likeCount?: number; error?: string };
    if (data.ok && typeof data.likeCount === "number") setLikes(data.likeCount);
    else setLiked(!next);
  }

  async function report(reason: string) {
    setReporting(false);
    const res = await fetch(`/api/proxy/posts/${post.id}/report`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ reason }),
    });
    const data = (await res.json()) as { ok?: boolean; removed?: boolean; error?: string };
    if (!data.ok) {
      setNotice(data.error ?? "That did not work.");
      return;
    }
    // The post disappears for the reporter straight away — the backend has
    // already recorded a per-viewer hide, so a refresh would drop it anyway.
    setNotice(
      data.removed
        ? "Reported. This post has been taken down."
        : "Reported, and hidden from your feed.",
    );
    if (onGone) setTimeout(() => onGone(post.id), 1200);
  }

  const warned = post.display === "blur" && !revealed;

  return (
    <article className="card post">
      <div className="post-head">
        <Avatar user={author} />
        <div className="stack-tight" style={{ minWidth: 0 }}>
          <div className="row" style={{ gap: "var(--s-2)" }}>
            {author ? (
              <Link href={`/u/${author.username}`} className="post-author">
                {author.displayName || author.username}
              </Link>
            ) : (
              <span className="post-author">someone</span>
            )}
            {author && <span className="faint">{author.age}</span>}
          </div>
          <span className="faint">{timeAgo(post.createdAt)}</span>
        </div>
        <div className="spacer" />
        {post.status && post.status !== "live" && (
          <span className={post.status === "blocked" ? "badge badge-blocked" : "badge badge-held"}>
            {post.status === "review" ? "held for review"
              : post.status === "blocked" ? "not published" : post.status}
          </span>
        )}
      </div>

      {post.body && <p className="post-body">{post.body}</p>}

      {/* Only the author ever sees this card, so the explanation can be
          direct about what happened and what comes next. */}
      {post.mine && post.status === "review" && (
        <div className="notice notice-warn">
          <strong>Held for review.</strong> Only you can see this. The safety
          checks flagged it and a moderator will decide — nothing has been
          deleted.
        </div>
      )}
      {post.mine && post.status === "blocked" && (
        <div className="notice notice-danger">
          <strong>Not published.</strong> This matched the safety rules, so it
          was never shown to anybody. The text is kept so the decision can be
          reviewed.
        </div>
      )}

      {post.media.length > 0 && (
        <div className={`post-media ${post.media.length > 1 ? "two" : ""}`}>
          {post.media.map((m) =>
            warned ? (
              <div className="media-warned" key={m.id}>
                <div className="media-placeholder" />
                <div className="media-warned-inner">
                  <strong>Content warning</strong>
                  <span className="faint">
                    {post.blurredFor.map(categoryLabel).join(", ")}
                  </span>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => setRevealed(true)}
                  >
                    Tap to reveal
                  </button>
                </div>
              </div>
            ) : (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                key={m.id}
                src={`/api/proxy/media/${m.id}`}
                alt=""
                loading="lazy"
                decoding="async"
              />
            ),
          )}
        </div>
      )}

      {post.tags.length > 0 && (
        <div className="chips">
          {post.tags.map((t) => (
            <span className="chip" key={t}>#{t}</span>
          ))}
        </div>
      )}

      {post.labels.length > 0 && !warned && (
        <div className="chips">
          {post.labels.map((l) => (
            <span className="chip" key={l}>{categoryLabel(l)}</span>
          ))}
        </div>
      )}

      {notice && <div className="notice notice-info">{notice}</div>}

      <div className="post-actions">
        <button type="button" className="btn btn-quiet" onClick={like} aria-pressed={liked}>
          {liked ? "♥" : "♡"} {likes}
        </button>
        <Link className="btn btn-quiet" href={`/post/${post.id}`}>Open</Link>
        <div className="spacer" />
        {!post.mine && (
          <button
            type="button"
            className="btn btn-quiet"
            onClick={() => setReporting((r) => !r)}
            aria-expanded={reporting}
          >
            Report
          </button>
        )}
      </div>

      {reporting && (
        <div className="stack-tight">
          <span className="label">Why are you reporting this?</span>
          <div className="chips">
            {REASONS.map((r) => (
              <button
                key={r.value}
                type="button"
                className="chip chip-btn"
                onClick={() => report(r.value)}
              >
                {r.label}
              </button>
            ))}
          </div>
          <span className="hint">
            It is hidden from you immediately. Two separate reports take a post down.
          </span>
        </div>
      )}
    </article>
  );
}
