"use client";

import { useState } from "react";

export function FollowButton({
  username,
  following,
}: {
  username: string;
  following: boolean;
}) {
  const [on, setOn] = useState(following);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function toggle() {
    setBusy(true);
    setError(null);
    const res = await fetch(`/api/proxy/users/${username}/${on ? "unfollow" : "follow"}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    const data = (await res.json()) as { ok?: boolean; following?: boolean; error?: string };
    setBusy(false);
    if (!data.ok) {
      setError(data.error ?? "That did not work.");
      return;
    }
    setOn(Boolean(data.following));
  }

  return (
    <>
      <button
        type="button"
        className={on ? "btn btn-ghost btn-sm" : "btn btn-sm"}
        onClick={toggle}
        disabled={busy}
      >
        {busy ? "…" : on ? "Following" : "Follow"}
      </button>
      {error && <span className="faint" style={{ color: "var(--danger)" }}>{error}</span>}
    </>
  );
}
