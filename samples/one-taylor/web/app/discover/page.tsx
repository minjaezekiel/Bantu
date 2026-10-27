import Link from "next/link";
import { redirect } from "next/navigation";
import { Shell } from "@/components/Shell";
import { Avatar } from "@/components/Avatar";
import { FollowButton } from "@/components/FollowButton";
import { call } from "@/lib/api";
import { isError } from "@/lib/types";
import type { DiscoverResponse, MeResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function DiscoverPage() {
  const me = await call<MeResponse>("/api/auth/me");
  if (isError(me)) redirect("/login");

  const d = await call<DiscoverResponse>("/api/discover?limit=24");

  return (
    <Shell band={me.band} username={me.user.username} wide>
      <div className="stack">
        <div>
          <h1>Discover</h1>
          {isError(d) ? null : (
            <p className="muted">
              {d.rangeLocked
                ? `People aged ${d.range.min}–${d.range.max}. Your range is fixed while you are under 18.`
                : `People aged ${d.range.min}–${d.range.max}, from your settings.`}
            </p>
          )}
        </div>

        {isError(d) ? (
          <div className="notice notice-danger">{d.error}</div>
        ) : d.matches.length === 0 ? (
          <div className="empty">
            <p><strong>No matches yet.</strong></p>
            <p className="faint">
              Add some interests in Settings — matching works on what you have in
              common, so an empty profile has nothing to match on.
            </p>
            <p><Link href="/settings">Go to settings</Link></p>
          </div>
        ) : (
          <div className="matches">
            {d.matches.map((m) => (
              <article className="card stack-tight" key={m.user.id}>
                <div className="post-head">
                  <Avatar user={m.user} />
                  <div className="stack-tight" style={{ minWidth: 0 }}>
                    <Link href={`/u/${m.user.username}`} className="post-author">
                      {m.user.displayName || m.user.username}
                    </Link>
                    <span className="faint">
                      {m.user.age} · {m.user.gender}
                    </span>
                  </div>
                </div>

                {m.user.bio && <p className="muted">{m.user.bio}</p>}

                {/* Why this person, in their own words rather than a score.
                    An unexplained recommendation in a dating app is
                    indistinguishable from a random one. */}
                <p className="match-why">{m.why}</p>

                {m.shared.length > 0 && (
                  <div className="chips">
                    {m.shared.slice(0, 6).map((s) => (
                      <span className="chip chip-on" key={s}>{s}</span>
                    ))}
                  </div>
                )}

                <div className="row">
                  <FollowButton username={m.user.username} following={false} />
                  <Link className="btn btn-quiet" href={`/u/${m.user.username}`}>
                    View profile
                  </Link>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </Shell>
  );
}
