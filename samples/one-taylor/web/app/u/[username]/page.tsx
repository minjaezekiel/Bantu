import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { Shell } from "@/components/Shell";
import { Avatar } from "@/components/Avatar";
import { FollowButton } from "@/components/FollowButton";
import { PostCard } from "@/components/PostCard";
import { call } from "@/lib/api";
import { isError } from "@/lib/types";
import type { MeResponse, ProfileResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function ProfilePage({
  params,
}: {
  params: Promise<{ username: string }>;
}) {
  const { username } = await params;
  const me = await call<MeResponse>("/api/auth/me");
  if (isError(me)) redirect("/login");

  const p = await call<ProfileResponse>(`/api/users/${encodeURIComponent(username)}`);
  if (isError(p)) {
    // The backend answers 404 across the age wall on purpose — "that account
    // exists but you may not see it" is exactly the fact the wall hides. So
    // the frontend shows a plain not-found and does not speculate.
    notFound();
  }

  const groups: { title: string; items: string[] }[] = [
    { title: "Interests", items: p.interests },
    { title: "Hobbies", items: p.hobbies },
    { title: "Looking for", items: p.dreamPartner },
  ];

  return (
    <Shell band={me.band} username={me.user.username}>
      <div className="stack">
        <section className="card stack">
          <div className="post-head">
            <Avatar user={p.user} large />
            <div className="stack-tight" style={{ minWidth: 0 }}>
              <h1 style={{ margin: 0 }}>{p.user.displayName || p.user.username}</h1>
              <span className="faint">
                @{p.user.username} · {p.user.age} · {p.user.gender}
              </span>
            </div>
          </div>

          {p.user.bio && <p>{p.user.bio}</p>}

          <div className="row">
            <span className="badge">{p.followers} followers</span>
            <span className="badge">{p.following} following</span>
            {p.canMatch && !p.isYou && <span className="badge badge-adult">In your range</span>}
            <div className="spacer" />
            {!p.isYou && <FollowButton username={p.user.username} following={p.youFollow} />}
            {/* Messaging needs BOTH directions, so the button appears only
                once they follow back. Showing it earlier and failing would
                teach people that the button is unreliable. */}
            {!p.isYou && p.youFollow && p.followsYou && (
              <Link className="btn btn-ghost btn-sm" href={`/messages/${p.user.username}`}>
                Message
              </Link>
            )}
          </div>

          {groups.map((g) =>
            g.items.length > 0 ? (
              <div key={g.title}>
                <span className="label">{g.title}</span>
                <div className="chips">
                  {g.items.map((t) => (
                    <span className="chip" key={t}>{t}</span>
                  ))}
                </div>
              </div>
            ) : null,
          )}
        </section>

        <h2>Posts</h2>
        {p.posts.length === 0 ? (
          <div className="empty">Nothing posted yet.</div>
        ) : (
          p.posts.map((post) => (
            <PostCard key={post.id} post={{ ...post, mine: p.isYou }} />
          ))
        )}
      </div>
    </Shell>
  );
}
