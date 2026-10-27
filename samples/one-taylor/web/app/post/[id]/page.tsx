import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { Shell } from "@/components/Shell";
import { PostCard } from "@/components/PostCard";
import { call } from "@/lib/api";
import { isError } from "@/lib/types";
import type { MeResponse, Post } from "@/lib/types";

export const dynamic = "force-dynamic";

interface PostResponse { ok: true; post: Post }

export default async function PostPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const me = await call<MeResponse>("/api/auth/me");
  if (isError(me)) redirect("/login");

  const r = await call<PostResponse>(`/api/posts/${encodeURIComponent(id)}`);
  if (isError(r)) notFound();

  return (
    <Shell band={me.band} username={me.user.username}>
      <div className="stack">
        <Link className="btn btn-quiet" href="/feed">← Back to the feed</Link>
        <PostCard post={r.post} />
        {r.post.mine && r.post.status && r.post.status !== "live" && (
          <div className="notice notice-warn">
            This post is <strong>{r.post.status}</strong>, so nobody else can see it.
            A moderator can review the decision.
          </div>
        )}
      </div>
    </Shell>
  );
}
