import { redirect } from "next/navigation";
import { Shell } from "@/components/Shell";
import { Composer } from "@/components/Composer";
import { FeedList } from "@/components/FeedList";
import { call } from "@/lib/api";
import { isError } from "@/lib/types";
import type { FeedResponse, MeResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function FeedPage() {
  const me = await call<MeResponse>("/api/auth/me");
  if (isError(me)) {
    // An expired or revoked session is the common case; a band transition
    // revokes every session, so turning 18 lands here.
    if (me.error.includes("sign in")) redirect("/login");
    return (
      <Shell>
        <div className="notice notice-danger">{me.error}</div>
      </Shell>
    );
  }

  const feed = await call<FeedResponse>("/api/feed?limit=20");

  return (
    <Shell band={me.band} username={me.user.username}>
      <div className="stack">
        <Composer band={me.band} />

        {isError(feed) ? (
          <div className="notice notice-danger">{feed.error}</div>
        ) : (
          <>
            <p className="faint">
              {feed.band === "minor"
                ? `Posts from people aged ${feed.ageWindow.min}–${feed.ageWindow.max}. Nobody over 17 appears here, and nobody over 17 can see you.`
                : "Posts from adults. No account under 18 appears here, or can see you."}
            </p>
            <FeedList initial={feed.posts} initialBefore={feed.nextBefore} />
          </>
        )}
      </div>
    </Shell>
  );
}
