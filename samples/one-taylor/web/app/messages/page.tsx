import { redirect } from "next/navigation";
import { Shell } from "@/components/Shell";
import { Messenger } from "@/components/Messenger";
import { call } from "@/lib/api";
import { isError } from "@/lib/types";
import type { ConversationsResponse, MeResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function MessagesPage() {
  const me = await call<MeResponse>("/api/auth/me");
  if (isError(me)) redirect("/login");

  const threads = await call<ConversationsResponse>("/api/conversations?limit=40");

  return (
    <Shell band={me.band} username={me.user.username} wide>
      <div className="stack">
        <div>
          <h1>Messages</h1>
          <p className="muted">
            You can message, call and video call anyone you both follow. Voice
            and video are peer to peer — the server passes the introduction and
            never sees the call.
          </p>
        </div>
        <Messenger
          me={me.user}
          initialThreads={isError(threads) ? [] : threads.conversations}
          openWith={null}
        />
      </div>
    </Shell>
  );
}
