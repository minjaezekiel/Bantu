import { redirect } from "next/navigation";
import { Shell } from "@/components/Shell";
import { Messenger } from "@/components/Messenger";
import { call } from "@/lib/api";
import { isError } from "@/lib/types";
import type { ConversationsResponse, MeResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

/**
 * The same messenger, opened on one person. Reached from a profile's "Message"
 * button, so arriving here also creates the conversation if it does not exist
 * — which the backend only allows when both people follow each other.
 */
export default async function ThreadPage({
  params,
}: {
  params: Promise<{ username: string }>;
}) {
  const { username } = await params;
  const me = await call<MeResponse>("/api/auth/me");
  if (isError(me)) redirect("/login");

  const opened = await call<{ ok: true; conversationId: number }>(
    `/api/conversations/${encodeURIComponent(username)}`, { method: "POST", body: {} });

  const threads = await call<ConversationsResponse>("/api/conversations?limit=40");

  return (
    <Shell band={me.band} username={me.user.username} wide>
      <div className="stack">
        {isError(opened) && (
          <div className="notice notice-warn">{opened.error}</div>
        )}
        <Messenger
          me={me.user}
          initialThreads={isError(threads) ? [] : threads.conversations}
          openWith={isError(opened) ? null : username}
        />
      </div>
    </Shell>
  );
}
