import { redirect } from "next/navigation";
import { Shell } from "@/components/Shell";
import { SettingsPanels } from "@/components/SettingsPanels";
import { call } from "@/lib/api";
import { isError } from "@/lib/types";
import type { MeResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function SettingsPage() {
  const me = await call<MeResponse>("/api/auth/me");
  if (isError(me)) redirect("/login");

  return (
    <Shell band={me.band} username={me.user.username}>
      <div className="stack">
        <h1>Settings</h1>
        <SettingsPanels me={me} />
      </div>
    </Shell>
  );
}
