import { redirect } from "next/navigation";
import { Shell } from "@/components/Shell";
import { SearchPanel } from "@/components/SearchPanel";
import { call } from "@/lib/api";
import { isError } from "@/lib/types";
import type { MeResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function SearchPage() {
  const me = await call<MeResponse>("/api/auth/me");
  if (isError(me)) redirect("/login");

  return (
    <Shell band={me.band} username={me.user.username}>
      <div className="stack">
        <div>
          <h1>Search</h1>
          <p className="muted">
            {me.band === "minor"
              ? "You can find accounts within two years of your age. Nothing else is searchable."
              : "You can find adult accounts. No account under 18 is searchable from here."}
          </p>
        </div>
        <SearchPanel band={me.band} />
      </div>
    </Shell>
  );
}
