import { Shell } from "@/components/Shell";
import { ModerationQueue } from "@/components/ModerationQueue";

export const dynamic = "force-dynamic";

/**
 * The review queue. Reachable only when the server holds
 * ONE_TAYLOR_ADMIN_TOKEN — the proxy attaches it, and without it the backend
 * answers 404 rather than 403, so the route is invisible rather than merely
 * forbidden.
 */
export default function ModerationPage() {
  return (
    <Shell>
      <div className="stack">
        <div>
          <h1>Review queue</h1>
          <p className="muted">
            This exists because the classifier is not an oracle. Blocked posts,
            borderline ones, and posts whose reports looked like a brigade all
            land here for a person to look at.
          </p>
        </div>
        <ModerationQueue />
      </div>
    </Shell>
  );
}
