import { initials } from "@/lib/format";
import type { PublicUser } from "@/lib/types";

/**
 * Initials, with the band's colour and an optional presence dot.
 *
 * The colour is not decoration: a minor account is teal and an adult account
 * is rose everywhere in the app, so which side of the 18 wall you are looking
 * at is visible at a glance rather than inferred.
 */
export function Avatar({
  user, large, online,
}: {
  user: PublicUser | null;
  large?: boolean;
  /** undefined means "unknown", which draws no dot at all. */
  online?: boolean;
}) {
  const name = user?.displayName || user?.username || "?";
  const minor = user?.band === "minor";
  return (
    <div
      className={[
        "avatar",
        minor ? "avatar-minor" : "",
        large ? "avatar-lg" : "",
      ].filter(Boolean).join(" ")}
      data-online={online === undefined ? undefined : String(online)}
      aria-hidden="true"
    >
      {initials(name)}
    </div>
  );
}
