/** Formatting shared by server and client components. */

export function initials(name: string): string {
  const clean = name.trim();
  if (!clean) return "?";
  return clean.slice(0, 2).toUpperCase();
}

/**
 * SQLite hands back "YYYY-MM-DD HH:MM:SS" in UTC. Date.parse needs the "T"
 * and the "Z", and without them Safari returns NaN while Chrome guesses
 * local time — so a relative timestamp silently differs by hours between
 * browsers.
 */
export function parseSqlDate(s: string): Date | null {
  if (!s) return null;
  const iso = s.includes("T") ? s : `${s.replace(" ", "T")}Z`;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function timeAgo(s: string, now: number = Date.now()): string {
  const d = parseSqlDate(s);
  if (!d) return "";
  const secs = Math.max(0, Math.round((now - d.getTime()) / 1000));
  if (secs < 45) return "just now";
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  const weeks = Math.round(days / 7);
  if (weeks < 5) return `${weeks}w ago`;
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

export const CATEGORY_LABELS: Record<string, string> = {
  suggestive: "Suggestive",
  nudity: "Nudity",
  sexual: "Sexual content",
  fetish: "Fetish",
  violence: "Violence",
  gore: "Graphic injury",
  substances: "Drugs or alcohol",
};

export function categoryLabel(c: string): string {
  return CATEGORY_LABELS[c] ?? c;
}
