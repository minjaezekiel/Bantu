import Link from "next/link";
import { ThemeToggle } from "./ThemeToggle";
import { SignOutButton } from "./SignOutButton";
import type { Band } from "@/lib/types";

/**
 * The band badge is in the top bar on every page, deliberately. Which side of
 * the 18 wall you are on determines everything you can see, so it should never
 * be something a user has to infer from an empty feed.
 */
export function Shell({
  band,
  username,
  children,
  wide,
}: {
  band?: Band;
  username?: string;
  children: React.ReactNode;
  wide?: boolean;
}) {
  return (
    <div className="shell">
      <header className="topbar">
        <Link href="/feed" className="brand">
          One<span>Taylor</span>
        </Link>
        {band && (
          <span className={band === "minor" ? "badge badge-minor" : "badge badge-adult"}>
            {band === "minor" ? "Under 18" : "18+"}
          </span>
        )}
        <nav className="nav">
          <Link className="nav-link" href="/feed">Feed</Link>
          <Link className="nav-link" href="/discover">Discover</Link>
          <Link className="nav-link" href="/messages">Messages</Link>
          <Link className="nav-link" href="/search">Search</Link>
          {username && (
            <Link className="nav-link" href={`/u/${username}`}>You</Link>
          )}
          <Link className="nav-link" href="/settings">Settings</Link>
          <ThemeToggle />
          {username && <SignOutButton />}
        </nav>
      </header>
      <main className="main">
        <div className={wide ? "container-wide" : "container"}>{children}</div>
      </main>
    </div>
  );
}
