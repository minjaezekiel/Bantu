import Link from "next/link";

export default function NotFound() {
  return (
    <div className="shell">
      <main className="main">
        <div className="container stack">
          <h1>Not found</h1>
          <p className="muted">
            This page does not exist, or it belongs to an account outside your age
            band. Those are the same answer on purpose — telling you which would
            leak the thing One Taylor exists to keep separate.
          </p>
          <p><Link href="/feed">Back to your feed</Link></p>
        </div>
      </main>
    </div>
  );
}
