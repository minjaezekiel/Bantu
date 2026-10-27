import Link from "next/link";
import { AuthForm } from "@/components/AuthForm";
import { ThemeToggle } from "@/components/ThemeToggle";

export default function SignupPage() {
  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">One<span>Taylor</span></span>
        <div className="nav"><ThemeToggle /></div>
      </header>
      <main className="main">
        <div className="container stack">
          <div>
            <h1>Join One Taylor</h1>
            <p className="muted">
              Share what your life is actually like, and meet people your own age.
            </p>
          </div>

          {/* Stated before the form, not buried in a tooltip after it. The age
              rule is the product, and a user who finds out afterwards that
              half the app is invisible to them has been misled. */}
          <div className="notice notice-info">
            <strong>How age works here.</strong> If you are under 18, you will only
            ever see — and be seen by — people within two years of your age. If you
            are 18 or over, you will only ever see other adults. The two sides never
            meet, and that is not a setting anyone can change.
          </div>

          <AuthForm mode="signup" />

          <p className="faint">
            Already have an account? <Link href="/login">Sign in</Link>.
          </p>
        </div>
      </main>
    </div>
  );
}
