import Link from "next/link";
import { AuthForm } from "@/components/AuthForm";
import { ThemeToggle } from "@/components/ThemeToggle";

export default function LoginPage() {
  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">One<span>Taylor</span></span>
        <div className="nav"><ThemeToggle /></div>
      </header>
      <main className="main">
        <div className="container stack">
          <div>
            <h1>Welcome back</h1>
            <p className="muted">Sign in with your username and password.</p>
          </div>
          <AuthForm mode="login" />
          <p className="faint">
            New here? <Link href="/signup">Create an account</Link>.
          </p>
        </div>
      </main>
    </div>
  );
}
