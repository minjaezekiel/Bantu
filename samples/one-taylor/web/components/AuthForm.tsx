"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

const GENDERS = [
  { value: "woman", label: "Woman" },
  { value: "man", label: "Man" },
  { value: "non-binary", label: "Non-binary" },
  { value: "other", label: "Other" },
  { value: "prefer-not-to-say", label: "Prefer not to say" },
];

/** Completed years, computed the same way the backend does. */
function ageFrom(dob: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dob);
  if (!m) return null;
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const now = new Date();
  let age = now.getFullYear() - y;
  const beforeBirthday =
    now.getMonth() + 1 < mo || (now.getMonth() + 1 === mo && now.getDate() < d);
  if (beforeBirthday) age -= 1;
  return age >= 0 && age < 130 ? age : null;
}

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [dob, setDob] = useState("");
  const [gender, setGender] = useState("prefer-not-to-say");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const age = useMemo(() => (dob ? ageFrom(dob) : null), [dob]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    const body =
      mode === "signup"
        ? { mode, username, password, email, dob, gender }
        : { mode, username, password };
    const res = await fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await res.json()) as { ok?: boolean; error?: string };
    setBusy(false);
    if (!data.ok) {
      setError(data.error ?? "That did not work.");
      return;
    }
    router.replace("/feed");
    router.refresh();
  }

  return (
    <form className="card stack" onSubmit={submit} noValidate>
      <div>
        <label className="label" htmlFor="username">Username</label>
        <input
          id="username"
          className="input"
          autoComplete="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        {mode === "signup" && (
          <span className="hint">3–20 characters: letters, numbers, dot, underscore.</span>
        )}
      </div>

      {mode === "signup" && (
        <div>
          <label className="label" htmlFor="email">Email</label>
          <input
            id="email"
            className="input"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <span className="hint">
            Kept private. It is encrypted before it is stored and never shown to
            anybody, including you.
          </span>
        </div>
      )}

      <div>
        <label className="label" htmlFor="password">Password</label>
        <input
          id="password"
          className="input"
          type="password"
          autoComplete={mode === "signup" ? "new-password" : "current-password"}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {mode === "signup" && (
          <span className="hint">
            At least 10 characters. A short phrase you will remember beats a short
            password with punctuation in it.
          </span>
        )}
      </div>

      {mode === "signup" && (
        <>
          <div>
            <label className="label" htmlFor="dob">Date of birth</label>
            <input
              id="dob"
              className="input"
              type="date"
              value={dob}
              onChange={(e) => setDob(e.target.value)}
              required
            />
            <span className="hint">
              {age === null
                ? "We ask for the date, not the age, so it stays right as you get older."
                : age < 13
                  ? "You need to be at least 13 to use One Taylor."
                  : age < 18
                    ? `You are ${age}. You will see people aged ${Math.max(13, age - 2)}–${Math.min(17, age + 2)}, and nobody over 17 will see you.`
                    : `You are ${age}. You will see adults aged 18 and over, and no one under 18 will see you.`}
            </span>
          </div>

          <div>
            <span className="label">Gender</span>
            <div className="chips" role="radiogroup" aria-label="Gender">
              {GENDERS.map((g) => (
                <button
                  key={g.value}
                  type="button"
                  role="radio"
                  aria-checked={gender === g.value}
                  className={`chip chip-btn ${gender === g.value ? "chip-on" : ""}`}
                  onClick={() => setGender(g.value)}
                >
                  {g.label}
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      {error && <div className="notice notice-danger">{error}</div>}

      <button className="btn" type="submit" disabled={busy}>
        {busy ? "One moment…" : mode === "signup" ? "Create account" : "Sign in"}
      </button>
    </form>
  );
}
