"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { ThemeToggle } from "./ThemeToggle";
import { categoryLabel } from "@/lib/format";
import type { ExplicitCategory, MeResponse } from "@/lib/types";

const CATEGORIES: ExplicitCategory[] = [
  "suggestive", "nudity", "sexual", "fetish", "violence", "gore", "substances",
];

function useSaver() {
  const [state, setState] = useState<{ busy: boolean; msg: string | null; bad: boolean }>({
    busy: false, msg: null, bad: false,
  });
  async function save(path: string, body: unknown) {
    setState({ busy: true, msg: null, bad: false });
    const res = await fetch(`/api/proxy/${path}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = (await res.json()) as { ok?: boolean; error?: string };
    setState({
      busy: false,
      msg: data.ok ? "Saved." : (data.error ?? "That did not save."),
      bad: !data.ok,
    });
    return data;
  }
  return { ...state, save };
}

export function SettingsPanels({ me }: { me: MeResponse }) {
  const router = useRouter();
  const minor = me.band === "minor";

  const profile = useSaver();
  const tagsSaver = useSaver();
  const rangeSaver = useSaver();
  const explicitSaver = useSaver();

  const [displayName, setDisplayName] = useState(me.user.displayName ?? me.user.username);
  const [bio, setBio] = useState(me.user.bio ?? "");

  const [interests, setInterests] = useState(me.interests.join(", "));
  const [hobbies, setHobbies] = useState(me.hobbies.join(", "));
  const [dreamPartner, setDreamPartner] = useState(me.dreamPartner.join(", "));

  const [minAge, setMinAge] = useState(String(me.user.seekMinAge ?? ""));
  const [maxAge, setMaxAge] = useState(String(me.user.seekMaxAge ?? ""));

  const [optins, setOptins] = useState<ExplicitCategory[]>(
    (me.user.explicitOptins ?? "")
      .split(",")
      .map((s) => s.trim())
      .filter((s): s is ExplicitCategory => (CATEGORIES as string[]).includes(s)),
  );

  const split = (s: string) => s.split(",").map((t) => t.trim()).filter(Boolean);

  return (
    <div className="stack">
      {/* ── appearance ─────────────────────────────────────────────────── */}
      <section className="card stack">
        <h2>Appearance</h2>
        <p className="muted">
          Auto follows your device. Light and dark override it on this browser.
        </p>
        <ThemeToggle />
      </section>

      {/* ── profile ────────────────────────────────────────────────────── */}
      <section className="card stack">
        <h2>Profile</h2>
        <div>
          <label className="label" htmlFor="dn">Display name</label>
          <input id="dn" className="input" value={displayName}
                 onChange={(e) => setDisplayName(e.target.value)} />
        </div>
        <div>
          <label className="label" htmlFor="bio">Bio</label>
          <textarea id="bio" className="textarea" value={bio} maxLength={600}
                    onChange={(e) => setBio(e.target.value)} />
          <span className="hint">
            {/* A bio goes through the same safety checks as a post, and that
                is worth saying plainly rather than surprising someone with. */}
            Your bio is checked by the same safety rules as a post.{" "}
            {bio.length}/600
          </span>
        </div>
        {profile.msg && (
          <div className={profile.bad ? "notice notice-danger" : "notice notice-ok"}>
            {profile.msg}
          </div>
        )}
        <button className="btn" type="button" disabled={profile.busy}
                onClick={async () => {
                  await profile.save("settings/profile", { displayName, bio });
                  router.refresh();
                }}>
          {profile.busy ? "Saving…" : "Save profile"}
        </button>
      </section>

      {/* ── interests ──────────────────────────────────────────────────── */}
      <section className="card stack">
        <h2>Interests</h2>
        <p className="muted">
          This is what matching runs on. Overlap with another person is the
          largest part of their score, so an empty list has nothing to match.
        </p>
        <div>
          <label className="label" htmlFor="int">Interests</label>
          <input id="int" className="input" value={interests}
                 onChange={(e) => setInterests(e.target.value)} />
        </div>
        <div>
          <label className="label" htmlFor="hob">Hobbies</label>
          <input id="hob" className="input" value={hobbies}
                 onChange={(e) => setHobbies(e.target.value)} />
        </div>
        <div>
          <label className="label" htmlFor="dp">What you are looking for</label>
          <input id="dp" className="input" value={dreamPartner}
                 onChange={(e) => setDreamPartner(e.target.value)} />
          <span className="hint">
            Qualities or interests you hope to find. Matched against what other
            people actually list about themselves.
          </span>
        </div>
        {tagsSaver.msg && (
          <div className={tagsSaver.bad ? "notice notice-danger" : "notice notice-ok"}>
            {tagsSaver.msg}
          </div>
        )}
        <button className="btn" type="button" disabled={tagsSaver.busy}
                onClick={() => tagsSaver.save("settings/tags", {
                  interests: split(interests),
                  hobbies: split(hobbies),
                  dreamPartner: split(dreamPartner),
                })}>
          {tagsSaver.busy ? "Saving…" : "Save interests"}
        </button>
        <span className="hint">Up to 12 of each.</span>
      </section>

      {/* ── age range ──────────────────────────────────────────────────── */}
      <section className="card stack">
        <h2>Who you want to meet</h2>
        {minor ? (
          <div className="notice notice-info">
            While you are under 18 your range is fixed at{" "}
            <strong>{me.user.seekMinAge}–{me.user.seekMaxAge}</strong> — within two
            years of your age. It is not a setting, and nobody over 17 can appear
            here or see you.
          </div>
        ) : (
          <>
            <div className="grid-2">
              <div>
                <label className="label" htmlFor="minA">From</label>
                <input id="minA" className="input" type="number" inputMode="numeric"
                       value={minAge} onChange={(e) => setMinAge(e.target.value)} />
              </div>
              <div>
                <label className="label" htmlFor="maxA">To</label>
                <input id="maxA" className="input" type="number" inputMode="numeric"
                       value={maxAge} onChange={(e) => setMaxAge(e.target.value)} />
              </div>
            </div>
            <span className="hint">
              The lower end cannot go below 18. If you type something lower it is
              raised to 18 and saved that way.
            </span>
            {rangeSaver.msg && (
              <div className={rangeSaver.bad ? "notice notice-danger" : "notice notice-ok"}>
                {rangeSaver.msg}
              </div>
            )}
            <button className="btn" type="button" disabled={rangeSaver.busy}
                    onClick={async () => {
                      const data = await rangeSaver.save("settings/range", {
                        minAge: Number(minAge) || 18,
                        maxAge: Number(maxAge) || 99,
                      }) as { range?: { min: number; max: number } };
                      if (data.range) {
                        setMinAge(String(data.range.min));
                        setMaxAge(String(data.range.max));
                      }
                    }}>
              {rangeSaver.busy ? "Saving…" : "Save range"}
            </button>
          </>
        )}
      </section>

      {/* ── explicit categories: absent entirely for a minor ──────────── */}
      {!minor && (
        <section className="card stack">
          <h2>Content you are willing to see</h2>
          <p className="muted">
            Posts labelled with a category you have not chosen arrive blurred, with
            the category named, and you tap to reveal. Nothing here is shown by
            default.
          </p>
          <div className="chips">
            {CATEGORIES.map((c) => {
              const on = optins.includes(c);
              return (
                <button key={c} type="button" aria-pressed={on}
                        className={`chip chip-btn ${on ? "chip-on" : ""}`}
                        onClick={() => setOptins((cur) =>
                          on ? cur.filter((x) => x !== c) : [...cur, c])}>
                  {categoryLabel(c)}
                </button>
              );
            })}
          </div>
          {explicitSaver.msg && (
            <div className={explicitSaver.bad ? "notice notice-danger" : "notice notice-ok"}>
              {explicitSaver.msg}
            </div>
          )}
          <button className="btn" type="button" disabled={explicitSaver.busy}
                  onClick={() => explicitSaver.save("settings/explicit", { categories: optins })}>
            {explicitSaver.busy ? "Saving…" : "Save choices"}
          </button>
        </section>
      )}

      {/* ── account facts ──────────────────────────────────────────────── */}
      <section className="card stack">
        <h2>Your account</h2>
        <p className="muted">
          You are <strong>{me.user.age}</strong>, so you are on the{" "}
          <strong>{minor ? "under-18" : "18+"}</strong> side of One Taylor.
        </p>
        <p className="faint">
          Your email address is encrypted before it is stored and is never shown
          to anybody — including on this page. Your date of birth is used to work
          out your age and is never sent to another user.
        </p>
      </section>
    </div>
  );
}
