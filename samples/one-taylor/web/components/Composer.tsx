"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { categoryLabel } from "@/lib/format";
import type { Band, ExplicitCategory } from "@/lib/types";

const CATEGORIES: ExplicitCategory[] = [
  "suggestive", "nudity", "sexual", "fetish", "violence", "gore", "substances",
];

const MAX_BYTES = 4 * 1024 * 1024;
const MAX_EDGE = 1600;

/**
 * Resize in the browser before uploading.
 *
 * Not a nicety: sua has no multipart parser, so the image travels as base64
 * inside the JSON body and costs about 1.34x its own size on the wire. A
 * straight-from-camera 12 MB JPEG would exceed the server's body cap, and the
 * error would arrive as a connection reset rather than a message.
 */
async function prepare(file: File): Promise<string> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, MAX_EDGE / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("This browser cannot resize images.");
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();

  // Step the quality down until it fits, rather than failing at 0.9.
  for (const q of [0.85, 0.7, 0.55, 0.4]) {
    const dataUrl = canvas.toDataURL("image/jpeg", q);
    const bytes = Math.floor((dataUrl.length - dataUrl.indexOf(",") - 1) * 0.75);
    if (bytes <= MAX_BYTES) return dataUrl;
  }
  throw new Error("That image is too large even after resizing.");
}

export function Composer({ band }: { band: Band }) {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);
  const [body, setBody] = useState("");
  const [tags, setTags] = useState("");
  const [media, setMedia] = useState<string[]>([]);
  const [labels, setLabels] = useState<ExplicitCategory[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [blocked, setBlocked] = useState<string[] | null>(null);
  const [held, setHeld] = useState(false);
  const [mediaErrors, setMediaErrors] = useState<string[] | null>(null);
  const [ok, setOk] = useState<string | null>(null);

  async function addFiles(files: FileList | null) {
    if (!files) return;
    setError(null);
    const next: string[] = [];
    for (const f of Array.from(files).slice(0, 4 - media.length)) {
      try {
        next.push(await prepare(f));
      } catch (e) {
        setError(e instanceof Error ? e.message : "That image could not be read.");
      }
    }
    setMedia((m) => [...m, ...next].slice(0, 4));
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBlocked(null);
    setOk(null);
    setMediaErrors(null);
    setBusy(true);
    const res = await fetch("/api/proxy/posts", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        body,
        tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
        media: media.map((data) => ({ data })),
        labels,
      }),
    });
    const data = (await res.json()) as {
      ok?: boolean;
      error?: string;
      status?: string;
      verdict?: "allow" | "label" | "watch" | "hold" | "block";
      message?: string | null;
      reasons?: string[];
      queuedForReview?: boolean;
      mediaErrors?: string[];
    };
    setBusy(false);

    if (data.verdict === "block" || data.status === "blocked") {
      setBlocked(data.reasons ?? []);
      return;
    }
    if (!data.ok) {
      setError(data.error ?? "That did not post.");
      return;
    }
    setBody("");
    setTags("");
    setMedia([]);
    setLabels([]);
    if (fileRef.current) fileRef.current.value = "";
    // "hold" and "watch" are different outcomes and the author should be told
    // which: held means nobody else can see it yet, watched means it is live.
    setHeld(data.verdict === "hold");
    setOk(data.message ?? null);
    // Surfaced rather than swallowed. The server already reported "could not
    // store an image" and this component dropped it on the floor, so a post
    // arrived without its picture and said nothing — which is exactly how the
    // missing-upload-directory bug stayed invisible.
    setMediaErrors(data.mediaErrors && data.mediaErrors.length > 0
      ? data.mediaErrors : null);
    router.refresh();
  }

  return (
    <form className="card stack" onSubmit={submit}>
      <div>
        <label className="label" htmlFor="body">Share something</label>
        <textarea
          id="body"
          className="textarea"
          placeholder="What has your day been like?"
          value={body}
          maxLength={5000}
          onChange={(e) => setBody(e.target.value)}
        />
      </div>

      <div>
        <label className="label" htmlFor="tags">Tags</label>
        <input
          id="tags"
          className="input"
          placeholder="chess, baking, football"
          value={tags}
          onChange={(e) => setTags(e.target.value)}
        />
        <span className="hint">Comma separated. Tags are how the feed finds people like you.</span>
      </div>

      <div>
        <span className="label">Pictures</span>
        <input
          ref={fileRef}
          type="file"
          accept="image/jpeg,image/png,image/gif,image/webp"
          multiple
          onChange={(e) => addFiles(e.target.files)}
        />
        {media.length > 0 && (
          <div className="row" style={{ marginTop: "var(--s-2)" }}>
            {media.map((m, i) => (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                key={i}
                src={m}
                alt=""
                style={{
                  width: 64, height: 64, objectFit: "cover",
                  borderRadius: "var(--radius-s)", border: "1px solid var(--border)",
                }}
              />
            ))}
            <button
              type="button"
              className="btn btn-quiet"
              onClick={() => { setMedia([]); if (fileRef.current) fileRef.current.value = ""; }}
            >
              Clear
            </button>
          </div>
        )}
        <span className="hint">Up to four, resized in your browser before upload.</span>
      </div>

      {/* Absent for a minor, not disabled. A disabled control is an
          advertisement for something they cannot have, and there is no
          audience for a labelled post on their side of the wall anyway. */}
      {band === "adult" && (
        <div>
          <span className="label">Content warnings</span>
          <div className="chips">
            {CATEGORIES.map((c) => {
              const on = labels.includes(c);
              return (
                <button
                  key={c}
                  type="button"
                  aria-pressed={on}
                  className={`chip chip-btn ${on ? "chip-on" : ""}`}
                  onClick={() =>
                    setLabels((cur) => (on ? cur.filter((x) => x !== c) : [...cur, c]))
                  }
                >
                  {categoryLabel(c)}
                </button>
              );
            })}
          </div>
          <span className="hint">
            Adults who have not opted into a category see your post blurred until they tap it.
          </span>
        </div>
      )}

      {error && <div className="notice notice-danger">{error}</div>}
      {ok && <div className={held ? "notice notice-warn" : "notice notice-ok"}>{ok}</div>}
      {mediaErrors && (
        <div className="notice notice-warn">
          <strong>The text posted, but a picture did not.</strong>
          <ul style={{ margin: "var(--s-2) 0 0", paddingLeft: "var(--s-5)" }}>
            {mediaErrors.map((m, i) => <li key={i}>{m}</li>)}
          </ul>
        </div>
      )}
      {blocked && (
        <div className="notice notice-danger">
          <strong>This was not posted.</strong>{" "}
          {blocked.length > 0
            ? `It matched: ${blocked.join(", ")}.`
            : "It matched the safety rules."}{" "}
          The text has been kept so a moderator can review the decision — this
          is not final.
        </div>
      )}

      <div className="row">
        <button className="btn" type="submit" disabled={busy || (!body.trim() && media.length === 0)}>
          {busy ? "Posting…" : "Post"}
        </button>
        <span className="faint">{body.length}/5000</span>
      </div>
    </form>
  );
}
