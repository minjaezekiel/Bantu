import { cookies } from "next/headers";
import type { ApiError } from "./types";

/**
 * Server-side calls into the Bantu backend.
 *
 * Only ever called from a server component or a route handler. The session
 * token lives in an httpOnly cookie, so client JavaScript cannot read it and
 * cannot be made to leak it; the token is attached here, on the server.
 */

export const API_BASE =
  process.env.ONE_TAYLOR_API ?? "http://127.0.0.1:4100";

export const SESSION_COOKIE = "ot_session";

export async function sessionToken(): Promise<string | null> {
  const jar = await cookies();
  return jar.get(SESSION_COOKIE)?.value ?? null;
}

interface CallOptions {
  method?: string;
  body?: unknown;
  /** Attach the session (default true). */
  auth?: boolean;
  /** Forwarded to fetch; the feed must never be cached across users. */
  cache?: RequestCache;
}

export async function call<T>(
  path: string,
  opts: CallOptions = {},
): Promise<T | ApiError> {
  const { method = "GET", body, auth = true, cache = "no-store" } = opts;
  const headers: Record<string, string> = { "content-type": "application/json" };

  if (auth) {
    const token = await sessionToken();
    if (token) headers.authorization = `Bearer ${token}`;
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      cache,
    });
  } catch {
    // A backend that is not running is the most common failure in
    // development, and "fetch failed" in a stack trace does not say so.
    return {
      ok: false,
      error: `The One Taylor backend is not reachable at ${API_BASE}. Start it with: cd samples/one-taylor/backend && ONE_TAYLOR_SECRET=$(openssl rand -hex 32) bantu run server.b`,
    };
  }

  const text = await res.text();
  if (!text) return { ok: false, error: `${res.status} with an empty body` };

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, error: `${res.status}: ${text.slice(0, 200)}` };
  }
  return parsed as T | ApiError;
}

/** True when there is a session cookie at all — not that it is still valid. */
export async function hasSession(): Promise<boolean> {
  return (await sessionToken()) !== null;
}
