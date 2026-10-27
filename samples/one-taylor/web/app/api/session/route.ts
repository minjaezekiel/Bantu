import { NextRequest } from "next/server";
import { API_BASE, SESSION_COOKIE } from "@/lib/api";

/**
 * Sign in, sign up, sign out — the only three places a session cookie is
 * written. The token the backend issues is put straight into an httpOnly
 * cookie and is never included in the response body, so it does not reach
 * client JavaScript at any point.
 */

interface Credentials {
  mode: "login" | "signup";
  [k: string]: unknown;
}

export async function POST(req: NextRequest) {
  let payload: Credentials;
  try {
    payload = (await req.json()) as Credentials;
  } catch {
    return Response.json({ ok: false, error: "bad request" }, { status: 400 });
  }

  const { mode, ...fields } = payload;
  const path = mode === "signup" ? "/api/auth/signup" : "/api/auth/login";

  let upstream: Response;
  try {
    upstream = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(fields),
      cache: "no-store",
    });
  } catch {
    return Response.json(
      {
        ok: false,
        error: `The backend is not reachable at ${API_BASE}. Start it with: cd samples/one-taylor/backend && ONE_TAYLOR_SECRET=$(openssl rand -hex 32) bantu run server.b`,
      },
      { status: 502 },
    );
  }

  const data = (await upstream.json()) as Record<string, unknown>;
  if (!upstream.ok || data.ok !== true) {
    return Response.json(data, { status: upstream.status });
  }

  const token = String(data.token ?? "");
  const expires = Number(data.expires ?? 0);

  // The token is removed from the body on purpose. If it were returned, a
  // client could store it in localStorage "just for convenience" and the
  // httpOnly cookie would stop being a protection.
  const { token: _dropped, ...safe } = data;

  const res = Response.json(safe, { status: upstream.status });
  res.headers.append(
    "set-cookie",
    [
      `${SESSION_COOKIE}=${token}`,
      "Path=/",
      "HttpOnly",
      "SameSite=Lax",
      expires > 0 ? `Max-Age=${Math.max(0, expires - Math.floor(Date.now() / 1000))}` : "",
      // Secure is conditional because sua serves plain HTTP: setting it
      // unconditionally would break every local session. In production,
      // behind the TLS terminator the README calls for, NODE_ENV is
      // "production" and the flag is on.
      process.env.NODE_ENV === "production" ? "Secure" : "",
    ]
      .filter(Boolean)
      .join("; "),
  );
  return res;
}

export async function DELETE() {
  // Best effort upstream, then clear the cookie regardless. A logout that
  // fails because the backend is down must still log you out locally.
  try {
    const { cookies } = await import("next/headers");
    const token = (await cookies()).get(SESSION_COOKIE)?.value;
    if (token) {
      await fetch(`${API_BASE}/api/auth/logout`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
        body: "{}",
        cache: "no-store",
      });
    }
  } catch {
    // ignored on purpose — see above
  }
  const res = Response.json({ ok: true });
  res.headers.append(
    "set-cookie",
    `${SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0`,
  );
  return res;
}
