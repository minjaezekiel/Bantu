import { NextRequest } from "next/server";
import { API_BASE, sessionToken } from "@/lib/api";

/**
 * The one door from the browser to the Bantu backend.
 *
 * Why a proxy and not a direct fetch from the client: the session token lives
 * in an httpOnly cookie. Client JavaScript cannot read it, which means an
 * injected script cannot steal it — and that property only holds if the token
 * is attached on the server. It also means the browser never learns the
 * backend's address.
 *
 * Only /api/* paths are forwarded. Without that check this is an open proxy
 * for anything reachable from the server, which is a far worse bug than the
 * convenience is worth.
 */

const ALLOW = new Set(["GET", "POST", "PUT", "DELETE", "PATCH"]);

async function forward(req: NextRequest, path: string[]) {
  if (!ALLOW.has(req.method)) {
    return Response.json({ ok: false, error: "method not allowed" }, { status: 405 });
  }

  const joined = path.join("/");
  if (joined.includes("..")) {
    return Response.json({ ok: false, error: "not found" }, { status: 404 });
  }

  const search = req.nextUrl.search;
  const target = `${API_BASE}/api/${joined}${search}`;

  const headers: Record<string, string> = {};
  const contentType = req.headers.get("content-type");
  if (contentType) headers["content-type"] = contentType;
  const token = await sessionToken();
  if (token) headers.authorization = `Bearer ${token}`;

  // An admin token is read from the server environment, never from the
  // browser. A moderator opens /moderation and the page works because the
  // server holds the token; the client never sees it.
  const admin = process.env.ONE_TAYLOR_ADMIN_TOKEN;
  if (admin && joined.startsWith("admin/")) headers["x-admin-token"] = admin;

  const body =
    req.method === "GET" || req.method === "DELETE"
      ? undefined
      : await req.text();

  let upstream: Response;
  try {
    upstream = await fetch(target, { method: req.method, headers, body, cache: "no-store" });
  } catch {
    return Response.json(
      { ok: false, error: "the backend is not reachable" },
      { status: 502 },
    );
  }

  // Images come back as bytes; everything else as JSON. Streaming the body
  // through unchanged is what lets /api/media/:id work at all.
  const upstreamType = upstream.headers.get("content-type") ?? "application/json";
  const buf = await upstream.arrayBuffer();
  return new Response(buf, {
    status: upstream.status,
    headers: {
      "content-type": upstreamType,
      "cache-control": upstream.headers.get("cache-control") ?? "no-store",
    },
  });
}

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, (await ctx.params).path);
}
export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, (await ctx.params).path);
}
export async function PUT(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, (await ctx.params).path);
}
export async function DELETE(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, (await ctx.params).path);
}
