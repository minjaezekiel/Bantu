import type { NextConfig } from "next";

// The browser never talks to the Bantu backend directly: every call goes
// through a route handler in app/api/*, which attaches the session token from
// an httpOnly cookie. That is what keeps the token out of client JavaScript,
// and it also sidesteps sua's one-Set-Cookie-per-response limit.
const config: NextConfig = {
  reactStrictMode: true,
  env: {
    // Read at build time only for the dev banner; the proxy reads it at runtime.
    ONE_TAYLOR_API: process.env.ONE_TAYLOR_API ?? "http://127.0.0.1:4100",
  },
};

export default config;
