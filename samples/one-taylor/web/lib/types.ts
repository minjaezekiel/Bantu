// Shapes the Bantu backend actually returns. Written by hand against
// auth/privacy.b's publicUser projection and recsys/posts.b's hydrate, which
// are the only two places a user or a post is serialised.

export type Band = "minor" | "adult";

export type ExplicitCategory =
  | "suggestive" | "nudity" | "sexual" | "fetish"
  | "violence" | "gore" | "substances";

export interface PublicUser {
  id: number;
  username: string;
  band: Band;
  age: number;
  gender: string;
  joined: string;
  self?: boolean;
  displayName?: string | null;
  bio?: string | null;
  avatar?: string | null;
  seekMinAge?: number;
  seekMaxAge?: number;
  seekGenders?: string;
  /** Present only when the viewer is this user. */
  theme?: "light" | "dark" | "system";
  explicitOptins?: string;
}

export interface PostMedia {
  id: number;
  mime: string;
}

/** "show" renders inline, "blur" needs a tap to reveal, "deny" never arrives. */
export type Display = "show" | "blur";

export interface Post {
  id: number;
  body: string;
  createdAt: string;
  likeCount: number;
  author: PublicUser | null;
  media: PostMedia[];
  tags: string[];
  labels: ExplicitCategory[];
  display: Display;
  blurredFor: ExplicitCategory[];
  score?: number;
  mine?: boolean;
  status?: string;
}

export interface Match {
  user: PublicUser;
  shared: string[];
  why: string;
  score: number;
}

export interface AgeWindow { min: number; max: number }

export interface FeedResponse {
  ok: true;
  posts: Post[];
  nextBefore: number | null;
  band: Band;
  ageWindow: AgeWindow;
}

export interface DiscoverResponse {
  ok: true;
  matches: Match[];
  band: Band;
  range: AgeWindow;
  rangeLocked: boolean;
}

export interface MeResponse {
  ok: true;
  user: PublicUser;
  interests: string[];
  hobbies: string[];
  dreamPartner: string[];
  band: Band;
}

export interface ProfileResponse {
  ok: true;
  user: PublicUser;
  interests: string[];
  hobbies: string[];
  dreamPartner: string[];
  posts: Post[];
  followers: number;
  following: number;
  youFollow: boolean;
  followsYou: boolean;
  canMatch: boolean;
  isYou: boolean;
}

export interface SearchResponse {
  ok: true;
  query: string;
  results: PublicUser[];
  band: Band;
  ageWindow: AgeWindow;
}

export interface ApiError {
  ok: false;
  error: string;
  /** POST /api/posts answers 422 with these when moderation refused it. */
  reasons?: string[];
  status?: string;
  appealable?: boolean;
  blurredFor?: ExplicitCategory[];
}

export type ApiResult<T> = T | ApiError;

export function isError<T>(r: ApiResult<T>): r is ApiError {
  return (r as ApiError).ok === false;
}

/* ── messaging and calls ─────────────────────────────────────────────────── */

export interface ChatMessage {
  id: number;
  body: string;
  mine: boolean;
  kind: "text" | "call";
  /** "live" delivered · "held" awaiting review · "blocked" refused */
  status: "live" | "held" | "blocked";
  createdAt: string;
  readAt: string | null;
}

export interface ThreadSummary {
  id: number;
  with: PublicUser;
  lastMessageAt: string | null;
  preview: string;
  unread: number;
}

export interface ConversationsResponse {
  ok: true;
  conversations: ThreadSummary[];
}

export interface HistoryResponse {
  ok: true;
  conversationId: number;
  with: PublicUser;
  messages: ChatMessage[];
  nextBefore: number | null;
}

export interface RealtimeConfig {
  ok: true;
  wsUrl: string;
  iceServers: RTCIceServer[];
  hasTurn: boolean;
  me: PublicUser;
}

export interface TicketResponse {
  ok: true;
  ticket: string;
  ttl: number;
}

/** Frames the hub sends. Discriminated on `type`, so a switch is exhaustive. */
export type InboundFrame =
  | { type: "welcome"; self: PublicUser; conversations: ThreadSummary[] }
  | { type: "presence"; username: string; online: boolean }
  | { type: "message"; with: string; from: PublicUser; message: ChatMessage }
  | { type: "message-sent"; with: string; message: ChatMessage; verdict: string; note: string | null }
  | { type: "read"; with: string; count: number }
  | { type: "typing"; with: string; on: boolean }
  | { type: "call-ringing"; with: string; from: PublicUser; video: boolean; sdp: RTCSessionDescriptionInit }
  | { type: "call-answer"; with: string; sdp: RTCSessionDescriptionInit }
  | { type: "call-ice"; with: string; candidate: RTCIceCandidateInit }
  | { type: "call-reject"; with: string; reason?: string }
  | { type: "call-end"; with: string }
  | { type: "pong" }
  | { type: "error"; message: string };

/** Frames the client sends. */
export type OutboundFrame =
  | { type: "hello"; ticket: string }
  | { type: "ping" }
  | { type: "message"; to: string; body: string }
  | { type: "read"; to: string }
  | { type: "typing"; to: string; on: boolean }
  | { type: "call-offer"; to: string; sdp: RTCSessionDescriptionInit; video: boolean }
  | { type: "call-answer"; to: string; sdp: RTCSessionDescriptionInit }
  | { type: "call-ice"; to: string; candidate: RTCIceCandidateInit }
  | { type: "call-reject"; to: string }
  | { type: "call-end"; to: string };

export type WireState = "connecting" | "open" | "closed";
