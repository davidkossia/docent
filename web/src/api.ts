// Types mirror the API contract in PLAN section 5 (docs/api.md).

export type Citation = { passage: number; cited_text: string };
export type Segment = { text: string; citations: Citation[] };
export type Passage = {
  n: number;
  chunk_id: string;
  book_title: string;
  section_title: string;
  page_start: number;
  page_end: number;
  text: string;
};
export type AskStatus = "answered" | "no_answer" | "refused";
export type AskResponse = {
  status: AskStatus;
  segments: Segment[];
  passages: Passage[];
  meta: { model: string; setup: string; latency_ms: number };
};

export type ErrorCode = "invalid_question" | "daily_limit" | "upstream_error" | "timeout";
export type AskError = { code: ErrorCode | "network" | "bad_response"; message: string };
export type AskResult = { ok: true; data: AskResponse } | { ok: false; error: AskError };

export const MAX_QUESTION_LENGTH = 500;

const ERROR_CODES: ReadonlySet<string> = new Set([
  "invalid_question",
  "daily_limit",
  "upstream_error",
  "timeout",
]);

export function apiUrl(): string {
  return import.meta.env.VITE_API_URL ?? "http://localhost:8787";
}

function isAskResponse(b: unknown): b is AskResponse {
  if (typeof b !== "object" || b === null) return false;
  const r = b as Record<string, unknown>;
  return (
    (r.status === "answered" || r.status === "no_answer" || r.status === "refused") &&
    Array.isArray(r.segments) &&
    Array.isArray(r.passages)
  );
}

const TIMEOUT: AskResult = { ok: false, error: { code: "timeout", message: "The request timed out." } };

function isAbort(e: unknown, signal?: AbortSignal): boolean {
  return (
    !!signal?.aborted ||
    (e instanceof DOMException && (e.name === "TimeoutError" || e.name === "AbortError"))
  );
}

// Callers pass an AbortSignal (App uses AbortSignal.timeout(30_000)); an abort maps to `timeout`.
export async function ask(question: string, signal?: AbortSignal): Promise<AskResult> {
  let res: Response;
  try {
    res = await fetch(`${apiUrl()}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question.trim() }),
      signal,
    });
  } catch (e) {
    if (isAbort(e, signal)) return TIMEOUT;
    return { ok: false, error: { code: "network", message: "Could not reach the server." } };
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch (e) {
    if (isAbort(e, signal)) return TIMEOUT;
    return { ok: false, error: { code: "bad_response", message: "The server sent an unreadable response." } };
  }
  if (res.ok) {
    if (isAskResponse(body)) return { ok: true, data: body };
    return { ok: false, error: { code: "bad_response", message: "The server sent an unreadable response." } };
  }
  const err = (body as { error?: { code?: string; message?: string } } | null)?.error;
  if (err && typeof err.code === "string" && ERROR_CODES.has(err.code)) {
    return { ok: false, error: { code: err.code as ErrorCode, message: err.message ?? "" } };
  }
  return { ok: false, error: { code: "bad_response", message: "The server sent an unexpected error." } };
}
