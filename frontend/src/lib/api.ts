/**
 * Typed API client.
 *
 * Two base URLs, deliberately: Server Components resolve the backend by its
 * compose service name, while the browser can only reach the published port on
 * localhost. Collapsing them into one variable breaks under Docker.
 *
 * Every call goes through this module so that the Claude-backed endpoints
 * stay server-side and the API key never reaches the browser bundle.
 */

import type {
  Course,
  CourseCreate,
  CoursePage,
  CourseUpdate,
  ReadinessResponse,
} from "./types";

export const SERVER_API_BASE_URL = process.env.API_BASE_URL ?? "http://localhost:8000";
export const BROWSER_API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const DEFAULT_TIMEOUT_MS = 5000;

/**
 * A result type rather than exceptions: an unreachable backend is an expected
 * state for this app, so callers should be forced to render it.
 */
export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: string };

interface JsonRequestOptions extends Omit<RequestInit, "cache" | "signal"> {
  baseUrl?: string;
  timeoutMs?: number;
  /** Non-2xx statuses whose JSON body is still meaningful (e.g. 503 readiness). */
  acceptStatuses?: readonly number[];
}

export async function requestJson<T>(
  path: string,
  options: JsonRequestOptions = {},
): Promise<ApiResult<T>> {
  const {
    baseUrl = SERVER_API_BASE_URL,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    acceptStatuses = [],
    ...init
  } = options;

  try {
    const response = await fetch(`${baseUrl}${path}`, {
      ...init,
      headers: { accept: "application/json", ...init.headers },
      signal: AbortSignal.timeout(timeoutMs),
      // No caching: these are live signals, and a stale health report is worse
      // than no health report.
      cache: "no-store",
    });

    if (!response.ok && !acceptStatuses.includes(response.status)) {
      // The body of an error response is often a FastAPI {"detail": ...},
      // which is far more useful to show than the bare status.
      return { ok: false, error: await describeError(response, path) };
    }
    // 204 has no body at all, so response.json() would throw. Callers that
    // expect one declare ApiResult<null>.
    if (response.status === 204) {
      return { ok: true, data: null as T };
    }
    return { ok: true, data: (await response.json()) as T };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "unknown error" };
  }
}

async function describeError(response: Response, path: string): Promise<string> {
  const fallback = `HTTP ${response.status} from ${path}`;
  try {
    const body: unknown = await response.json();
    const detail = (body as { detail?: unknown })?.detail;
    return typeof detail === "string" ? detail : fallback;
  } catch {
    return fallback;
  }
}

const JSON_HEADERS = { "content-type": "application/json" } as const;

/** A 503 here is a successful *report* of an unhealthy system, not a failure. */
export function getReadiness(baseUrl?: string): Promise<ApiResult<ReadinessResponse>> {
  return requestJson<ReadinessResponse>("/health/ready", { baseUrl, acceptStatuses: [503] });
}

// --- courses -----------------------------------------------------------------
//
// Thin wrappers, deliberately: they name the endpoints and their types, and
// leave every policy decision (caching, error rendering, when to refetch) to
// the components in M6. The screens do not exist yet, but the client does, so
// the DTO mirrors in types.ts are exercised by the compiler rather than only
// by hand.

export function listCourses(
  params: { limit?: number; offset?: number } = {},
  baseUrl?: string,
): Promise<ApiResult<CoursePage>> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));
  const suffix = query.size > 0 ? `?${query}` : "";
  return requestJson<CoursePage>(`/api/v1/courses${suffix}`, { baseUrl });
}

export function getCourse(id: string, baseUrl?: string): Promise<ApiResult<Course>> {
  return requestJson<Course>(`/api/v1/courses/${id}`, { baseUrl });
}

export function createCourse(
  course: CourseCreate,
  baseUrl?: string,
): Promise<ApiResult<Course>> {
  return requestJson<Course>("/api/v1/courses", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(course),
    baseUrl,
  });
}

export function updateCourse(
  id: string,
  changes: CourseUpdate,
  baseUrl?: string,
): Promise<ApiResult<Course>> {
  return requestJson<Course>(`/api/v1/courses/${id}`, {
    method: "PATCH",
    headers: JSON_HEADERS,
    body: JSON.stringify(changes),
    baseUrl,
  });
}

/** Deletes the course and, by cascade, its documents and chunks. */
export function deleteCourse(id: string, baseUrl?: string): Promise<ApiResult<null>> {
  return requestJson<null>(`/api/v1/courses/${id}`, { method: "DELETE", baseUrl });
}
