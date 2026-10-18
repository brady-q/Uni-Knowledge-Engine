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

import type { ReadinessResponse } from "./types";

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
      return { ok: false, error: `HTTP ${response.status} from ${path}` };
    }
    return { ok: true, data: (await response.json()) as T };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "unknown error" };
  }
}

/** A 503 here is a successful *report* of an unhealthy system, not a failure. */
export function getReadiness(baseUrl?: string): Promise<ApiResult<ReadinessResponse>> {
  return requestJson<ReadinessResponse>("/health/ready", { baseUrl, acceptStatuses: [503] });
}
