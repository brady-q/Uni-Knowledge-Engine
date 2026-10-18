/**
 * DTOs mirroring backend/app/schemas.py.
 *
 * Hand-written for now. Generating them from the backend's OpenAPI schema is
 * the obvious next step once there are more than a handful; at that point the
 * duplication starts to cost more than the tooling.
 */

export type HealthStatus = "ok" | "error";

export interface LivenessResponse {
  status: "ok";
  service: string;
  version: string;
}

export interface ComponentStatus {
  status: HealthStatus;
  detail: string | null;
}

export interface ReadinessResponse {
  status: HealthStatus;
  checks: Record<string, ComponentStatus>;
}
