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

/** Mirrors app/schemas.py CourseRead. */
export interface Course {
  id: string;
  code: string;
  title: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface CourseCreate {
  code: string;
  title: string;
  description?: string | null;
}

/**
 * PATCH body. Every field is optional, and the distinction matters: omitting a
 * key leaves the column alone, while sending `description: null` clears it.
 */
export interface CourseUpdate {
  code?: string;
  title?: string;
  description?: string | null;
}

export interface CoursePage {
  items: Course[];
  /** Total matching rows, not the length of `items`. */
  total: number;
  limit: number;
  offset: number;
}
