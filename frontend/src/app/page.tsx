/**
 * System status page.
 *
 * A Server Component: the fetch runs inside the container and reaches the API
 * by its compose service name, which makes this page a live check of the
 * service wiring rather than a static placeholder. It is replaced by the
 * course list once there are courses to list.
 */

import { getReadiness, SERVER_API_BASE_URL } from "@/lib/api";
import type { ComponentStatus } from "@/lib/types";

// Never prerendered at build time: the backend is not up during `next build`.
export const dynamic = "force-dynamic";

const MILESTONES: ReadonlyArray<{ id: string; label: string }> = [
  { id: "M1", label: "Scaffolding, Docker, health checks — done" },
  { id: "M2", label: "Schema, migrations, course CRUD" },
  { id: "M3", label: "PDF upload, text extraction, chunking" },
  { id: "M4", label: "Embeddings and vector retrieval" },
  { id: "M5", label: "Cited answers via Claude" },
  { id: "M6", label: "Course and Q&A screens" },
  { id: "M7", label: "Eval harness and hardening" },
];

function Check({ name, check }: { name: string; check: ComponentStatus }) {
  return (
    <div className="check">
      <span className={check.status === "ok" ? "dot dot-ok" : "dot dot-err"} />
      <span className="check-name">{name}</span>
      <span className="check-detail">{check.detail ?? check.status}</span>
    </div>
  );
}

export default async function HomePage() {
  const readiness = await getReadiness();

  return (
    <main>
      <h1>Uni Knowledge Engine</h1>
      <p className="lede">
        Retrieval-augmented question answering over university course PDFs, with every claim cited
        back to a document and page.
      </p>

      <section className="card">
        <h2>Backend readiness</h2>
        {readiness.ok ? (
          Object.entries(readiness.data.checks).map(([name, check]) => (
            <Check key={name} name={name} check={check} />
          ))
        ) : (
          <p className="error">
            unreachable at {SERVER_API_BASE_URL}: {readiness.error}
          </p>
        )}
      </section>

      <section className="card">
        <h2>Milestones</h2>
        <ul className="roadmap">
          {MILESTONES.map((milestone) => (
            <li key={milestone.id}>
              <strong>{milestone.id}</strong>
              {milestone.label}
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
