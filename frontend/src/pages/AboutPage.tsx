import Card from "../components/ui/Card";

const NOTES = [
  {
    title: "What changed here",
    body:
      "This dockerized project packages the React frontend, the FastAPI backend, and the nginx proxy as one deployable bundle.",
  },
  {
    title: "How it stays low-risk",
    body:
      "The deployment bundle keeps its own copied frontend and backend files under /home/tobi/photoguard/dockerized without rewriting the original demo paths.",
  },
  {
    title: "What it is aiming for",
    body:
      "The goal is a production-shaped delivery path: React handles upload and masking, a Python API handles generation, and nginx serves the UI while proxying API requests.",
  },
];

const STACK = [
  "React + Vite frontend in /home/tobi/photoguard/dockerized/frontend",
  "FastAPI backend in /home/tobi/photoguard/dockerized/backend",
  "nginx proxy and production image build in /home/tobi/photoguard/dockerized/proxy",
  "Model artifacts loaded from /home/tobi/photoguard/artifacts/local_inpaint_model by default",
  "Shared immunization logic copied from the original backend modules",
];

export default function AboutPage() {
  return (
    <main className="w-full">
      <section className="px-6 pt-16 pb-8 text-center" style={{ backgroundColor: "var(--background)" }}>
        <h1 className="text-4xl md:text-5xl font-semibold mb-5" style={{ color: "var(--foreground)" }}>
          About The Dockerized Workspace
        </h1>
        <p className="max-w-3xl mx-auto text-base leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
          This directory is the deployment-oriented version of PhotoGuard: containerized frontend,
          containerized backend, and a proxy layer that can point at a local or remote API.
        </p>
      </section>

      <div className="max-w-7xl mx-auto px-4 md:px-8 pb-10 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {NOTES.map((note) => (
            <Card key={note.title}>
              <h2 className="text-lg font-semibold mb-3" style={{ color: "var(--foreground)" }}>
                {note.title}
              </h2>
              <p className="text-sm leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
                {note.body}
              </p>
            </Card>
          ))}
        </div>

        <Card>
          <h2 className="text-lg font-semibold mb-4" style={{ color: "var(--foreground)" }}>
            Working stack
          </h2>
          <div className="space-y-2 text-sm" style={{ color: "var(--muted-foreground)" }}>
            {STACK.map((item) => (
              <div key={item} className="flex gap-3 items-start">
                <span style={{ color: "var(--accent)" }}>•</span>
                <span>{item}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <h2 className="text-lg font-semibold mb-4" style={{ color: "var(--foreground)" }}>
            Launch sequence
          </h2>
          <div className="space-y-3 text-sm" style={{ color: "var(--muted-foreground)" }}>
            <p>
              1. Build and start the stack from <code>/home/tobi/photoguard/dockerized</code> with <code>docker compose up --build</code>.
            </p>
            <p>
              2. For GPU runs, use <code>docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build</code>.
            </p>
            <p>
              3. Open the web app through nginx and use the built-in mask editor instead of the old Gradio page.
            </p>
          </div>
        </Card>
      </div>
    </main>
  );
}
