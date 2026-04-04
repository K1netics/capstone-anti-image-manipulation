import Card from "../components/ui/Card";

const NOTES = [
  {
    title: "What changed here",
    body:
      "This integrated project is now its own frontend and API workspace. It no longer embeds the Gradio demo, and it leaves the original PhotoGuard codepath untouched for comparison and fallback.",
  },
  {
    title: "How it stays low-risk",
    body:
      "All new UI and API files live under /home/tobi/photoguard/intergrated. The original backend, original frontend, and the existing demo entrypoint remain exactly where they were.",
  },
  {
    title: "What it is aiming for",
    body:
      "The goal is a normal product architecture: React handles upload, masking, and result presentation; a Python API handles model loading, immunization, and generation; Gradio is no longer part of the main request path.",
  },
];

const STACK = [
  "React + Vite frontend in /home/tobi/photoguard/intergrated/src",
  "FastAPI-style Python backend in /home/tobi/photoguard/intergrated/api",
  "Model artifacts loaded from /home/tobi/photoguard/artifacts/local_inpaint_model by default",
  "Shared immunization logic reused from the untouched original backend modules",
];

export default function AboutPage() {
  return (
    <main className="w-full">
      <section className="px-6 pt-16 pb-8 text-center" style={{ backgroundColor: "var(--background)" }}>
        <h1 className="text-4xl md:text-5xl font-semibold mb-5" style={{ color: "var(--foreground)" }}>
          About The Integrated Workspace
        </h1>
        <p className="max-w-3xl mx-auto text-base leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
          This directory is a sandbox for the real product shape: separate frontend, separate API,
          and the original PhotoGuard app left intact while we test the new flow.
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
              1. Start the API from <code>/home/tobi/photoguard/intergrated</code> with <code>python -m uvicorn api.main:app --reload --port 8000</code>.
            </p>
            <p>
              2. Start the frontend from <code>/home/tobi/photoguard/intergrated</code> with <code>npm install</code> and <code>npm run dev</code>.
            </p>
            <p>
              3. Open the Vite app and use the built-in mask editor instead of the old Gradio page.
            </p>
          </div>
        </Card>
      </div>
    </main>
  );
}
