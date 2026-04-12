import Card from "../components/ui/Card";

const NOTES = [
  {
    title: "What changed here",
    body:
      "This upgraded project is its own frontend and API workspace. It no longer embeds the Gradio demo, and it keeps a local backend copy so model changes can happen here without depending on the older app.",
  },
  {
    title: "How it stays low-risk",
    body:
      "All new UI, API, and backend files live under /home/tobi/photoguard/upgraded. The upgraded workspace does not need to import code from elsewhere in the repo.",
  },
  {
    title: "What it is aiming for",
    body:
      "The goal is a normal product architecture: React handles upload, masking, and result presentation; a Python API handles model loading, immunization, and generation; Gradio is no longer part of the main request path.",
  },
];

const STACK = [
  "React + Vite frontend in /home/tobi/photoguard/upgraded/src",
  "FastAPI-style Python backend in /home/tobi/photoguard/upgraded/api",
  "Local backend defense code in /home/tobi/photoguard/upgraded/backend",
  "Optional local model artifacts in /home/tobi/photoguard/upgraded/artifacts/local_inpaint_model",
];

export default function AboutPage() {
  return (
    <main className="w-full">
      <section className="px-6 pt-16 pb-8 text-center" style={{ backgroundColor: "var(--background)" }}>
        <h1 className="text-4xl md:text-5xl font-semibold mb-5" style={{ color: "var(--foreground)" }}>
          About The Upgraded Workspace
        </h1>
        <p className="max-w-3xl mx-auto text-base leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
          This directory is a sandbox for the upgraded defense work: separate frontend, separate API,
          and a local backend copy that can evolve independently.
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
              1. Start the API from <code>/home/tobi/photoguard/upgraded</code> with <code>python -m uvicorn api.main:app --reload --port 8000</code>.
            </p>
            <p>
              2. Start the frontend from <code>/home/tobi/photoguard/upgraded</code> with <code>npm install</code> and <code>npm run dev</code>.
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
