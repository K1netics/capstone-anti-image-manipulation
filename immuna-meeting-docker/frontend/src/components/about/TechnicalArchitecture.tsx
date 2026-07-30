import { Monitor, Server, Lock, CheckCircle2, ArrowRight } from "lucide-react";
import Card from "../ui/Card";

const STEPS = [
  {
    icon: Monitor,
    label: "User",
    sub: "Web Application\n(React)",
  },
  {
    icon: Server,
    label: "Cloud",
    sub: "Azure VM\nnginx (TLS / HTTPS)\nContainerized Docker",
  },
  {
    icon: Lock,
    label: "Engine",
    sub: "Python\nFastAPI Back End\nContainerized Docker",
  },
  {
    icon: CheckCircle2,
    label: "Output",
    sub: "Protected Image\nAI Manipulation Fails / Degrades",
  },
];

export default function SystemArchitecture() {
  return (
    <Card>
      <div className="flex items-center gap-3 mb-6">
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: "var(--muted)" }}
        >
          <Server className="w-5 h-5" style={{ color: "#c7d4f0" }} />
        </div>
        <h2 className="text-lg font-semibold" style={{ color: "var(--foreground)" }}>
          System Architecture
        </h2>
      </div>

      <div className="rounded-xl p-5 md:p-6 my-2" style={{ backgroundColor: "transparent" }}>
        <div className="flex items-center justify-center gap-4">
          {STEPS.map((step, i) => {
            const Icon = step.icon;

            return (
              <div key={step.label} className="flex items-center">
                <div
                  className="flex flex-col items-center justify-center gap-3 rounded-2xl w-[240px] h-[230px] shrink-0"
                  style={{
                    border: "2px solid #c7d4f0",
                    backgroundColor: "rgba(10, 10, 15, 0.92)",
                  }}
                >
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center"
                    style={{
                      backgroundColor: "#11131c",
                      color: "#ffffff",
                    }}
                  >
                    <Icon className="w-5 h-5" />
                  </div>

                  <div className="text-center px-4">
                    <p className="text-base font-semibold tracking-wide" style={{ color: "#ffffff" }}>
                      {step.label}
                    </p>
                    <p
                      className="text-sm md:text-base leading-tight mt-1 whitespace-pre-line"
                      style={{ color: "#c7d4f0" }}
                    >
                      {step.sub}
                    </p>
                  </div>
                </div>

                {i < STEPS.length - 1 && (
                  <div className="w-10 h-[230px] flex items-center justify-center shrink-0">
                    <ArrowRight className="w-5 h-5" style={{ color: "#c7d4f0" }} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <p className="text-base leading-relaxed mt-4" style={{ color: "#bcb1da" }}>
        The architecture enforces strict separation between the frontend, cloud entry point, processing engine, and output layers, 
        ensuring controlled data flow and execution boundaries. Each request is processed sequentially, with the backend operating 
        in an ephemeral, containerized environment. This approach minimizes data exposure by eliminating persistent storage and 
        ensures images are handled securely during processing.
      </p>
    </Card>
  );
}