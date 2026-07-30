import { Shield, Workflow, Eye } from "lucide-react";
import Card from "../ui/Card";

const APPROACH_CARDS = [
  {
    icon: Shield,
    title: "Preemptive Protection",
    desc: "Immuna prevents manipulation at the source instead of detecting it after distribution.",
  },
  {
    icon: Workflow,
    title: "Adversarial Perturbations",
    desc: "Low-visibility, model-aware perturbations are applied to selected regions of the image.",
  },
  {
    icon: Eye,
    title: "Human-Friendly, AI-Hostile",
    desc: "The image remains visually intact for people while causing AI edits to fail or degrade.",
  },
];

export default function OurApproach() {
  return (
    <Card>
      <h2 className="text-lg font-semibold mb-3" style={{ color: "var(--foreground)" }}>
        Our Approach
      </h2>

      <p className="text-base mb-6" style={{ color: "#bcb1da" }}>
        Instead of detecting manipulation after it occurs, Immuna prevents it at the generation stage.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {APPROACH_CARDS.map((item) => {
          const Icon = item.icon;

          return (
            <div
              key={item.title}
              className="rounded-xl border p-4"
              style={{ borderColor: "var(--border)" }}
            >
              <div className="flex items-center gap-3 mb-3">
                <div
                  className="w-9 h-9 rounded-full flex items-center justify-center"
                  style={{ backgroundColor: "var(--secondary)" }}
                >
                  <Icon className="w-4 h-4" style={{ color: "var(--accent)" }} />
                </div>
                <h3 className="font-semibold" style={{ color: "var(--foreground)" }}>
                  {item.title}
                </h3>
              </div>

              <p className="text-sm leading-relaxed m-0" style={{ color: "#bcb1da" }}>
                {item.desc}
              </p>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
