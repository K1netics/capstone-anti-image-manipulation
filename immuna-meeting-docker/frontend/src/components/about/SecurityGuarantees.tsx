import { Lock, CheckCircle2 } from "lucide-react";
import Card from "../ui/Card";

const GUARANTEES = [
  {
    title: "Ephemeral Processing",
    desc: "Images are processed in-memory and are never written to disk.",
  },
  {
    title: "Zero Data Retention",
    desc: "No images, logs, or user data are stored after processing.",
  },
  {
    title: "Isolated Execution",
    desc: "Your images never leave our secure processing environment. We don't use third-party analytics, CDNs, or external services.",
  },
  {
    title: "Containerized Isolation",
    desc: "Backend processing runs in containerized environments with strict boundaries.",
  },
  {
    title: "Input Safeguards",
    desc: "Validation ensures only safe file types and sizes are processed.",
  },
  {
    title: "Secure Connection",
    desc: "All data is transmitted over encrypted HTTPS connections.",
  },
];

export default function SecurityGuarantees() {
  return (
    <Card>
      <div className="flex items-center gap-3 mb-6">
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: "var(--secondary)" }}
        >
          <Lock className="w-5 h-5" style={{ color: "var(--accent)" }} />
        </div>
        <h2 className="text-lg font-semibold" style={{ color: "var(--foreground)" }}>
          Security &amp; Privacy Guarantees
        </h2>

        <p
        className="text-sm mt-1 leading-relaxed"
        style={{ color: "#bcb1da" }}
        >
        Designed with a privacy-first architecture to ensure secure, ephemeral image processing.
        </p>
        
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-10 gap-y-6">
        {GUARANTEES.map((item) => (
          <div key={item.title} className="flex items-start gap-3">
            <CheckCircle2
              className="w-5 h-5 flex-shrink-0 mt-0.5"
              style={{ color: "var(--accent)" }}
            />
            <div>
              <h3
                className="text-base font-semibold mb-1"
                style={{ color: "var(--foreground)" }}
              >
                {item.title}
              </h3>
              <p className="text-base leading-relaxed" style={{ color: "#bcb1da" }}>
                {item.desc}
              </p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
