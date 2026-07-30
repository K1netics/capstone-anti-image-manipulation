import { Shield } from "lucide-react";
import Card from "../ui/Card";
import React from "react";

export default function ProjectMotivation() {
  return (
    <Card>
      <div className="flex items-center gap-3 mb-5">
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: "var(--muted)" }}
        >
          <svg width="0" height="0">
            <defs>
              <linearGradient id="iconGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#8FE3F7" />
                <stop offset="50%" stopColor="#C6A4FF" />
                <stop offset="100%" stopColor="#F3B6D6" />
              </linearGradient>
            </defs>
          </svg>

          {React.cloneElement(<Shield className="w-5 h-5" />, {
            stroke: "url(#iconGradient)",
          })}
        </div>
        <h2 className="text-lg font-semibold" style={{ color: "var(--foreground)" }}>
          Project Motivation
        </h2>
      </div>

      <div className="space-y-4 text-base leading-relaxed" style={{ color: "#bcb1da" }}>
        <p>
          Modern AI image editing tools can realistically modify faces, environments, and identities with minimal effort.
          While powerful, these capabilities introduce serious risks:
        </p>

        <ul className="list-disc ml-6 space-y-1">
          <li>Non-consensual image manipulation and synthethic explicit content.</li>
          <li>Identity misuse and deepfake generation.</li>
          <li>Harassment, reputational damage, and misinformation.</li>
          <li>Unauthorized reuse of personal or creative media.</li>
        </ul>

        <p>
          Current safeguards are largely reactive:
        </p>

        <ul className="list-disc ml-6 space-y-1">
          <li>Detection occurs after content is already distributed.</li>
          <li>Reporting and moderation are inconsistent and slow.</li>
          <li>Watermarks and visible protections are easily removed.</li>
        </ul>

        <p>
          This creates a fundamental gap:
          <br />
          <span className="font-bold">
          Images can be manipulated long before any protection or response is applied.
          </span>
        </p>

        <p>
          Immuna addresses this by embedding protection directly into the image itself, before exposure to AI systems.
        </p>
      </div>
    </Card>
  );
}
