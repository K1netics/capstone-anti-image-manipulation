import React from "react";
import Card from "./Card";

interface FeatureCardProps {
  icon: any;
  title: string;
  description: string;
}

export default function FeatureCard({ icon, title, description }: FeatureCardProps) {
  return (
    <Card>
      <div
        className="w-10 h-10 rounded-full flex items-center justify-center mb-4"
        style={{ backgroundColor: "var(--muted)" }}
      >
        {/* Gradient definition */}
        <svg width="0" height="0">
          <defs>
            <linearGradient id="iconGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#8FE3F7" />
              <stop offset="50%" stopColor="#C6A4FF" />
              <stop offset="100%" stopColor="#F3B6D6" />
            </linearGradient>
          </defs>
        </svg>

        {/* Apply gradient to icon */}
        {React.cloneElement(icon, {
          stroke: "url(#iconGradient)",
        })}
      </div>

      <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--foreground)" }}>
        {title}
      </h3>

      <p className="text-xs leading-relaxed" style={{ color: "var(--muted-foreground)" }}>
        {description}
      </p>
    </Card>
  );
}
