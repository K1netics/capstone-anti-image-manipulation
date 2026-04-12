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
        <span style={{ color: "var(--accent)" }}>{icon}</span>
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
