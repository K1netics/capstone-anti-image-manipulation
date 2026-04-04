import { Shield, ScanFace, Workflow } from "lucide-react";
import FeatureCard from "../ui/FeatureCard";

const FEATURES = [
  {
    icon: <ScanFace className="w-5 h-5" />,
    title: "Human-region masking",
    description:
      "Paint the face or other sensitive region directly in the new editor before sending the request to the API.",
  },
  {
    icon: <Shield className="w-5 h-5" />,
    title: "No Gradio dependency",
    description:
      "This integrated workspace talks to its own API layer instead of embedding the original Gradio application.",
  },
  {
    icon: <Workflow className="w-5 h-5" />,
    title: "Safe iteration path",
    description:
      "The original PhotoGuard demo stays untouched while we shape the eventual product frontend and backend contract here.",
  },
];

export default function FeaturesGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {FEATURES.map((feature) => (
        <FeatureCard
          key={feature.title}
          icon={feature.icon}
          title={feature.title}
          description={feature.description}
        />
      ))}
    </div>
  );
}
