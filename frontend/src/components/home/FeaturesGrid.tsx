import { MessagesSquare, ScanFace, Shield } from "lucide-react";
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
      "This dockerized workspace talks to its own API layer instead of embedding the original Gradio application.",
  },
  {
    icon: <MessagesSquare className="w-5 h-5" />,
    title: "Readable request feedback",
    description:
      "Each `/process` call now returns user-friendly validation, progress, and success messages that the packaged UI shows after every run.",
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
