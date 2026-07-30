import { Shield, Star, Zap } from "lucide-react";
import FeatureCard from "../ui/FeatureCard";

const FEATURES = [
  {
    icon: <Shield className="w-5 h-5" />,
    title: "Proactive Defense",
    description:
      "Images are immunized at the source using model-aware pertubations designed to interfere with AI editing systems.",
  },
  {
    icon: <Star className="w-5 h-5" />,
    title: "Visual Integrity Preserved",
    description:
      "Protection is applied without noticeable impact to human perception, maintaining near-original image quality.",
  },
  {
    icon: <Zap className="w-5 h-5" />,
    title: "Real-Time Processing",
    description:
      "Optimized backend pipelines enable fast, interactive image protection with immediate feedback.",
  },
];

export default function FeaturesGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {FEATURES.map((f) => (
        <FeatureCard key={f.title} icon={f.icon} title={f.title} description={f.description} />
      ))}
    </div>
  );
}
