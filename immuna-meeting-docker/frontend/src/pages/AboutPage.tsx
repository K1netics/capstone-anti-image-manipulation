import HeroSection from "../components/about/HeroSection";
import ProjectMotivation from "../components/about/ProjectMotivation";
import OurApproach from "../components/about/OurApproach";
import SecurityGuarantees from "../components/about/SecurityGuarantees";
import TechnicalArchitecture from "../components/about/TechnicalArchitecture";
import AcademicContext from "../components/about/AcademicContext";
 
export default function AboutPage() {
  return (
    <main className="w-full">
      <HeroSection />
 
      <div className="max-w-7xl mx-auto px-4 md:px-8 pt-16 pb-10 space-y-8">
        <ProjectMotivation />
        <OurApproach />
        <SecurityGuarantees />
        <TechnicalArchitecture />
        <AcademicContext />
      </div>
    </main>
  );
}
 
