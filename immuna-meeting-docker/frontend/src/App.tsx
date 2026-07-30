import { useState } from "react";
import Header from "./components/Layout/Header";
import Footer from "./components/Layout/Footer";
import HomePage from "./pages/HomePage";
import AboutPage from "./pages/AboutPage";

type Page = "home" | "about";

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("home");

return (
  <div className="relative isolate min-h-screen flex flex-col bg-[#0A0A0F]">
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
  <div
    style={{
      position: "absolute",
      right: "-300px",
      bottom: "-400px",
      width: "900px",
      height: "900px",
      backgroundImage: "url('/logo.png')",
      backgroundRepeat: "no-repeat",
      backgroundSize: "contain",
      backgroundPosition: "center",
      opacity: 0.30,
      filter: "brightness(0.7) saturate(0.6)",
      transform: "rotate(90deg)",
      transformOrigin: "center",
    }}
  />
    </div>
    <div className="relative z-10 flex flex-col min-h-screen">
      <Header currentPage={currentPage} onNavigate={setCurrentPage} />
      <div className="flex-1">
        {currentPage === "home" ? <HomePage /> : <AboutPage />}
      </div>
      <Footer />
    </div>
  </div>
);
}
