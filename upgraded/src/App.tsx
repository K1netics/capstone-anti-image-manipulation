import { useState } from "react";
import Header from "./components/Layout/Header";
import Footer from "./components/Layout/Footer";
import HomePage from "./pages/HomePage";
import AboutPage from "./pages/AboutPage";

type Page = "home" | "about";

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("home");

  return (
    <div className="min-h-screen flex flex-col" style={{ backgroundColor: "var(--background)" }}>
      <Header currentPage={currentPage} onNavigate={setCurrentPage} />
      <div className="flex-1">
        {currentPage === "home" ? <HomePage /> : <AboutPage />}
      </div>
      <Footer />
    </div>
  );
}
