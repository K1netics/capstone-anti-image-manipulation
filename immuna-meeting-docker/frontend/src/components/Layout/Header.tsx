import { BookOpen, Github } from "lucide-react";

type Page = "home" | "about";

interface HeaderProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
}

export default function Header({ currentPage, onNavigate }: HeaderProps) {
  return (
    <header
      className="sticky top-0 z-50 w-full"
      style={{
        backgroundImage: "url('/holographic.jpg')",
        backgroundSize: "cover",
        backgroundPosition: "right top",
        borderBottom: "1px solid rgba(255,255,255,0.25)",
      }}
    >
      <div className="max-w-7xl mx-auto px-8 py-4 flex items-center justify-between">
        <button
          onClick={() => onNavigate("home")}
          className="flex items-center gap-3 hover:opacity-80 transition-opacity"
        >
          <div className="w-12 h-12 rounded-xl overflow-hidden flex items-center justify-center">
            <img
              src="/logo.png"
              alt="Immuna logo"
              className="w-full h-full object-contain"
              style={{ 
                transform: "rotate(90deg)",
                filter: "drop-shadow(0 0 4px rgb(126, 78, 189))"
              }}
            />
          </div>

          <div className="text-left">
            <div 
              className="text-base font-semibold leading-tight"
              style={{ 
                color: "#65508c",
                fontFamily: "'Monument Extended', sans-serif",
                textTransform: "uppercase"
              }}
            >
              Immuna
            </div>
            </div>
        </button>

        <nav className="flex items-center gap-8">
          {(["home", "about"] as Page[]).map((page) => (
            <button
              key={page}
              onClick={() => onNavigate(page)}
              className="text-base capitalize transition-colors hover:text-[#1A1A24]"
              style={{
                color: currentPage === page
                ? "#412574"
                : "#634e9c",
              fontWeight: currentPage === page ? 500 : 400,
              }}
            >
            {page.charAt(0).toUpperCase() + page.slice(1)}
        </button>
          ))}

          <a
            href="https://github.com/mochibearr/immuna"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-opacity hover:opacity-70 text-white"
            aria-label="GitHub"
          >
            <Github className="w-6 h-6" />
          </a>
        </nav>
      </div>
    </header>
  );
}