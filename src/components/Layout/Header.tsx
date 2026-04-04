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
        backgroundColor: "rgba(250,247,255,0.92)",
        borderBottom: "1px solid var(--border)",
        backdropFilter: "blur(10px)",
      }}
    >
      <div className="max-w-7xl mx-auto px-6 md:px-8 py-4 flex items-center justify-between gap-6">
        <button onClick={() => onNavigate("home")} className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <div className="w-11 h-11 rounded-xl flex items-center justify-center" style={{ backgroundColor: "var(--primary)" }}>
            <BookOpen className="w-6 h-6" style={{ color: "var(--foreground)" }} />
          </div>
          <div className="text-left">
            <div className="text-base font-semibold leading-tight" style={{ color: "var(--foreground)" }}>
              PhotoGuard
            </div>
            <div className="text-sm leading-tight" style={{ color: "var(--muted-foreground)" }}>
              Integrated standalone app
            </div>
          </div>
        </button>

        <nav className="flex items-center gap-6 md:gap-8">
          {(["home", "about"] as Page[]).map((page) => (
            <button
              key={page}
              onClick={() => onNavigate(page)}
              className="text-base capitalize transition-colors"
              style={{
                color: currentPage === page ? "var(--accent)" : "var(--foreground)",
                fontWeight: currentPage === page ? 500 : 400,
              }}
            >
              {page}
            </button>
          ))}
          <a
            href="https://github.com/MadryLab/photoguard"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-opacity hover:opacity-70"
            style={{ color: "var(--foreground)" }}
            aria-label="GitHub"
          >
            <Github className="w-6 h-6" />
          </a>
        </nav>
      </div>
    </header>
  );
}
