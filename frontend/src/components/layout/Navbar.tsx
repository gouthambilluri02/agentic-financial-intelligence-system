import { BrainCircuit, Moon, Sun } from "lucide-react";

interface NavbarProps {
  isDarkMode: boolean;
  onToggleTheme: () => void;
}

export function Navbar({ isDarkMode, onToggleTheme }: NavbarProps) {
  return (
    <header className="navbar">
      <div className="navbar__brand">
        <div className="navbar__logo">
          <BrainCircuit size={22} />
        </div>

        <div>
          <h1>FinAgent AI</h1>
          <p>Agentic Financial Intelligence</p>
        </div>
      </div>

      <button
        type="button"
        className="navbar__theme-button"
        onClick={onToggleTheme}
        aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
      >
        {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
      </button>
    </header>
  );
}