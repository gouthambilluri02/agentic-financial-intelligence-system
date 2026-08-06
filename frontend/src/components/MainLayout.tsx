import { useState, type ReactNode } from "react";
import { Navbar } from "./layout/Navbar";

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  const [isDarkMode, setIsDarkMode] = useState(false);

  function toggleTheme() {
    setIsDarkMode((prev) => !prev);
  }

  return (
    <div className={isDarkMode ? "app app--dark" : "app"}>
      <Navbar
        isDarkMode={isDarkMode}
        onToggleTheme={toggleTheme}
      />

      <main className="app__content">
        {children}
      </main>
    </div>
  );
}