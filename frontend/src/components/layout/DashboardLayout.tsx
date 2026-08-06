import type { ReactNode } from "react";

interface DashboardLayoutProps {
  leftPanel: ReactNode;
  mainPanel: ReactNode;
  rightPanel: ReactNode;
}

export function DashboardLayout({
  leftPanel,
  mainPanel,
  rightPanel,
}: DashboardLayoutProps) {
  return (
    <div className="dashboard">
      <aside className="dashboard__left">{leftPanel}</aside>

      <section className="dashboard__main">{mainPanel}</section>

      <aside className="dashboard__right">{rightPanel}</aside>
    </div>
  );
}