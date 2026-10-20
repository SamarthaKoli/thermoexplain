import type { ReactNode } from "react";

export type PageId =
  | "overview"
  | "cfd"
  | "explainability"
  | "generative"
  | "knowledge"
  | "evaluation";

interface AppLayoutProps {
  children: ReactNode;
  activePage: PageId;
  onNavigate: (page: PageId) => void;
}

const navigationItems: Array<{ id: PageId; label: string; description: string }> = [
  { id: "overview", label: "Overview", description: "Workspace status" },
  { id: "cfd", label: "CFD Explorer", description: "Inspect solver data" },
  { id: "explainability", label: "Explainable AI Lab", description: "Surrogate and explanations" },
  { id: "generative", label: "Generative AI Lab", description: "VAE and GAN experiments" },
  { id: "knowledge", label: "Knowledge Assistant", description: "Search technical sources" },
  { id: "evaluation", label: "Evaluation & Validation", description: "Review evidence and limits" },
];

export function AppLayout({ children, activePage, onNavigate }: AppLayoutProps) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="ThermoExplain AI home">
          <span className="brand__mark" aria-hidden="true">
            T
          </span>
          <span>
            <strong>ThermoExplain</strong>
            <span className="brand__suffix">AI</span>
          </span>
        </a>
        <span className="topbar__label">RESEARCH WORKSPACE · NOT AN ENGINEERING APPROVAL TOOL</span>
      </header>
      <div className="workspace">
        <aside className="sidebar" aria-label="Application sections">
          <p className="sidebar__label">WORKSPACE</p>
          <nav className="page-nav" aria-label="Main navigation">
            {navigationItems.map((item) => (
              <button
                aria-current={activePage === item.id ? "page" : undefined}
                className={activePage === item.id ? "page-nav__active" : ""}
                key={item.id}
                onClick={() => onNavigate(item.id)}
                type="button"
              >
                <span>{item.label}</span>
                <small>{item.description}</small>
              </button>
            ))}
          </nav>
          <p className="sidebar__note">
            Results are limited to scanned data and locally available models.
          </p>
        </aside>
        <main className="main-content">{children}</main>
      </div>
      <footer className="footer">
        EV battery thermal management <span aria-hidden="true">·</span> Research
        foundation
      </footer>
    </div>
  );
}
