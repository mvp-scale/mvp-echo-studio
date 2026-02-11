import type { ReactNode } from "react";
import HealthIndicator from "./HealthIndicator";
import ApiKeyInput from "./ApiKeyInput";
import { LatestBadge, TotalBadge } from "./UsageBadge";
import type { LastRun } from "./UsageBadge";

interface LayoutProps {
  children: ReactNode;
  usageRefreshKey?: number;
  lastRun?: LastRun | null;
}

export default function Layout({ children, usageRefreshKey = 0, lastRun = null }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col bg-surface-0">
      <header className="flex items-center px-8 py-3.5 border-b border-border gap-6">
        {/* Left — title */}
        <h1 className="text-lg font-semibold tracking-tight text-white shrink-0">
          MVP-Echo Scribe
        </h1>

        {/* Middle — usage stats, fixed positions */}
        <div className="flex-1 flex items-center justify-between min-w-0">
          <div className="shrink-0">
            <LatestBadge lastRun={lastRun} />
          </div>
          <div className="shrink-0">
            <TotalBadge refreshKey={usageRefreshKey} />
          </div>
        </div>

        {/* Right — controls */}
        <div className="flex items-center gap-3 shrink-0">
          <ApiKeyInput />
          <HealthIndicator />
        </div>
      </header>
      <main className="flex-1 flex flex-col min-h-0">{children}</main>
    </div>
  );
}
