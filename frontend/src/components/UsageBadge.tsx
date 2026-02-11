import { useEffect, useState } from "react";
import { fetchUsage } from "../api";
import type { UsageResponse } from "../types";

export interface LastRun {
  audioDuration: number;
  processingTime: number;
  fileSize: number;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

const dot = <span className="text-gray-600 mx-1.5">&middot;</span>;

export function LatestBadge({ lastRun }: { lastRun: LastRun | null }) {
  if (!lastRun || lastRun.audioDuration <= 0 || lastRun.processingTime <= 0) return null;

  const rtFactor = Math.round(lastRun.audioDuration / lastRun.processingTime);

  return (
    <div className="flex items-center text-[13px]">
      <span className="text-white font-semibold mr-2">Latest</span>
      <span className="text-white">{formatBytes(lastRun.fileSize)}</span>
      {dot}
      <span className="text-white">{formatDuration(lastRun.audioDuration)}</span>
      <span className="text-gray-500 mx-1.5">&rarr;</span>
      <span className="text-white">{formatDuration(lastRun.processingTime)}</span>
      {rtFactor > 0 && (
        <span className="text-base font-extrabold text-mvp-blue-light ml-2">
          {rtFactor}x Realtime
        </span>
      )}
    </div>
  );
}

export function TotalBadge({ refreshKey }: { refreshKey: number }) {
  const [usage, setUsage] = useState<UsageResponse | null>(null);

  useEffect(() => {
    fetchUsage().then(setUsage);
  }, [refreshKey]);

  if (!usage || usage.request_count === 0) return null;

  const avgRtFactor =
    usage.total_gpu_seconds > 0
      ? Math.round((usage.total_audio_minutes * 60) / usage.total_gpu_seconds)
      : null;

  return (
    <div className="flex items-center text-[13px]">
      <span className="text-white font-semibold mr-2">Total</span>
      <span className="text-white">{usage.request_count}</span>
      <span className="text-gray-500 ml-0.5">req{usage.request_count !== 1 ? "s" : ""}</span>
      {dot}
      <span className="text-white">{Math.round(usage.total_audio_minutes)}</span>
      <span className="text-gray-500 ml-0.5">min</span>
      {dot}
      <span className="text-white">{formatBytes(usage.total_bytes_uploaded)}</span>
      {dot}
      <span className="text-white">{formatDuration(usage.total_gpu_seconds)}</span>
      <span className="text-gray-500 ml-0.5">GPU</span>
      {avgRtFactor && (
        <span className="text-base font-extrabold text-mvp-blue-light ml-2">
          avg {avgRtFactor}x Realtime
        </span>
      )}
    </div>
  );
}
