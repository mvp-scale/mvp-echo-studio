import { useState, useRef, useEffect } from "react";
import type { Segment } from "../types";
import { speakerColor, speakerName } from "./SpeakerBadge";
import { renderText } from "../utils/highlight-text";

interface Props {
  segment: Segment;
  isActive?: boolean;
  searchQuery?: string;
  onClickTimestamp?: (time: number) => void;
  onRenameSpeaker?: (originalSpeaker: string, newName: string) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 100);
  return `${m}:${String(s).padStart(2, "0")}.${String(ms).padStart(2, "0")}`;
}

export default function SpeakerSegment({
  segment,
  isActive,
  searchQuery,
  onClickTimestamp,
  onRenameSpeaker,
}: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const color = segment.speaker
    ? speakerColor(segment.speaker, segment.originalSpeaker)
    : "var(--border)";

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  const startEditing = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!segment.originalSpeaker || !onRenameSpeaker) return;
    setDraft(speakerName(segment.speaker ?? ""));
    setEditing(true);
  };

  const commitRename = () => {
    setEditing(false);
    if (!segment.originalSpeaker || !onRenameSpeaker) return;
    const trimmed = draft.trim();
    onRenameSpeaker(segment.originalSpeaker, trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      commitRename();
    } else if (e.key === "Escape") {
      setEditing(false);
    }
  };

  return (
    <div
      className={`flex items-baseline gap-3 px-3.5 py-1.5 border-l-[3px] cursor-pointer transition-colors
        ${isActive ? "bg-mvp-blue/5" : "hover:bg-surface-2"}`}
      style={{ borderLeftColor: color }}
      onClick={() => onClickTimestamp?.(segment.start)}
    >
      {/* Speaker + timestamp (fixed width for alignment) */}
      <div className="flex items-baseline gap-2 shrink-0 w-[200px]">
        {segment.speaker && (
          editing ? (
            <input
              ref={inputRef}
              type="text"
              className="text-xs font-bold bg-surface-3 border border-mvp-blue rounded px-1 py-0.5 w-24 outline-none"
              style={{ color }}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commitRename}
              onKeyDown={handleKeyDown}
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <span
              className="group/speaker inline-flex items-center gap-1 text-xs font-bold truncate"
              style={{ color }}
            >
              {speakerName(segment.speaker)}
              {onRenameSpeaker && segment.originalSpeaker && (
                <button
                  className="opacity-0 group-hover/speaker:opacity-60 hover:!opacity-100 transition-opacity"
                  onClick={startEditing}
                  title="Rename speaker"
                >
                  <svg className="w-2.5 h-2.5" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M12.146.854a.5.5 0 0 1 .708 0l2.292 2.292a.5.5 0 0 1 0 .708l-9.5 9.5a.5.5 0 0 1-.168.11l-3.5 1.5a.5.5 0 0 1-.65-.65l1.5-3.5a.5.5 0 0 1 .11-.168l9.5-9.5zM11.207 2.5 13.5 4.793 14.793 3.5 12.5 1.207 11.207 2.5zm-1 1L3 10.707V11h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.293l7.207-7.207-2.5-2.5z"/>
                  </svg>
                </button>
              )}
            </span>
          )
        )}
        <span className="text-[10px] font-mono text-gray-500 whitespace-nowrap">
          {formatTime(segment.start)} &rarr; {formatTime(segment.end)}
        </span>
      </div>

      {/* Text */}
      <span className="text-[13px] leading-relaxed text-gray-200 min-w-0">
        {renderText(segment.text, searchQuery)}
      </span>
    </div>
  );
}
