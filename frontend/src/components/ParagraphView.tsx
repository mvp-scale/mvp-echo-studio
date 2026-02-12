import { useState, useRef, useEffect } from "react";
import type { Paragraph } from "../types";
import { ENTITY_TYPE_CONFIG, SENTIMENT_CONFIG } from "../types";
import { speakerColor, speakerName } from "./SpeakerBadge";
import { renderText } from "../utils/highlight-text";

interface Props {
  paragraphs: Paragraph[];
  currentTime: number;
  searchQuery: string;
  onClickTimestamp: (time: number) => void;
  showEntities: boolean;
  visibleEntityTypes: Record<string, boolean>;
  showSentiment: boolean;
  visibleSentimentTypes: Record<string, boolean>;
  onRenameSpeaker?: (originalSpeaker: string, newName: string) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 100);
  return `${m}:${String(s).padStart(2, "0")}.${String(ms).padStart(2, "0")}`;
}

function InlineSpeakerName({
  speaker,
  originalSpeaker,
  color,
  onRenameSpeaker,
}: {
  speaker: string;
  originalSpeaker?: string;
  color: string;
  onRenameSpeaker?: (originalSpeaker: string, newName: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  const startEditing = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!originalSpeaker || !onRenameSpeaker) return;
    setDraft(speakerName(speaker));
    setEditing(true);
  };

  const commitRename = () => {
    setEditing(false);
    if (!originalSpeaker || !onRenameSpeaker) return;
    const trimmed = draft.trim();
    onRenameSpeaker(originalSpeaker, trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      commitRename();
    } else if (e.key === "Escape") {
      setEditing(false);
    }
  };

  if (editing) {
    return (
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
    );
  }

  return (
    <span
      className="group/speaker inline-flex items-center gap-1 text-xs font-bold"
      style={{ color }}
    >
      {speakerName(speaker)}
      {onRenameSpeaker && originalSpeaker && (
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
  );
}

export default function ParagraphView({
  paragraphs,
  currentTime,
  searchQuery,
  onClickTimestamp,
  showEntities,
  visibleEntityTypes,
  showSentiment,
  visibleSentimentTypes,
  onRenameSpeaker,
}: Props) {
  if (!paragraphs.length) return null;

  return (
    <div className="space-y-4 p-4">
      {paragraphs.map((para, i) => {
        const isActive = currentTime >= para.start && currentTime <= para.end;
        const color = para.speaker ? speakerColor(para.speaker, para.originalSpeaker) : "var(--border)";

        // Filter by search
        if (
          searchQuery &&
          !para.text.toLowerCase().includes(searchQuery.toLowerCase())
        ) {
          return null;
        }

        return (
          <div
            key={i}
            className={`p-3.5 rounded-lg border-l-[3px] cursor-pointer transition-colors
              ${isActive ? "bg-mvp-blue/5" : "bg-surface-2 hover:bg-surface-3"}`}
            style={{ borderLeftColor: color }}
            onClick={() => onClickTimestamp(para.start)}
          >
            <div className="flex items-center gap-2.5 mb-1.5 flex-wrap">
              {para.speaker && (
                <InlineSpeakerName
                  speaker={para.speaker}
                  originalSpeaker={para.originalSpeaker}
                  color={color}
                  onRenameSpeaker={onRenameSpeaker}
                />
              )}
              <span className="text-[10px] font-mono text-gray-500">
                {formatTime(para.start)} &rarr; {formatTime(para.end)}
              </span>
              {showEntities && para.entity_counts && (() => {
                const pills = ENTITY_TYPE_CONFIG
                  .filter((t) => visibleEntityTypes[t.key] && para.entity_counts![t.key])
                  .map((t) => (
                    <span
                      key={t.key}
                      className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium ${t.colorClass}`}
                    >
                      {para.entity_counts![t.key]} {t.label}
                    </span>
                  ));
                return pills.length > 0 ? (
                  <div className="flex items-center gap-1.5 ml-2 px-2 py-0.5 rounded-md bg-white/[0.03] border border-white/[0.04]">
                    <span className="text-[9px] uppercase tracking-wider text-gray-600 font-semibold">Entities</span>
                    {pills}
                  </div>
                ) : null;
              })()}
              <div className="ml-auto" />
              {showSentiment && para.sentiment && (() => {
                const cfg = SENTIMENT_CONFIG[para.sentiment.label];
                if (!cfg) return null;
                if (!(visibleSentimentTypes[para.sentiment.label] ?? true)) return null;
                return (
                  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-white/[0.03] border border-white/[0.04]">
                    <span className="text-[9px] uppercase tracking-wider text-gray-600 font-semibold">Sentiment</span>
                    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${cfg.colorClass}`}>
                      {cfg.label}
                    </span>
                  </div>
                );
              })()}
            </div>
            <div className="text-[13px] leading-relaxed text-gray-200">
              {renderText(para.text, searchQuery)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
