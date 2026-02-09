import { useState, useCallback, useMemo, useEffect } from "react";
import type {
  Segment,
  Paragraph,
  Statistics,
  DetectedTopic,
  AppState,
  TranscriptionResponse,
  TranscriptionOptions,
  TextRule,
  TextRuleCategory,
  ViewMode,
} from "./types";
import { DEFAULT_OPTIONS } from "./types";
import { DEFAULT_TEXT_RULES } from "./utils/text-rule-presets";
import { transcribe, annotateEntities, annotateSentiment } from "./api";
import { applyPostProcessing } from "./utils/post-processing";
import Layout from "./components/Layout";
import UploadZone from "./components/UploadZone";
import ProgressBar from "./components/ProgressBar";
import TranscriptViewer from "./components/TranscriptViewer";
import ParagraphView from "./components/ParagraphView";
import AudioPlayer from "./components/AudioPlayer";
import SearchBar from "./components/SearchBar";
import ExportBar from "./components/ExportBar";
import SettingsPanel from "./components/SettingsPanel";
import CodeSamplePanel from "./components/CodeSamplePanel";
import SpeakerTimeline from "./components/SpeakerTimeline";
import SpeakerStats from "./components/SpeakerStats";
type LeftTab = "features" | "code";

export default function App() {
  const [state, setState] = useState<AppState>("idle");
  const [file, setFile] = useState<File | null>(null);
  // Raw data from API (never mutated by client-side transforms)
  const [rawSegments, setRawSegments] = useState<Segment[]>([]);
  const [rawParagraphs, setRawParagraphs] = useState<Paragraph[]>([]);
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [topics, setTopics] = useState<DetectedTopic[]>([]);
  const [, setFullText] = useState("");
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const [seekTo, setSeekTo] = useState<number | undefined>(undefined);
  const [searchQuery, setSearchQuery] = useState("");
  const [options, setOptions] = useState<TranscriptionOptions>(DEFAULT_OPTIONS);
  const [viewMode, setViewMode] = useState<ViewMode>("para");
  const [leftTab, setLeftTab] = useState<LeftTab>("features");

  // Restore text rules + category from localStorage on mount (or load defaults)
  // Storage version: bump to invalidate stale data when defaults change
  const STORAGE_KEY = "echo-studio-text-rules-v3";
  const CATEGORY_KEY = "echo-studio-text-rule-category";
  useEffect(() => {
    // Clear stale keys from previous versions
    localStorage.removeItem("echo-studio-text-rules");
    localStorage.removeItem("echo-studio-text-rules-v2");
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const rules: TextRule[] = JSON.parse(saved);
        if (Array.isArray(rules) && rules.length > 0) {
          const cat = (localStorage.getItem(CATEGORY_KEY) ?? "all") as TextRuleCategory;
          setOptions((prev) => ({ ...prev, textRules: rules, textRuleCategory: cat }));
          return;
        }
      }
    } catch {
      // Fall through to defaults
    }
    // No saved rules — load built-in defaults
    setOptions((prev) => ({ ...prev, textRules: DEFAULT_TEXT_RULES }));
  }, []);

  // Persist text rules + category to localStorage on change
  useEffect(() => {
    if (options.textRules.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(options.textRules));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
    localStorage.setItem(CATEGORY_KEY, options.textRuleCategory);
  }, [options.textRules, options.textRuleCategory]);

  // Client-side post-processing: recomputes instantly when options change
  const { segments, paragraphs } = useMemo(
    () => applyPostProcessing(rawSegments, rawParagraphs, options),
    [rawSegments, rawParagraphs, options]
  );

  // Extract unique speaker IDs from RAW segments (before label rename)
  const detectedSpeakers = useMemo(() => {
    const speakers = new Set<string>();
    for (const seg of rawSegments) {
      if (seg.speaker && seg.speaker !== "unknown") {
        speakers.add(seg.speaker);
      }
    }
    return Array.from(speakers).sort();
  }, [rawSegments]);

  // Post-hoc entity detection: when toggled on after transcription, annotate existing paragraphs
  useEffect(() => {
    if (!options.detectEntities || state !== "done" || rawParagraphs.length === 0) return;
    // Already annotated — at least one paragraph has entity_counts
    if (rawParagraphs.some((p) => p.entity_counts)) return;

    let cancelled = false;
    annotateEntities(rawParagraphs.map((p) => ({ text: p.text }))).then((counts) => {
      if (cancelled) return;
      setRawParagraphs((prev) =>
        prev.map((p, i) => ({ ...p, entity_counts: counts[i] ?? undefined }))
      );
    }).catch((err) => {
      console.error("Entity annotation failed:", err);
    });
    return () => { cancelled = true; };
  }, [options.detectEntities, state, rawParagraphs]);

  // Post-hoc sentiment analysis: when toggled on after transcription, annotate existing paragraphs
  useEffect(() => {
    if (!options.detectSentiment || state !== "done" || rawParagraphs.length === 0) return;
    if (rawParagraphs.some((p) => p.sentiment)) return;

    let cancelled = false;
    annotateSentiment(rawParagraphs.map((p) => ({ text: p.text }))).then((results) => {
      if (cancelled) return;
      setRawParagraphs((prev) =>
        prev.map((p, i) => ({ ...p, sentiment: results[i] ?? undefined }))
      );
    }).catch((err) => {
      console.error("Sentiment annotation failed:", err);
    });
    return () => { cancelled = true; };
  }, [options.detectSentiment, state, rawParagraphs]);

  // File selection (does NOT auto-start transcription)
  const handleFileSelect = useCallback((f: File) => {
    setFile(f);
    setError("");
    setRawSegments([]);
    setRawParagraphs([]);
    setStatistics(null);
    setTopics([]);
    setFullText("");
    setSearchQuery("");
    setState("idle");
  }, []);

  // Run transcription (triggered by Run button)
  const handleRun = useCallback(async () => {
    if (!file) return;
    setState("transcribing");
    setError("");
    setRawSegments([]);
    setRawParagraphs([]);
    setStatistics(null);
    setTopics([]);
    setFullText("");
    setSearchQuery("");

    try {
      const result: TranscriptionResponse = await transcribe(file, options);
      setRawSegments(result.segments ?? []);
      setRawParagraphs(result.paragraphs ?? []);
      setStatistics(result.statistics ?? null);
      setTopics(result.topics ?? []);
      setFullText(result.text);
      setDuration(result.duration ?? 0);
      setState("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transcription failed");
      setState("error");
    }
  }, [file, options]);

  const handleClickTimestamp = useCallback((time: number) => {
    setSeekTo(time);
  }, []);

  const handleRemoveFile = useCallback(() => {
    setFile(null);
    setState("idle");
    setRawSegments([]);
    setRawParagraphs([]);
    setStatistics(null);
    setTopics([]);
    setFullText("");
    setError("");
    setSearchQuery("");
    setSeekTo(undefined);
  }, []);

  const filteredCount = searchQuery
    ? segments.filter((s) =>
        s.text.toLowerCase().includes(searchQuery.toLowerCase())
      ).length
    : undefined;

  const isProcessing = state === "transcribing";
  const isDone = state === "done" && rawSegments.length > 0;

  return (
    <Layout>
      {/* Upload area - always at top */}
      <div className="border-b border-border bg-gradient-to-br from-[#1a1040] via-surface-0 to-[#101820] px-8 py-6">
        <div className="max-w-[1440px] mx-auto">
          {!file ? (
            <div className="max-w-xl mx-auto">
              <h2 className="text-lg font-semibold text-center mb-5">Audio Input</h2>
              <UploadZone onFile={handleFileSelect} />
            </div>
          ) : (
            <div className="flex items-center gap-3 max-w-xl mx-auto px-4 py-2.5 bg-surface-3 rounded-lg">
              <span className="text-base">&#127925;</span>
              <span className="flex-1 text-sm font-medium text-gray-200 truncate">
                {file.name}
              </span>
              <span className="text-xs text-gray-500">
                {(file.size / 1048576).toFixed(1)} MB
              </span>
              <button
                className="text-gray-500 hover:text-red-400 text-lg leading-none px-1"
                onClick={handleRemoveFile}
              >
                &times;
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex items-center px-8 py-2.5 border-b border-border gap-3">
        {/* Left panel tab switcher */}
        <div className="flex gap-0.5 bg-surface-2 rounded-lg p-0.5">
          <button
            className={`px-3.5 py-1 text-xs font-medium rounded-md transition-colors ${
              leftTab === "features"
                ? "bg-surface-3 text-white"
                : "text-gray-400 hover:text-white"
            }`}
            onClick={() => setLeftTab("features")}
          >
            Features
          </button>
          <button
            className={`px-3.5 py-1 text-xs font-medium rounded-md transition-colors ${
              leftTab === "code"
                ? "bg-surface-3 text-white"
                : "text-gray-400 hover:text-white"
            }`}
            onClick={() => setLeftTab("code")}
          >
            Code Sample
          </button>
        </div>

        {/* Run button */}
        <button
          className="inline-flex items-center gap-2 px-6 py-1.5 rounded-lg text-sm font-bold
            transition-all disabled:opacity-40 disabled:cursor-not-allowed
            bg-emerald-500 text-surface-0 hover:brightness-110 ml-auto"
          disabled={!file || isProcessing}
          onClick={handleRun}
        >
          <span>&#9654;</span> Run
        </button>
      </div>

      {/* Main two-panel layout */}
      <div className="flex-1 flex min-h-0">
        {/* Left panel */}
        <div className="w-[380px] shrink-0 border-r border-border overflow-y-auto">
          {leftTab === "features" ? (
            <SettingsPanel
              options={options}
              onChange={setOptions}
              disabled={isProcessing}
              detectedSpeakers={detectedSpeakers}
            />
          ) : (
            <CodeSamplePanel options={options} />
          )}
        </div>

        {/* Right panel */}
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {/* Progress */}
          {isProcessing && (
            <div className="flex-1 flex items-center justify-center">
              <ProgressBar active filename={file?.name} />
            </div>
          )}

          {/* Error */}
          {state === "error" && (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 p-8">
              <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center">
                <svg className="w-6 h-6 text-red-400" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                </svg>
              </div>
              <p className="text-red-400 text-sm">{error}</p>
              <button
                onClick={handleRemoveFile}
                className="px-4 py-2 text-sm bg-surface-2 border border-border rounded-lg text-gray-300 hover:text-white hover:border-border-light"
              >
                Try again
              </button>
            </div>
          )}

          {/* Empty state */}
          {!isProcessing && !isDone && state !== "error" && (
            <div className="flex-1 flex flex-col items-center justify-center text-gray-500 gap-3">
              <div className="text-4xl opacity-25">&#128196;</div>
              <p className="text-sm max-w-[280px] text-center">
                Upload audio and click <strong className="text-gray-300">Run</strong> to see output here.
              </p>
            </div>
          )}

          {/* Results */}
          {isDone && (
            <>
              {/* Audio Player */}
              <AudioPlayer
                file={file}
                onTimeUpdate={setCurrentTime}
                seekTo={seekTo}
              />

              {/* Speaker Timeline (uses raw segments for accurate colors) */}
              <SpeakerTimeline
                segments={rawSegments}
                duration={duration}
                currentTime={currentTime}
                onSeek={handleClickTimestamp}
              />

              {/* Speaker Stats */}
              {statistics && <SpeakerStats statistics={statistics} />}

              {/* Topics */}
              {topics.length > 0 && (
                <div className="px-5 py-3 border-b border-border">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-1.5">
                    Topics ({topics.length})
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {topics.map((t, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-mvp-blue-dim text-mvp-blue-light"
                      >
                        {t.text}
                        <span className="text-[9px] opacity-60">x{t.count}</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Search */}
              <SearchBar
                value={searchQuery}
                onChange={setSearchQuery}
                resultCount={filteredCount}
              />

              {/* View toggle + export */}
              <div className="flex items-center gap-2 px-5 py-2.5 border-b border-border">
                <div className="flex gap-0.5 bg-surface-2 rounded-lg p-0.5">
                  <button
                    className={`px-3 py-1 text-[11px] font-medium rounded-md ${
                      viewMode === "line"
                        ? "bg-surface-3 text-white"
                        : "text-gray-400 hover:text-white"
                    }`}
                    onClick={() => setViewMode("line")}
                  >
                    Line by Line
                  </button>
                  <button
                    className={`px-3 py-1 text-[11px] font-medium rounded-md ${
                      viewMode === "para"
                        ? "bg-surface-3 text-white"
                        : "text-gray-400 hover:text-white"
                    }`}
                    onClick={() => setViewMode("para")}
                  >
                    Paragraph
                  </button>
                </div>
                <div className="flex-1" />
                <ExportBar segments={segments} paragraphs={paragraphs} viewMode={viewMode} filename={file?.name ?? "transcript"} />
              </div>

              {/* Transcript */}
              <div className="flex-1 overflow-y-auto">
                {viewMode === "line" ? (
                  <TranscriptViewer
                    segments={segments}
                    currentTime={currentTime}
                    searchQuery={searchQuery}
                    onClickTimestamp={handleClickTimestamp}
                  />
                ) : (
                  <ParagraphView
                    paragraphs={paragraphs}
                    currentTime={currentTime}
                    searchQuery={searchQuery}
                    onClickTimestamp={handleClickTimestamp}
                    visibleEntityTypes={options.visibleEntityTypes}
                    showSentiment={options.detectSentiment}
                    visibleSentimentTypes={options.visibleSentimentTypes}
                  />
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </Layout>
  );
}
