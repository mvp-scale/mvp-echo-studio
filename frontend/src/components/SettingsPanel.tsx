import { useRef } from "react";
import { type TranscriptionOptions, type TextRule, type TextRuleset, type TextRuleCategory, ENTITY_TYPE_CONFIG, SENTIMENT_TYPE_CONFIG } from "../types";
import { DEFAULT_TEXT_RULES } from "../utils/text-rule-presets";

interface Props {
  options: TranscriptionOptions;
  onChange: (options: TranscriptionOptions) => void;
  disabled?: boolean;
  detectedSpeakers?: string[];
}

function Toggle({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="relative inline-flex w-9 h-5 shrink-0 cursor-pointer">
      <input
        type="checkbox"
        className="sr-only peer"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
      />
      <span
        className="absolute inset-0 rounded-full border transition-all
          bg-surface-3 border-border peer-checked:bg-mvp-blue-dim peer-checked:border-mvp-blue
          peer-disabled:opacity-30 peer-disabled:cursor-not-allowed"
      />
      <span
        className="absolute top-[3px] left-[3px] w-3.5 h-3.5 rounded-full transition-all
          bg-gray-500 peer-checked:translate-x-4 peer-checked:bg-mvp-blue"
      />
    </label>
  );
}

function NumberInput({
  value,
  onChange,
  placeholder,
  min = 1,
  max = 20,
}: {
  value?: number;
  onChange: (v?: number) => void;
  placeholder: string;
  min?: number;
  max?: number;
}) {
  return (
    <input
      type="number"
      className="w-14 px-1.5 py-0.5 bg-surface-3 border border-border rounded text-xs
        text-gray-200 placeholder-gray-600 focus:border-mvp-blue focus:outline-none"
      placeholder={placeholder}
      min={min}
      max={max}
      value={value ?? ""}
      onChange={(e) => {
        const v = e.target.value;
        onChange(v === "" ? undefined : parseInt(v, 10));
      }}
    />
  );
}

function parseRules(text: string): TextRule[] | null {
  try {
    const parsed = JSON.parse(text);
    // Accept bare array or envelope with rules array
    const rules: unknown[] = Array.isArray(parsed) ? parsed : parsed?.rules;
    if (!Array.isArray(rules)) return null;
    return rules.map((r: any) => ({
      name: r.name ?? "",
      find: r.find ?? "",
      replace: r.replace ?? "",
      isRegex: r.isRegex ?? false,
      flags: r.flags ?? "gi",
      category: r.category ?? "replace",
      enabled: r.enabled ?? true,
    }));
  } catch {
    return null;
  }
}

export default function SettingsPanel({ options, onChange, disabled, detectedSpeakers = [] }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const set = <K extends keyof TranscriptionOptions>(
    key: K,
    value: TranscriptionOptions[K]
  ) => {
    onChange({ ...options, [key]: value });
  };

  const handleReset = () => {
    set("textRules", DEFAULT_TEXT_RULES.map((r) => ({ ...r })));
    set("textRuleCategory", "all");
  };

  const handleImport = () => fileInputRef.current?.click();

  const handleFileImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const rules = parseRules(reader.result as string);
      if (rules) set("textRules", rules);
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const handleExport = () => {
    const ruleset: TextRuleset = {
      version: 1,
      name: "Echo Studio Text Rules",
      rules: options.textRules,
    };
    const blob = new Blob([JSON.stringify(ruleset, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "text-rules.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const totalCount = options.textRules.length;
  const fillerCount = options.textRules.filter((r) => r.category === "filler").length;
  const piiCount = options.textRules.filter((r) => r.category === "pii").length;
  const replaceCount = options.textRules.filter((r) => r.category === "replace").length;

  const activeCategory = options.textRuleCategory;
  const activeCount = activeCategory === "all"
    ? totalCount
    : options.textRules.filter((r) => r.category === activeCategory).length;

  const categories: { key: TextRuleCategory; label: string; count: number }[] = [
    { key: "all", label: "All", count: totalCount },
    { key: "filler", label: "Fillers", count: fillerCount },
    { key: "pii", label: "PII", count: piiCount },
    { key: "replace", label: "Replace", count: replaceCount },
  ];

  const statusLabel = activeCategory === "all"
    ? `${activeCount} rules applied`
    : activeCategory === "filler"
    ? `${activeCount} filler rules applied`
    : activeCategory === "pii"
    ? `${activeCount} PII rules applied`
    : `${activeCount} replace rules applied`;

  return (
    <div className="p-5 space-y-6 text-sm">
      {/* Transcription */}
      <div>
        <div className="text-[11px] font-bold uppercase tracking-wider text-gray-500 mb-3 pb-1.5 border-b border-border">
          Transcription
        </div>

        {/* Diarization */}
        <div className="py-2.5">
          <div className="flex items-start gap-2.5">
            <Toggle
              checked={options.diarize}
              onChange={(v) => set("diarize", v)}
              disabled={disabled}
            />
            <div className="flex-1">
              <div className="text-[13px] font-semibold">Diarization</div>
              <div className="text-[10px] font-mono text-mvp-blue-light mt-0.5">
                diarize={String(options.diarize)}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">
                Identify and label speakers.
              </div>
            </div>
          </div>
          {options.diarize && (
            <div className="mt-2 pl-[46px] space-y-1.5">
              <div className="flex items-center gap-1.5">
                <label className="text-[11px] text-gray-400 whitespace-nowrap">
                  Min speakers:
                </label>
                <NumberInput
                  value={options.minSpeakers}
                  onChange={(v) => set("minSpeakers", v)}
                  placeholder="Auto"
                />
              </div>
              <div className="flex items-center gap-1.5">
                <label className="text-[11px] text-gray-400 whitespace-nowrap">
                  Max speakers:
                </label>
                <NumberInput
                  value={options.maxSpeakers}
                  onChange={(v) => set("maxSpeakers", v)}
                  placeholder="Auto"
                />
              </div>
              <div className="flex items-center gap-1.5">
                <label className="text-[11px] text-gray-400 whitespace-nowrap">
                  Exact count:
                </label>
                <NumberInput
                  value={options.numSpeakers}
                  onChange={(v) => set("numSpeakers", v)}
                  placeholder="Auto"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Post-Processing */}
      <div>
        <div className="text-[11px] font-bold uppercase tracking-wider text-gray-500 mb-3 pb-1.5 border-b border-border">
          Post-Processing
        </div>

        {/* Detect Paragraphs */}
        <div className="py-2.5">
          <div className="flex items-start gap-2.5">
            <Toggle
              checked={options.detectParagraphs}
              onChange={(v) => set("detectParagraphs", v)}
              disabled={disabled}
            />
            <div className="flex-1">
              <div className="text-[13px] font-semibold">Detect Paragraphs</div>
              <div className="text-[10px] font-mono text-mvp-blue-light mt-0.5">
                detect_paragraphs={String(options.detectParagraphs)}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">
                Group segments by speaker turn.
              </div>
            </div>
          </div>
          {options.detectParagraphs && (
            <div className="mt-2 pl-[46px]">
              <div className="flex items-center gap-1.5">
                <label className="text-[11px] text-gray-400 whitespace-nowrap">
                  Silence gap:
                </label>
                <input
                  type="range"
                  min="0.2"
                  max="5"
                  step="0.1"
                  value={options.paragraphSilenceThreshold}
                  onChange={(e) =>
                    set("paragraphSilenceThreshold", parseFloat(e.target.value))
                  }
                  className="flex-1 accent-mvp-blue"
                />
                <span className="text-[10px] font-mono text-gray-200 min-w-[32px] text-right">
                  {options.paragraphSilenceThreshold.toFixed(1)}s
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Confidence Filter */}
        <div className="py-2.5">
          <div className="flex items-start gap-2.5">
            <Toggle
              checked={options.minConfidence > 0}
              onChange={(v) => set("minConfidence", v ? 0.5 : 0)}
              disabled={disabled}
            />
            <div className="flex-1">
              <div className="text-[13px] font-semibold">Confidence Filter</div>
              <div className="text-[10px] font-mono text-mvp-blue-light mt-0.5">
                min_confidence={options.minConfidence.toFixed(2)}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">
                Filter low-confidence segments.
              </div>
            </div>
          </div>
          {options.minConfidence > 0 && (
            <div className="mt-2 pl-[46px]">
              <div className="flex items-center gap-1.5">
                <label className="text-[11px] text-gray-400 whitespace-nowrap">
                  Threshold:
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.05"
                  value={options.minConfidence}
                  onChange={(e) =>
                    set("minConfidence", parseFloat(e.target.value))
                  }
                  className="flex-1 accent-mvp-blue"
                />
                <span className="text-[10px] font-mono text-gray-200 min-w-[32px] text-right">
                  {options.minConfidence.toFixed(2)}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Text Rules */}
        <div className="py-2.5">
          <div className="flex items-start gap-2.5">
            <Toggle
              checked={options.textRulesEnabled}
              onChange={(v) => set("textRulesEnabled", v)}
              disabled={disabled}
            />
            <div className="flex-1">
              <div className="text-[13px] font-semibold">Text Rules</div>
              <div className="text-[10px] font-mono text-mvp-blue-light mt-0.5">
                text_rules={options.textRulesEnabled ? `${activeCount} active` : "off"}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">
                Filler removal, PII redaction, text replacements.
              </div>
            </div>
          </div>
          {options.textRulesEnabled && (
            <div className="mt-2 pl-[46px] space-y-2">
              {/* Category pills */}
              <div className="flex gap-1">
                {categories.map((cat) => (
                  <button
                    key={cat.key}
                    className={`px-2.5 py-1 text-[11px] rounded-full border transition-colors ${
                      activeCategory === cat.key
                        ? "bg-mvp-blue-dim border-mvp-blue text-mvp-blue-light"
                        : "bg-surface-3 border-border text-gray-500 hover:text-gray-300"
                    }`}
                    onClick={() => set("textRuleCategory", cat.key)}
                  >
                    {cat.label}{cat.count > 0 ? ` (${cat.count})` : ""}
                  </button>
                ))}
              </div>

              {/* Status message */}
              {totalCount > 0 && (
                <div className="text-[11px] text-gray-500">
                  {statusLabel}
                </div>
              )}

              {/* Import / Export / Defaults */}
              <div className="flex gap-1.5 flex-wrap">
                <button
                  className="px-2.5 py-1 text-[11px] bg-surface-3 border border-border rounded
                    text-gray-400 hover:text-white hover:border-mvp-blue"
                  onClick={handleImport}
                >
                  Import JSON
                </button>
                <button
                  className="px-2.5 py-1 text-[11px] bg-surface-3 border border-border rounded
                    text-gray-400 hover:text-white hover:border-mvp-blue"
                  onClick={handleExport}
                  disabled={totalCount === 0}
                >
                  Export JSON
                </button>
                <button
                  className="px-2.5 py-1 text-[11px] bg-surface-3 border border-border rounded
                    text-gray-400 hover:text-white hover:border-mvp-blue"
                  onClick={handleReset}
                >
                  Defaults
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={handleFileImport}
                />
              </div>

              {totalCount === 0 && (
                <div className="text-[11px] text-gray-600">
                  No rules loaded. Import a JSON ruleset to get started.
                </div>
              )}
            </div>
          )}
        </div>

        {/* Custom Speaker Labels - auto-enabled when speakers detected */}
        {detectedSpeakers.length > 1 && (
          <div className="py-2.5">
            <div className="text-[13px] font-semibold">
              Speaker Labels
              <span className="text-[10px] font-normal text-gray-500 ml-1">
                ({detectedSpeakers.length} detected)
              </span>
            </div>
            <div className="text-[11px] text-gray-500 mt-0.5 mb-2">
              Rename speakers in output and exports.
            </div>
            <div className="space-y-1.5">
              {detectedSpeakers.map((spk, i) => {
                const color = `var(--speaker-${i % 8})`;
                return (
                  <div key={spk} className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ background: color }}
                    />
                    <span className="text-[10px] text-gray-500 whitespace-nowrap w-[18px]">
                      {i + 1}
                    </span>
                    <input
                      type="text"
                      className="flex-1 px-1.5 py-0.5 bg-surface-3 border border-border rounded text-[11px]
                        text-gray-200 placeholder-gray-600 focus:border-mvp-blue focus:outline-none"
                      placeholder={`Speaker ${i + 1}`}
                      value={options.speakerLabels[spk] ?? ""}
                      onChange={(e) => {
                        const labels = { ...options.speakerLabels };
                        if (e.target.value) {
                          labels[spk] = e.target.value;
                        } else {
                          delete labels[spk];
                        }
                        set("speakerLabels", labels);
                      }}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Audio Intelligence */}
      <div>
        <div className="text-[11px] font-bold uppercase tracking-wider text-gray-500 mb-3 pb-1.5 border-b border-border">
          Audio Intelligence
        </div>

        {/* Entity Detection */}
        <div className="py-2.5">
          <div className="flex items-start gap-2.5">
            <Toggle
              checked={options.detectEntities}
              onChange={(v) => set("detectEntities", v)}
              disabled={disabled}
            />
            <div className="flex-1">
              <div className="text-[13px] font-semibold">Entity Detection</div>
              <div className="text-[10px] font-mono text-mvp-blue-light mt-0.5">
                detect_entities={String(options.detectEntities)}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">
                Detect people, organizations, locations, dates.
              </div>
            </div>
          </div>
          {options.detectEntities && (
            <div className="mt-2 pl-[46px]">
              <div className="flex flex-wrap gap-1.5">
                {ENTITY_TYPE_CONFIG.map((t) => {
                  const active = options.visibleEntityTypes[t.key] ?? true;
                  return (
                    <button
                      key={t.key}
                      className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all
                        ${active
                          ? t.colorClass
                          : "bg-surface-3 text-gray-600 border border-border"
                        }`}
                      onClick={() => {
                        set("visibleEntityTypes", {
                          ...options.visibleEntityTypes,
                          [t.key]: !active,
                        });
                      }}
                    >
                      {t.label}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Topic Detection */}
        <div className="py-2.5">
          <div className="flex items-start gap-2.5">
            <Toggle
              checked={options.detectTopics}
              onChange={(v) => set("detectTopics", v)}
              disabled={disabled}
            />
            <div className="flex-1">
              <div className="text-[13px] font-semibold">Topic Detection</div>
              <div className="text-[10px] font-mono text-mvp-blue-light mt-0.5">
                detect_topics={String(options.detectTopics)}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">
                Extract most-discussed subjects.
              </div>
            </div>
          </div>
        </div>

        {/* Sentiment Analysis */}
        <div className="py-2.5">
          <div className="flex items-start gap-2.5">
            <Toggle
              checked={options.detectSentiment}
              onChange={(v) => set("detectSentiment", v)}
              disabled={disabled}
            />
            <div className="flex-1">
              <div className="text-[13px] font-semibold">Sentiment Analysis</div>
              <div className="text-[10px] font-mono text-mvp-blue-light mt-0.5">
                detect_sentiment={String(options.detectSentiment)}
              </div>
              <div className="text-[11px] text-gray-500 mt-0.5">
                Positive, neutral, or negative per paragraph.
              </div>
            </div>
          </div>
          {options.detectSentiment && (
            <div className="mt-2 pl-[46px]">
              <div className="flex flex-wrap gap-1.5">
                {SENTIMENT_TYPE_CONFIG.map((t) => {
                  const active = options.visibleSentimentTypes[t.key] ?? true;
                  return (
                    <button
                      key={t.key}
                      className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all
                        ${active
                          ? t.colorClass
                          : "bg-surface-3 text-gray-600 border border-border"
                        }`}
                      onClick={() => {
                        set("visibleSentimentTypes", {
                          ...options.visibleSentimentTypes,
                          [t.key]: !active,
                        });
                      }}
                    >
                      {t.label}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Future features */}
        {[
          { name: "Summarization", phase: "Phase 3" },
        ].map((feat) => (
          <div key={feat.name} className="py-2">
            <div className="flex items-start gap-2.5">
              <Toggle checked={false} onChange={() => {}} disabled />
              <div className="flex-1">
                <div className="text-[13px] font-semibold text-gray-400">
                  {feat.name}{" "}
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-orange-500/10 text-orange-400 font-semibold ml-1">
                    {feat.phase}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
