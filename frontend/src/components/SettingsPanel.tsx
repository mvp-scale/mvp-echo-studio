import { type TranscriptionOptions, type TextRuleCategory, ENTITY_TYPE_CONFIG, SENTIMENT_TYPE_CONFIG } from "../types";

interface Props {
  options: TranscriptionOptions;
  onChange: (options: TranscriptionOptions) => void;
  disabled?: boolean;
  detectedSpeakers?: string[];
}

/* ── Shared pill class for inactive state (used by all toggleable pills) ── */
const PILL_INACTIVE = "bg-surface-3 text-gray-600 border border-border hover:border-mvp-blue";
const PILL_BASE = "px-2 py-0.5 rounded text-[10px] font-medium transition-all cursor-pointer";

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

/* ── Shared input class ── */
const INPUT_CLASS = `w-12 px-1.5 py-0.5 bg-surface-3 border border-border rounded text-[10px] text-center
  text-gray-200 placeholder-gray-600 hover:border-mvp-blue focus:border-mvp-blue focus:outline-none
  [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none`;

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
      type="text"
      inputMode="numeric"
      className={INPUT_CLASS}
      placeholder={placeholder}
      value={value ?? ""}
      onChange={(e) => {
        const v = e.target.value.trim();
        if (v === "") {
          onChange(undefined);
          return;
        }
        const n = parseInt(v, 10);
        if (!isNaN(n) && n >= min && n <= max) {
          onChange(n);
        }
      }}
    />
  );
}

export default function SettingsPanel({ options, onChange, disabled, detectedSpeakers = [] }: Props) {
  const set = <K extends keyof TranscriptionOptions>(
    key: K,
    value: TranscriptionOptions[K]
  ) => {
    onChange({ ...options, [key]: value });
  };

  const totalCount = options.textRules.length;
  const fillerCount = options.textRules.filter((r) => r.category === "filler").length;
  const piiCount = options.textRules.filter((r) => r.category === "pii").length;
  const replaceCount = options.textRules.filter((r) => r.category === "replace").length;

  const activeCategory = options.textRuleCategory;

  const categories: { key: TextRuleCategory; label: string; count: number; colorClass: string }[] = [
    { key: "all", label: "All", count: totalCount, colorClass: "bg-mvp-blue/15 text-mvp-blue-light" },
    { key: "filler", label: "Fillers", count: fillerCount, colorClass: "bg-orange-500/15 text-orange-400" },
    { key: "pii", label: "PII", count: piiCount, colorClass: "bg-red-500/15 text-red-400" },
    { key: "replace", label: "Replace", count: replaceCount, colorClass: "bg-cyan-500/15 text-cyan-400" },
  ];

  return (
    <div className="p-4 space-y-4 text-sm">
      {/* Transcription */}
      <div>
        <div className="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-2 pb-1 border-b border-border">
          Transcription
        </div>

        {/* Diarization */}
        <div className="py-1.5">
          <div className="flex items-center gap-2.5">
            <Toggle
              checked={options.diarize}
              onChange={(v) => set("diarize", v)}
              disabled={disabled}
            />
            <div className="text-[12px] font-semibold">Diarization</div>
          </div>
          {options.diarize && (
            <div className="mt-1.5 pl-[46px] flex items-center gap-2 flex-wrap">
              <label className="flex items-center gap-1 text-[10px] text-gray-400">
                Min <NumberInput value={options.minSpeakers} onChange={(v) => set("minSpeakers", v)} placeholder="Auto" />
              </label>
              <label className="flex items-center gap-1 text-[10px] text-gray-400">
                Max <NumberInput value={options.maxSpeakers} onChange={(v) => set("maxSpeakers", v)} placeholder="Auto" />
              </label>
              <label className="flex items-center gap-1 text-[10px] text-gray-400">
                Exact <NumberInput value={options.numSpeakers} onChange={(v) => set("numSpeakers", v)} placeholder="Auto" />
              </label>
            </div>
          )}
        </div>
      </div>

      {/* Post-Processing */}
      <div>
        <div className="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-2 pb-1 border-b border-border">
          Post-Processing
        </div>

        {/* Detect Paragraphs */}
        <div className="py-1.5">
          <div className="flex items-center gap-2.5">
            <Toggle
              checked={options.detectParagraphs}
              onChange={(v) => set("detectParagraphs", v)}
              disabled={disabled}
            />
            <div className="text-[12px] font-semibold">Paragraphs</div>
          </div>
          {options.detectParagraphs && (
            <div className="mt-1.5 pl-[46px] flex items-center gap-1.5">
              <span className="text-[10px] text-gray-400">Gap</span>
              <input
                type="range"
                min="0.2"
                max="5"
                step="0.1"
                value={options.paragraphSilenceThreshold}
                onChange={(e) =>
                  set("paragraphSilenceThreshold", parseFloat(e.target.value))
                }
                className="flex-1 accent-mvp-blue h-1"
              />
              <span className="text-[10px] font-mono text-gray-300 min-w-[28px] text-right">
                {options.paragraphSilenceThreshold.toFixed(1)}s
              </span>
            </div>
          )}
        </div>

        {/* Smart Redaction & PII */}
        <div className="py-1.5">
          <div className="flex items-center gap-2.5">
            <Toggle
              checked={options.textRulesEnabled}
              onChange={(v) => set("textRulesEnabled", v)}
              disabled={disabled}
            />
            <div className="text-[12px] font-semibold flex-1">Smart Redaction & PII</div>
          </div>
          {options.textRulesEnabled && (
            <div className="mt-1.5 pl-[46px]">
              <div className="flex flex-wrap gap-1">
                {categories.map((cat) => {
                  const active = activeCategory === cat.key;
                  return (
                    <button
                      key={cat.key}
                      className={`${PILL_BASE}
                        ${active ? cat.colorClass : PILL_INACTIVE}`}
                      onClick={() => set("textRuleCategory", cat.key)}
                    >
                      {cat.label}{cat.count > 0 ? ` (${cat.count})` : ""}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Speaker Labels - auto-enabled when speakers detected */}
        {detectedSpeakers.length > 1 && (
          <div className="py-1.5">
            <div className="flex items-center gap-2.5">
              <Toggle checked={true} onChange={() => {}} disabled />
              <div className="text-[12px] font-semibold">
                Speaker Labels
                <span className="text-[10px] font-normal text-gray-500 ml-1.5">
                  {detectedSpeakers.length} detected
                </span>
              </div>
            </div>
            <div className="mt-1.5 pl-[46px] space-y-1">
              {detectedSpeakers.map((spk, i) => {
                const color = `var(--speaker-${i % 8})`;
                return (
                  <div key={spk} className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ background: color }}
                    />
                    <input
                      type="text"
                      className={`flex-1 px-1.5 py-0.5 bg-surface-3 border border-border rounded text-[10px]
                        text-gray-200 placeholder-gray-600 hover:border-mvp-blue focus:border-mvp-blue focus:outline-none`}
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
        <div className="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-2 pb-1 border-b border-border">
          Audio Intelligence
        </div>

        {/* Entity Detection */}
        <div className="py-1.5">
          <div className="flex items-center gap-2.5">
            <Toggle
              checked={options.detectEntities}
              onChange={(v) => set("detectEntities", v)}
              disabled={disabled}
            />
            <div className="flex-1">
              <div className="text-[12px] font-semibold">Entity Detection</div>
              <div className="text-[10px] text-gray-500 mt-0.5">
                People, organizations, locations, dates.
              </div>
            </div>
          </div>
          {options.detectEntities && (
            <div className="mt-1.5 pl-[46px]">
              <div className="flex flex-wrap gap-1">
                {ENTITY_TYPE_CONFIG.map((t) => {
                  const active = options.visibleEntityTypes[t.key] ?? true;
                  return (
                    <button
                      key={t.key}
                      className={`${PILL_BASE}
                        ${active ? t.colorClass : PILL_INACTIVE}`}
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

        {/* Sentiment Analysis */}
        <div className="py-1.5">
          <div className="flex items-center gap-2.5">
            <Toggle
              checked={options.detectSentiment}
              onChange={(v) => set("detectSentiment", v)}
              disabled={disabled}
            />
            <div className="flex-1">
              <div className="text-[12px] font-semibold">Sentiment Analysis</div>
              <div className="text-[10px] text-gray-500 mt-0.5">
                Positive, neutral, or negative per paragraph.
              </div>
            </div>
          </div>
          {options.detectSentiment && (
            <div className="mt-1.5 pl-[46px]">
              <div className="flex flex-wrap gap-1">
                {SENTIMENT_TYPE_CONFIG.map((t) => {
                  const active = options.visibleSentimentTypes[t.key] ?? true;
                  return (
                    <button
                      key={t.key}
                      className={`${PILL_BASE}
                        ${active ? t.colorClass : PILL_INACTIVE}`}
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

      </div>

      {/* Coming Soon */}
      <div>
        <div className="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-2 pb-1 border-b border-border">
          Coming Soon
        </div>

        {[
          { name: "Real-time Streaming", desc: "Live transcription via WebSocket" },
          { name: "Multi-language", desc: "100+ languages (Whisper models)" },
          { name: "Custom Vocabulary", desc: "Boost domain-specific terms" },
          { name: "Audio Enhancement", desc: "Reduce background noise" },
          { name: "Word-level Timestamps", desc: "Finer granularity than segments" },
          { name: "Translation", desc: "Convert transcript to any language" },
          { name: "Chapter Detection", desc: "Auto-title sections of long recordings" },
          { name: "Topic Detection", desc: "Extract recurring themes" },
          { name: "Summarization", desc: "AI-powered condensed version" },
          { name: "Action Items", desc: "Extract tasks and decisions" },
          { name: "Key Moments", desc: "Auto-highlight important sections" },
          { name: "Speaker Identification", desc: "Voice-based speaker matching" },
          { name: "Emotion Detection", desc: "Detect excitement, frustration, confusion" },
          { name: "Meeting Notes", desc: "Auto-generate structured summary" },
        ].map((feat) => (
          <div key={feat.name} className="py-1.5">
            <div className="flex items-center gap-2.5">
              <Toggle checked={false} onChange={() => {}} disabled />
              <div className="flex-1">
                <div className="text-[12px] font-semibold text-gray-400">
                  {feat.name}
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-orange-500/10 text-orange-400 font-semibold ml-1.5">
                    Phase 3
                  </span>
                </div>
                <div className="text-[10px] text-gray-500 mt-0.5">{feat.desc}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
