export interface Segment {
  id: number;
  start: number;
  end: number;
  text: string;
  speaker?: string;
  originalSpeaker?: string;
  no_speech_prob?: number;
}

export interface SentimentResult {
  label: "positive" | "neutral" | "negative";
  score: number;
}

export interface Paragraph {
  speaker?: string;
  originalSpeaker?: string;
  start: number;
  end: number;
  text: string;
  segment_count: number;
  entity_counts?: Record<string, number>;
  sentiment?: SentimentResult;
}

export interface SpeakerStatistics {
  duration: number;
  percentage: number;
  word_count: number;
}

export interface Statistics {
  speakers: Record<string, SpeakerStatistics>;
  total_speakers: number;
}

export interface DetectedEntity {
  text: string;
  label: string;
  category: string;
  count: number;
}

export interface DetectedTopic {
  text: string;
  count: number;
}

export interface TranscriptionResponse {
  text: string;
  segments?: Segment[];
  language?: string;
  duration?: number;
  model?: string;
  paragraphs?: Paragraph[];
  statistics?: Statistics;
  entities?: DetectedEntity[];
  topics?: DetectedTopic[];
}

export interface TextRule {
  name: string;
  find: string;
  replace: string;
  isRegex: boolean;
  flags: string;
  category: "filler" | "replace" | "pii";
  enabled: boolean;
}

export interface TextRuleset {
  version: number;
  name: string;
  rules: TextRule[];
}

export type TextRuleCategory = "all" | "filler" | "replace" | "pii";

export interface TranscriptionOptions {
  diarize: boolean;
  numSpeakers?: number;
  minSpeakers?: number;
  maxSpeakers?: number;
  detectParagraphs: boolean;
  paragraphSilenceThreshold: number;
  textRulesEnabled: boolean;
  textRules: TextRule[];
  textRuleCategory: TextRuleCategory;
  minConfidence: number;
  speakerLabels: Record<string, string>;
  detectEntities: boolean;
  detectTopics: boolean;
  detectSentiment: boolean;
  visibleEntityTypes: Record<string, boolean>;
  visibleSentimentTypes: Record<string, boolean>;
}

export interface EntityTypeConfig {
  key: string;
  label: string;
  colorClass: string;
}

export const ENTITY_TYPE_CONFIG: EntityTypeConfig[] = [
  { key: "PERSON", label: "People", colorClass: "bg-purple-500/15 text-purple-400" },
  { key: "ORG", label: "Organizations", colorClass: "bg-blue-500/15 text-blue-400" },
  { key: "GPE", label: "Locations", colorClass: "bg-green-500/15 text-green-400" },
  { key: "DATE", label: "Dates", colorClass: "bg-yellow-500/15 text-yellow-500" },
  { key: "TIME", label: "Times", colorClass: "bg-yellow-500/15 text-yellow-500" },
  { key: "MONEY", label: "Money", colorClass: "bg-emerald-500/15 text-emerald-400" },
  { key: "CARDINAL", label: "Numbers", colorClass: "bg-gray-500/15 text-gray-400" },
  { key: "ORDINAL", label: "Ordinals", colorClass: "bg-gray-500/15 text-gray-400" },
  { key: "PRODUCT", label: "Products", colorClass: "bg-orange-500/15 text-orange-400" },
  { key: "EVENT", label: "Events", colorClass: "bg-pink-500/15 text-pink-400" },
  { key: "WORK_OF_ART", label: "Works", colorClass: "bg-indigo-500/15 text-indigo-400" },
  { key: "LAW", label: "Legal", colorClass: "bg-red-500/15 text-red-400" },
  { key: "NORP", label: "Groups", colorClass: "bg-teal-500/15 text-teal-400" },
];

export interface SentimentTypeConfig {
  key: string;
  label: string;
  colorClass: string;
}

export const SENTIMENT_TYPE_CONFIG: SentimentTypeConfig[] = [
  { key: "positive", label: "Positive", colorClass: "bg-emerald-500/15 text-emerald-400" },
  { key: "neutral", label: "Neutral", colorClass: "bg-gray-500/15 text-gray-400" },
  { key: "negative", label: "Negative", colorClass: "bg-red-500/15 text-red-400" },
];

export const SENTIMENT_CONFIG: Record<string, { label: string; colorClass: string }> = Object.fromEntries(
  SENTIMENT_TYPE_CONFIG.map((t) => [t.key, { label: t.label, colorClass: t.colorClass }])
);

const DEFAULT_VISIBLE_ENTITY_TYPES: Record<string, boolean> = Object.fromEntries(
  ENTITY_TYPE_CONFIG.map((t) => [t.key, true])
);

export const DEFAULT_OPTIONS: TranscriptionOptions = {
  diarize: true,
  detectParagraphs: true,
  paragraphSilenceThreshold: 0.8,
  textRulesEnabled: true,
  textRules: [],
  textRuleCategory: "all",
  minConfidence: 0.0,
  speakerLabels: {},
  detectEntities: true,
  detectTopics: false,
  detectSentiment: true,
  visibleEntityTypes: { ...DEFAULT_VISIBLE_ENTITY_TYPES },
  visibleSentimentTypes: { positive: true, neutral: true, negative: true },
};

/** Full config schema for import/export. Serializable version of TranscriptionOptions. */
export interface EchoStudioConfig {
  version: 1;
  name?: string;
  // Transcription
  diarize: boolean;
  numSpeakers?: number;
  minSpeakers?: number;
  maxSpeakers?: number;
  // Post-processing
  detectParagraphs: boolean;
  paragraphSilenceThreshold: number;
  minConfidence: number;
  // Smart Redaction & PII
  textRulesEnabled: boolean;
  textRules: TextRule[];
  textRuleCategory: TextRuleCategory;
  // Audio Intelligence
  detectEntities: boolean;
  detectTopics: boolean;
  detectSentiment: boolean;
  visibleEntityTypes: Record<string, boolean>;
  visibleSentimentTypes: Record<string, boolean>;
  // Speaker labels (optional — only present if user set them)
  speakerLabels?: Record<string, string>;
}

export function optionsToConfig(options: TranscriptionOptions, name?: string): EchoStudioConfig {
  return {
    version: 1,
    name: name ?? "Echo Scribe Config",
    diarize: options.diarize,
    ...(options.numSpeakers != null && { numSpeakers: options.numSpeakers }),
    ...(options.minSpeakers != null && { minSpeakers: options.minSpeakers }),
    ...(options.maxSpeakers != null && { maxSpeakers: options.maxSpeakers }),
    detectParagraphs: options.detectParagraphs,
    paragraphSilenceThreshold: options.paragraphSilenceThreshold,
    minConfidence: options.minConfidence,
    textRulesEnabled: options.textRulesEnabled,
    textRules: options.textRules,
    textRuleCategory: options.textRuleCategory,
    detectEntities: options.detectEntities,
    detectTopics: options.detectTopics,
    detectSentiment: options.detectSentiment,
    visibleEntityTypes: options.visibleEntityTypes,
    visibleSentimentTypes: options.visibleSentimentTypes,
    ...(Object.keys(options.speakerLabels).length > 0 && { speakerLabels: options.speakerLabels }),
  };
}

export function configToOptions(config: EchoStudioConfig): TranscriptionOptions {
  return {
    ...DEFAULT_OPTIONS,
    diarize: config.diarize ?? DEFAULT_OPTIONS.diarize,
    numSpeakers: config.numSpeakers,
    minSpeakers: config.minSpeakers,
    maxSpeakers: config.maxSpeakers,
    detectParagraphs: config.detectParagraphs ?? DEFAULT_OPTIONS.detectParagraphs,
    paragraphSilenceThreshold: config.paragraphSilenceThreshold ?? DEFAULT_OPTIONS.paragraphSilenceThreshold,
    minConfidence: config.minConfidence ?? DEFAULT_OPTIONS.minConfidence,
    textRulesEnabled: config.textRulesEnabled ?? DEFAULT_OPTIONS.textRulesEnabled,
    textRules: config.textRules ?? [],
    textRuleCategory: config.textRuleCategory ?? "all",
    detectEntities: config.detectEntities ?? DEFAULT_OPTIONS.detectEntities,
    detectTopics: config.detectTopics ?? DEFAULT_OPTIONS.detectTopics,
    detectSentiment: config.detectSentiment ?? DEFAULT_OPTIONS.detectSentiment,
    visibleEntityTypes: config.visibleEntityTypes ?? DEFAULT_OPTIONS.visibleEntityTypes,
    visibleSentimentTypes: config.visibleSentimentTypes ?? DEFAULT_OPTIONS.visibleSentimentTypes,
    speakerLabels: config.speakerLabels ?? {},
  };
}

export interface HealthResponse {
  status: string;
  engine: string;
  model_loaded: boolean;
  model_id: string;
  model_name?: string;
  cuda_available: boolean;
  gpu_name?: string;
  gpu_memory?: {
    allocated_mb: number;
    reserved_mb: number;
  };
  diarization_available: boolean;
}

export interface UsageResponse {
  request_count: number;
  total_audio_minutes: number;
  total_bytes_uploaded: number;
  total_gpu_seconds: number;
  last_used: string | null;
}

export type ExportFormat = "srt" | "vtt" | "txt" | "json";

export type AppState = "idle" | "uploading" | "transcribing" | "done" | "error";

export type ViewMode = "line" | "para";

export type OutputTab = "transcript" | "json" | "srt" | "vtt" | "txt";
