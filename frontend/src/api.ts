import type { TranscriptionResponse, TranscriptionOptions, HealthResponse, SentimentResult, UsageResponse } from "./types";

const BASE = import.meta.env.DEV ? "" : "";

const API_KEY_STORAGE_KEY = "echo-scribe-api-key";

export function getStoredApiKey(): string {
  return localStorage.getItem(API_KEY_STORAGE_KEY) || "";
}

export function setStoredApiKey(key: string): void {
  if (key) {
    localStorage.setItem(API_KEY_STORAGE_KEY, key);
  } else {
    localStorage.removeItem(API_KEY_STORAGE_KEY);
  }
}

function authHeaders(): HeadersInit {
  const key = getStoredApiKey();
  if (key) {
    return { Authorization: `Bearer ${key}` };
  }
  return {};
}

export async function transcribe(
  file: File,
  options: TranscriptionOptions
): Promise<TranscriptionResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("response_format", "verbose_json");
  form.append("diarize", String(options.diarize));

  // Diarization hints
  if (options.numSpeakers != null) {
    form.append("num_speakers", String(options.numSpeakers));
  }
  if (options.minSpeakers != null) {
    form.append("min_speakers", String(options.minSpeakers));
  }
  if (options.maxSpeakers != null) {
    form.append("max_speakers", String(options.maxSpeakers));
  }

  // Post-processing
  form.append("detect_paragraphs", String(options.detectParagraphs));
  form.append("paragraph_silence_threshold", String(options.paragraphSilenceThreshold));

  if (options.minConfidence > 0) {
    form.append("min_confidence", String(options.minConfidence));
  }

  // Text rules are applied client-side only (post-processing.ts) so raw data stays clean
  // and toggling on/off works without re-transcribing.

  if (Object.keys(options.speakerLabels).length > 0) {
    form.append("speaker_labels", JSON.stringify(options.speakerLabels));
  }

  // Audio intelligence
  if (options.detectEntities) {
    form.append("detect_entities", "true");
  }
  if (options.detectTopics) {
    form.append("detect_topics", "true");
  }
  if (options.detectSentiment) {
    form.append("detect_sentiment", "true");
  }

  const res = await fetch(`${BASE}/v1/audio/transcriptions`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function annotateEntities(
  paragraphTexts: { text: string }[]
): Promise<(Record<string, number> | null)[]> {
  const res = await fetch(`${BASE}/v1/audio/entities`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(paragraphTexts),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function annotateSentiment(
  paragraphTexts: { text: string }[]
): Promise<(SentimentResult | null)[]> {
  const res = await fetch(`${BASE}/v1/audio/sentiment`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(paragraphTexts),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchUsage(): Promise<UsageResponse | null> {
  const key = getStoredApiKey();
  if (!key) return null;
  try {
    const res = await fetch(`${BASE}/v1/audio/usage`, {
      headers: authHeaders(),
    });
    if (!res.ok) return null;
    const data = await res.json();
    if (!data || !data.request_count) return null;
    return data;
  } catch {
    return null;
  }
}
