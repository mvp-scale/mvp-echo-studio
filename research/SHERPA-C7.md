# sherpa-onnx

## MVP-Echo Studio Alignment Review (2026-02-10)

Comparison of our Sherpa-ONNX implementation against the Context7 reference documentation.
Validated during Docker rebuild after resolving model download failures and API signature mismatches.

### Diarization: `process()` takes samples only, no sample_rate

- **Context7**: `result = sd.process(audio, callback=progress_callback)`
- **Our code**: `diar_segments = self._diarization.process(audio)`
- **Decision**: Aligned. The `process()` signature accepts `(samples, callback=None)`. Our original code incorrectly passed `sample_rate` as a second positional arg, which was interpreted as the callback. Fixed 2026-02-10.

### ASR: High-level `from_transducer()` factory is the correct API

- **Context7**: `sherpa_onnx.OfflineRecognizer.from_transducer(encoder=..., decoder=..., joiner=..., tokens=...)`
- **Our code**: Same factory, plus `model_type="nemo_transducer"` and `provider="cuda"`
- **Decision**: Aligned. The low-level `OfflineTransducerModelConfig` constructor uses different parameter names (`encoder_filename` not `encoder`) which caused our original startup crash. The factory method is the documented API and handles config wiring internally.

### GPU: CUDA 12 + cuDNN 9 pip wheel, not CPU wheel + LD_LIBRARY_PATH hack

- **Context7**: Defaults to `provider="cpu"` in all examples (general-purpose docs)
- **Our code**: `provider="cuda"` with `sherpa-onnx==1.12.23+cuda12.cudnn9` pip wheel
- **Decision**: Diverges intentionally. Context7 targets broad compatibility. We target RTX 3090 performance. The CPU pip package silently falls back from `provider="cuda"` with no error — only a log warning. Switching to the GPU wheel resolved this.

### Batch decoding: `decode_streams()` — our performance advantage

- **Context7**: Shows only `decode_stream(stream)` (singular, one-at-a-time)
- **Our code**: `self._recognizer.decode_streams(streams)` (plural, batch GPU)
- **Decision**: Intentional divergence. Context7 doesn't document batch decoding, but it's the key to our ~5-7s target for 18 minutes of audio. All diarized segments are decoded in a single GPU batch call.

### Embedding model: CAM++ (27MB, bilingual) over ERes2Net (38MB, Chinese)

- **Context7**: `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` (38MB, zh)
- **Our code**: `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` (27MB, zh+en)
- **Decision**: Intentional. Our content is primarily English. CAM++ is lighter, bilingual, and the "advanced" variant. ERes2Net is Chinese-focused. Both are valid for the 3D-Speaker family.

### Model downloads: GitHub releases, not HuggingFace

- **Context7**: References model paths as local files (assumes pre-downloaded)
- **Our code**: `entrypoint-sherpa.sh` downloads from GitHub releases using `curl`
- **Decision**: Intentional. HuggingFace mirrors are gated for some models (embedding model returns 401). GitHub releases are public, no-auth, and canonical. Models persist in Docker volume between rebuilds.

### Diarization tuning params we omit (for now)

- **Context7**: Sets `min_duration_on=0.3`, `min_duration_off=0.5`, `num_clusters=4`
- **Our code**: Sets only `threshold=0.5` (auto-detect speaker count)
- **Decision**: Acceptable for initial validation. `min_duration_on/off` improves segment quality by filtering sub-300ms blips and merging short gaps. `num_clusters=-1` (auto) is correct when speaker count is unknown. Can tune after benchmarking.

### Diarization result ordering

- **Context7**: `result.sort_by_start_time()` for time-ordered iteration
- **Our code**: Iterates result directly without explicit sorting
- **Decision**: Should add `sort_by_start_time()` after validation passes. Not a correctness issue but ensures deterministic output order.

### Features available but not yet used

| Feature | Context7 Section | Status | Notes |
|---------|-----------------|--------|-------|
| Progress callback | Diarization | Deferred | Wire to WebSocket for long-file progress |
| VAD pre-processing | Silero VAD | Not needed | Diarization already segments speech from silence |
| Word-level timestamps | `result.timestamps` | Deferred | Available from ASR result, not yet surfaced |
| Keyword spotting | Wake word API | Out of scope | Not relevant for batch transcription |
| Audio tagging | Event classification | Out of scope | Could detect music/noise segments later |
| TTS | Text-to-speech | Out of scope | Not in project requirements |

---

sherpa-onnx is a comprehensive, cross-platform speech processing library that enables running various speech-related models locally without internet connectivity. Built on ONNX Runtime, it provides high-performance speech recognition (ASR), text-to-speech (TTS), voice activity detection (VAD), speaker diarization, speaker identification/verification, keyword spotting, audio tagging, and spoken language identification. The library supports both streaming (real-time) and non-streaming (offline) processing modes for applicable tasks.

The library offers bindings for 12+ programming languages including C++, C, Python, JavaScript/Node.js, Go, C#, Java, Kotlin, Swift, Dart, Rust, and Pascal. It runs on virtually all platforms: Linux, macOS, Windows, Android, iOS, HarmonyOS, WebAssembly, and embedded devices like Raspberry Pi, NVIDIA Jetson, and various NPU-equipped boards (Rockchip RKNN, Qualcomm QNN, Ascend NPU). Pre-trained models are available for multiple languages and architectures.

## Speech Recognition APIs

### Non-Streaming (Offline) ASR

Non-streaming ASR processes complete audio files at once, suitable for batch transcription. Supports multiple model architectures including Transducer, Paraformer, Whisper, NeMo CTC, SenseVoice, and Moonshine.

```python
import sherpa_onnx
import numpy as np
import wave

# Read audio file
def read_wave(filename):
    with wave.open(filename) as f:
        samples = f.readframes(f.getnframes())
        samples = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768
        return samples, f.getframerate()

# Create recognizer using Whisper model
recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
    encoder="./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx",
    decoder="./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx",
    tokens="./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt",
    num_threads=2,
    language="en",
    task="transcribe",
)

# Alternative: Create recognizer using Paraformer model
# recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
#     paraformer="./model.onnx",
#     tokens="./tokens.txt",
#     num_threads=2,
# )

# Alternative: Create recognizer using Transducer model
# recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
#     encoder="./encoder.onnx",
#     decoder="./decoder.onnx",
#     joiner="./joiner.onnx",
#     tokens="./tokens.txt",
#     num_threads=2,
# )

# Process audio files
samples, sample_rate = read_wave("./test.wav")
stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, samples)

recognizer.decode_stream(stream)
result = stream.result
print(f"Transcription: {result.text}")
print(f"Timestamps: {result.timestamps}")
```

### Streaming (Online) ASR

Streaming ASR processes audio in real-time chunks, ideal for live transcription from microphones or streams.

```python
import sherpa_onnx
import numpy as np

# Create streaming recognizer with Transducer model
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="./tokens.txt",
    encoder="./encoder-epoch-99-avg-1-chunk-16-left-64.onnx",
    decoder="./decoder-epoch-99-avg-1-chunk-16-left-64.onnx",
    joiner="./joiner-epoch-99-avg-1-chunk-16-left-64.onnx",
    num_threads=2,
    sample_rate=16000,
    feature_dim=80,
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.4,  # endpoint after 2.4s silence (no speech detected)
    rule2_min_trailing_silence=1.2,  # endpoint after 1.2s silence (speech detected)
    rule3_min_utterance_length=20,   # endpoint after 20s utterance
)

# Alternative: Streaming Paraformer
# recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
#     tokens="./tokens.txt",
#     encoder="./encoder.int8.onnx",
#     decoder="./decoder.int8.onnx",
#     num_threads=2,
# )

# Create stream and process audio chunks
stream = recognizer.create_stream()
sample_rate = 16000

# Simulate streaming: process audio in chunks
audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of audio
chunk_size = 1600  # 100ms chunks

for i in range(0, len(audio_data), chunk_size):
    chunk = audio_data[i:i+chunk_size]
    stream.accept_waveform(sample_rate, chunk)

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    result = recognizer.get_result(stream)
    if result:
        print(f"Partial: {result}")

    # Check for endpoint
    if recognizer.is_endpoint(stream):
        print(f"Final: {recognizer.get_result(stream)}")
        recognizer.reset(stream)

# Signal end of audio
stream.input_finished()
```

## Text-to-Speech (TTS) API

### Offline TTS Generation

Synthesize speech from text using various TTS engines including VITS, Matcha, Kokoro, and Kitten models.

```python
import sherpa_onnx
import soundfile as sf

# Configure TTS with VITS model (Piper)
tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
            tokens="./vits-piper-en_US-amy-low/tokens.txt",
            data_dir="./vits-piper-en_US-amy-low/espeak-ng-data",
        ),
        num_threads=2,
        provider="cpu",
    ),
    max_num_sentences=1,
)

# Alternative: Matcha TTS with Vocos vocoder
# tts_config = sherpa_onnx.OfflineTtsConfig(
#     model=sherpa_onnx.OfflineTtsModelConfig(
#         matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
#             acoustic_model="./matcha-icefall-en_US-ljspeech/model-steps-3.onnx",
#             vocoder="./vocos-22khz-univ.onnx",
#             tokens="./matcha-icefall-en_US-ljspeech/tokens.txt",
#             data_dir="./matcha-icefall-en_US-ljspeech/espeak-ng-data",
#         ),
#         num_threads=2,
#     ),
# )

# Alternative: Kokoro TTS (multi-speaker, multilingual)
# tts_config = sherpa_onnx.OfflineTtsConfig(
#     model=sherpa_onnx.OfflineTtsModelConfig(
#         kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
#             model="./kokoro-multi-lang-v1_0/model.onnx",
#             voices="./kokoro-multi-lang-v1_0/voices.bin",
#             tokens="./kokoro-multi-lang-v1_0/tokens.txt",
#             data_dir="./kokoro-multi-lang-v1_0/espeak-ng-data",
#         ),
#         num_threads=2,
#     ),
# )

tts = sherpa_onnx.OfflineTts(tts_config)

# Generate speech
text = "Hello world. This is a text to speech example."
audio = tts.generate(
    text,
    sid=0,      # Speaker ID for multi-speaker models
    speed=1.0,  # Speed factor: <1 slower, >1 faster
)

# Save to file
sf.write("output.wav", audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
print(f"Generated {len(audio.samples)/audio.sample_rate:.2f}s of audio at {audio.sample_rate}Hz")
```

## Voice Activity Detection (VAD) API

### Silero VAD with ASR Integration

Detect speech segments in audio and transcribe only the speech portions, reducing processing time and improving accuracy.

```python
import sherpa_onnx
import numpy as np
import sounddevice as sd

# Configure VAD
vad_config = sherpa_onnx.VadModelConfig()
vad_config.silero_vad.model = "./silero_vad.onnx"
vad_config.silero_vad.threshold = 0.5          # Speech detection threshold
vad_config.silero_vad.min_silence_duration = 0.25  # Min silence to end segment (seconds)
vad_config.silero_vad.min_speech_duration = 0.25   # Min speech duration (seconds)
vad_config.sample_rate = 16000

vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

# Create offline recognizer for transcribing detected speech
recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
    tokens="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
    num_threads=2,
    use_itn=True,
)

# Process audio from microphone
sample_rate = 16000
window_size = vad_config.silero_vad.window_size

with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
    buffer = np.array([], dtype=np.float32)

    while True:
        samples, _ = s.read(int(0.1 * sample_rate))  # Read 100ms
        samples = samples.reshape(-1)
        buffer = np.concatenate([buffer, samples])

        # Feed VAD in window-sized chunks
        while len(buffer) >= window_size:
            vad.accept_waveform(buffer[:window_size])
            buffer = buffer[window_size:]

        # Process detected speech segments
        while not vad.empty():
            speech_segment = vad.front

            # Transcribe the speech segment
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, speech_segment.samples)
            recognizer.decode_stream(stream)

            text = stream.result.text.strip()
            if text:
                print(f"[{speech_segment.start/sample_rate:.2f}s] {text}")

            vad.pop()
```

## Speaker Identification API

### Speaker Embedding and Recognition

Extract speaker embeddings and perform speaker identification/verification.

```python
import sherpa_onnx
import numpy as np
import soundfile as sf

# Create speaker embedding extractor
extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model="./wespeaker_en_resnet34.onnx",
    num_threads=2,
    provider="cpu",
)
extractor = sherpa_onnx.SpeakerEmbeddingExtractor(extractor_config)

# Create speaker embedding manager for registration and search
manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)

def compute_embedding(audio_file):
    """Compute speaker embedding from audio file."""
    audio, sample_rate = sf.read(audio_file, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # Use first channel

    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=audio)
    stream.input_finished()

    if extractor.is_ready(stream):
        return np.array(extractor.compute(stream))
    return None

# Register speakers
speakers = {
    "Alice": ["./alice_sample1.wav", "./alice_sample2.wav"],
    "Bob": ["./bob_sample1.wav"],
}

for name, files in speakers.items():
    embeddings = [compute_embedding(f) for f in files]
    avg_embedding = np.mean(embeddings, axis=0)
    manager.add(name, avg_embedding)
    print(f"Registered speaker: {name}")

# Identify unknown speaker
unknown_embedding = compute_embedding("./unknown_speaker.wav")
threshold = 0.5
identified_name = manager.search(unknown_embedding, threshold=threshold)

if identified_name:
    print(f"Identified as: {identified_name}")
else:
    print("Unknown speaker")

# Verify specific speaker
is_alice = manager.verify("Alice", unknown_embedding, threshold=threshold)
print(f"Is Alice: {is_alice}")
```

## Speaker Diarization API

### Offline Speaker Diarization

Segment audio by speaker and identify who spoke when.

```python
import sherpa_onnx
import soundfile as sf

# Configure speaker diarization
config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model="./sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
        ),
    ),
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model="./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    ),
    clustering=sherpa_onnx.FastClusteringConfig(
        num_clusters=4,     # Set if known, otherwise use -1
        threshold=0.5,      # Used when num_clusters=-1
    ),
    min_duration_on=0.3,    # Min speech segment duration
    min_duration_off=0.5,   # Min gap to merge segments
)

sd = sherpa_onnx.OfflineSpeakerDiarization(config)

# Load and process audio
audio, sample_rate = sf.read("./meeting.wav", dtype="float32")
if len(audio.shape) > 1:
    audio = audio[:, 0]

# Process with progress callback
def progress_callback(num_processed, num_total):
    print(f"Progress: {num_processed/num_total*100:.1f}%")
    return 0  # Return 0 to continue, non-zero to stop

result = sd.process(audio, callback=progress_callback)

# Print diarization results
for segment in result.sort_by_start_time():
    print(f"{segment.start:.3f} -- {segment.end:.3f}: Speaker_{segment.speaker:02d}")
```

## Keyword Spotting API

### Wake Word Detection

Detect predefined keywords or wake words in streaming audio.

```python
import sherpa_onnx
import numpy as np

# Create keyword spotter
kws = sherpa_onnx.KeywordSpotter(
    tokens="./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt",
    encoder="./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
    decoder="./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
    joiner="./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
    keywords_file="./keywords.txt",  # One keyword per line
    num_threads=2,
    keywords_score=1.0,
    keywords_threshold=0.25,
)

# Create stream with additional runtime keywords
# Format: "phonemes @display_text" separated by /
stream = kws.create_stream("h ə l oʊ @hello/w ɜː l d @world")

# Process audio
sample_rate = 16000
audio_samples = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds

# Feed audio in chunks
chunk_size = 1600  # 100ms
for i in range(0, len(audio_samples), chunk_size):
    chunk = audio_samples[i:i+chunk_size]
    stream.accept_waveform(sample_rate, chunk)

    while kws.is_ready(stream):
        kws.decode_stream(stream)
        result = kws.get_result(stream)

        if result:
            print(f"Detected keyword: {result}")
            kws.reset_stream(stream)  # Reset after detection
```

## Audio Tagging API

### Audio Event Classification

Classify audio events using AudioSet-trained models.

```python
import sherpa_onnx
import soundfile as sf
import numpy as np

# Create audio tagger
config = sherpa_onnx.AudioTaggingConfig(
    model=sherpa_onnx.AudioTaggingModelConfig(
        zipformer=sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(
            model="./sherpa-onnx-zipformer-audio-tagging-2024-04-09/model.onnx",
        ),
        num_threads=2,
        provider="cpu",
    ),
    labels="./sherpa-onnx-zipformer-audio-tagging-2024-04-09/class_labels_indices.csv",
    top_k=5,
)

tagger = sherpa_onnx.AudioTagging(config)

# Load and process audio
audio, sample_rate = sf.read("./audio_sample.wav", dtype="float32")
if len(audio.shape) > 1:
    audio = audio[:, 0]

stream = tagger.create_stream()
stream.accept_waveform(sample_rate=sample_rate, waveform=audio)

# Get top-k predictions
results = tagger.compute(stream)
for i, event in enumerate(results):
    print(f"{i+1}. {event.name}: {event.prob:.4f}")
```

## C API Example

### Streaming ASR in C

Use the C API for integration with C/C++ applications.

```c
#include <stdio.h>
#include <stdlib.h>
#include "sherpa-onnx/c-api/c-api.h"

int main() {
    // Configure recognizer
    SherpaOnnxOnlineRecognizerConfig config;
    memset(&config, 0, sizeof(config));

    config.model_config.tokens = "./tokens.txt";
    config.model_config.transducer.encoder = "./encoder.onnx";
    config.model_config.transducer.decoder = "./decoder.onnx";
    config.model_config.transducer.joiner = "./joiner.onnx";
    config.model_config.num_threads = 2;
    config.model_config.provider = "cpu";

    config.feat_config.sample_rate = 16000;
    config.feat_config.feature_dim = 80;

    config.decoding_method = "greedy_search";
    config.enable_endpoint = 1;
    config.rule1_min_trailing_silence = 2.4;
    config.rule2_min_trailing_silence = 1.2;

    // Create recognizer and stream
    const SherpaOnnxOnlineRecognizer *recognizer =
        SherpaOnnxCreateOnlineRecognizer(&config);
    const SherpaOnnxOnlineStream *stream =
        SherpaOnnxCreateOnlineStream(recognizer);

    // Read and process audio file
    const SherpaOnnxWave *wave = SherpaOnnxReadWave("./test.wav");

    // Process in chunks (simulate streaming)
    int chunk_size = 3200;  // 200ms at 16kHz
    for (int i = 0; i < wave->num_samples; i += chunk_size) {
        int end = (i + chunk_size > wave->num_samples) ?
                   wave->num_samples : i + chunk_size;

        SherpaOnnxOnlineStreamAcceptWaveform(
            stream, wave->sample_rate,
            wave->samples + i, end - i);

        while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
            SherpaOnnxDecodeOnlineStream(recognizer, stream);
        }

        const SherpaOnnxOnlineRecognizerResult *result =
            SherpaOnnxGetOnlineStreamResult(recognizer, stream);
        printf("Partial: %s\n", result->text);
        SherpaOnnxDestroyOnlineRecognizerResult(result);
    }

    // Cleanup
    SherpaOnnxFreeWave(wave);
    SherpaOnnxDestroyOnlineStream(stream);
    SherpaOnnxDestroyOnlineRecognizer(recognizer);

    return 0;
}
```

## Node.js API Example

### Non-Streaming ASR in JavaScript

Use sherpa-onnx with Node.js applications.

```javascript
const sherpa_onnx = require('sherpa-onnx-node');

// Configuration for Whisper model
const config = {
    featConfig: {
        sampleRate: 16000,
        featureDim: 80,
    },
    modelConfig: {
        whisper: {
            encoder: './sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx',
            decoder: './sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx',
        },
        tokens: './sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt',
        numThreads: 2,
        provider: 'cpu',
    }
};

// Create recognizer
const recognizer = new sherpa_onnx.OfflineRecognizer(config);

// Read audio file
const wave = sherpa_onnx.readWave('./test.wav');

// Process
const stream = recognizer.createStream();
stream.acceptWaveform({
    sampleRate: wave.sampleRate,
    samples: wave.samples
});

recognizer.decode(stream);
const result = recognizer.getResult(stream);

console.log('Transcription:', result.text);
console.log('Duration:', wave.samples.length / wave.sampleRate, 'seconds');
```

## Go API Example

### Non-Streaming ASR in Go

Use sherpa-onnx with Go applications.

```go
package main

import (
    "log"
    sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
    config := sherpa.OfflineRecognizerConfig{}

    // Configure Whisper model
    config.ModelConfig.Whisper.Encoder = "./whisper-encoder.onnx"
    config.ModelConfig.Whisper.Decoder = "./whisper-decoder.onnx"
    config.ModelConfig.Whisper.Language = "en"
    config.ModelConfig.Whisper.Task = "transcribe"
    config.ModelConfig.Tokens = "./tokens.txt"
    config.ModelConfig.NumThreads = 2
    config.ModelConfig.Provider = "cpu"

    config.FeatConfig.SampleRate = 16000
    config.FeatConfig.FeatureDim = 80

    // Create recognizer
    recognizer := sherpa.NewOfflineRecognizer(&config)
    defer sherpa.DeleteOfflineRecognizer(recognizer)

    // Read audio (implement readWave function)
    samples, sampleRate := readWave("./test.wav")

    // Process
    stream := sherpa.NewOfflineStream(recognizer)
    defer sherpa.DeleteOfflineStream(stream)

    stream.AcceptWaveform(sampleRate, samples)
    recognizer.Decode(stream)

    result := stream.GetResult()
    log.Printf("Text: %s", result.Text)
    log.Printf("Language: %s", result.Lang)
}
```

## Summary

sherpa-onnx provides a unified, cross-platform solution for deploying speech processing capabilities locally. The main use cases include: building voice assistants with wake word detection and streaming ASR; creating transcription applications for meetings, podcasts, or videos using offline ASR; developing accessibility tools with real-time speech-to-text; building IVR systems with speaker verification; and creating multilingual TTS applications. The library excels in edge deployment scenarios where internet connectivity is unavailable or where privacy concerns require local processing.

Integration patterns typically involve: (1) streaming ASR with VAD for real-time transcription that only processes speech segments; (2) speaker diarization followed by ASR for meeting transcription with speaker labels; (3) keyword spotting as a wake word trigger followed by full ASR; (4) speaker identification combined with ASR for personalized voice interfaces. Models are loaded once at startup and reused across requests. For optimal performance, use appropriate thread counts based on CPU cores, choose quantized (int8) models for resource-constrained devices, and leverage hardware acceleration (CUDA, CoreML) when available.