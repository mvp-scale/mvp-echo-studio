Comprehensive Technical Report: Architecting High-Performance Offline Meeting Transcription Systems via Containerized Sherpa-ONNX1. Executive SummaryThe convergence of end-to-end deep learning architectures and optimized inference runtimes has fundamentally altered the landscape of automated speech processing. Historically, high-accuracy meeting transcription—characterized by long-form audio, overlapping speech, and diverse acoustic environments—was the exclusive domain of cloud-based proprietary APIs. However, the emergence of the Sherpa-ONNX framework, combined with NVIDIA’s Parakeet-TDT (Token-and-Duration Transducer) architectures and Pyannote speaker diarization models, has democratized access to production-grade, offline transcription capabilities.This report provides an exhaustive technical analysis of designing, implementing, and deploying a batched meeting transcription system within a Dockerized environment. The analysis prioritizes data sovereignty, computational efficiency, and architectural robustness. It dissects the transition from Recurrent Neural Network Transducers (RNN-T) to Token-and-Duration Transducers (TDT), quantifying the efficiency gains achieved through frame-skipping decoding mechanisms. Furthermore, it explicates the integration of a multi-stage modular pipeline where speaker diarization (segmentation and clustering) precedes and informs the Automatic Speech Recognition (ASR) process, utilizing batch processing to maximize Graphics Processing Unit (GPU) throughput.Central to this report is the engineering rigor required to containerize heterogeneous AI stacks. We address specific challenges such as the compatibility matrix between CUDA versions, cuDNN libraries, and ONNX Runtime execution providers, specifically focusing on the intricacies of deploying on NVIDIA hardware using CUDA 11.8 and emerging CUDA 12.x standards. By synthesizing findings from recent benchmarks, architectural documentation, and implementation case studies, this document serves as a definitive blueprint for systems architects seeking to deploy local, high-fidelity speech intelligence platforms.2. Theoretical Foundations of Next-Generation Speech ArchitecturesTo understand the implementation decisions detailed later in this report, one must first grasp the theoretical shifts in the underlying neural architectures. The move from "Next-Gen Kaldi" to Sherpa-ONNX represents a migration from research-centric toolkits to deployment-centric engines that prioritize portability without sacrificing the modeling advances of the PyTorch ecosystem.2.1 The Sherpa-ONNX Inference EngineSherpa-ONNX represents a paradigm shift in how speech models are served. Unlike its predecessor, which relied on LibTorch (the C++ frontend for PyTorch), Sherpa-ONNX is built upon ONNX Runtime (ORT). This architectural decision decouples the model training environment from the inference environment. ORT provides graph-level optimizations—such as operator fusion, constant folding, and elimination of redundant subgraphs—that are critical for minimizing latency on embedded devices and maximizing throughput on server-grade GPUs.The engine is engineered to support a diverse array of speech tasks within a single binary footprint. It supports streaming and non-streaming ASR, speaker diarization, voice activity detection (VAD), and text-to-speech (TTS), all while maintaining a minimal memory profile. This unification addresses the "dependency hell" often encountered when stitching together disparate Python libraries for each task (e.g., combining torchaudio for ASR with pyannote.audio for diarization). In Sherpa-ONNX, these tasks share the same underlying runtime and memory management principles, allowing for seamless data handoffs and reduced context switching overhead.2.2 The Evolution of ASR: Parakeet-TDTThe ASR component of the proposed pipeline utilizes the Parakeet-TDT family of models, specifically the sherpa-onnx-nemo-parakeet-tdt-0.6b-v3 variant. To appreciate its selection, one must understand the limitations of standard Transducers.2.2.1 Mechanics of Token-and-Duration TransducersTraditional RNN-Transducers (RNN-T) operate on a frame-by-frame basis. For every acoustic frame (typically 10ms to 40ms of audio), the decoder must decide whether to emit a text token (character or subword) or a "blank" symbol (indicating no output for that frame). This results in a sequential dependency where the number of decoding steps is roughly proportional to the number of audio frames, creating a bottleneck for long audio files.The Token-and-Duration Transducer (TDT) architecture introduces a secondary prediction head. At each decoding step, the model predicts not only the next token but also the duration (the number of acoustic frames) that this token consumes. This allows the decoder to advance its state by multiple frames in a single step, effectively "skipping" the redundancy inherent in speech signals. Empirical evidence suggests that TDT architectures can reduce the number of decoder calls by a factor of 2x to 3x compared to standard RNN-Ts, resulting in a Real-Time Factor (RTF) improvement of up to 64% without degradation in Word Error Rate (WER).2.2.2 Multilingual Robustness and V3 SpecificsThe "v3" iteration of the Parakeet-TDT model (0.6 billion parameters) is trained on a massive multilingual corpus, supporting 25 European languages including English, French, German, Spanish, and Russian. This multilingual capability is embedded within a single model weight file, eliminating the need for language-specific model switching in diverse business environments. The model utilizes a FastConformer encoder, which employs aggressive subsampling (typically 8x reduction) to further compress the acoustic sequence length before it reaches the decoder, synergizing with the TDT mechanism to maximize inference speed.2.3 Speaker Diarization: The Pyannote ParadigmFor meeting transcription, transcribing what was said is insufficient; the system must identify who said it. Pyannote-audio has established itself as the state-of-the-art open-source framework for this task. Sherpa-ONNX integrates Pyannote's segmentation architecture, specifically the pyannote/segmentation-3.0 model, converted to ONNX.2.3.1 Segmentation and Permutation Invariant TrainingThe core of the diarization pipeline is the segmentation model. Unlike older approaches that relied on Voice Activity Detection (VAD) followed by clustering, the Pyannote segmentation model treats diarization as a multi-label classification problem. It outputs a probability matrix over time, where each row corresponds to a potential speaker. Crucially, this architecture allows for the detection of overlapping speech—a scenario where two active speakers are identified simultaneously. This capability is vital for meeting analysis, where interruptions and concurrent speech are commonplace.2.3.2 Embedding and ClusteringOnce speech segments are identified, they are passed to a speaker embedding model. The reference implementation uses wespeaker-en-voxceleb-resnet34, a ResNet-based architecture trained on the VoxCeleb dataset. This model maps variable-length audio segments into a fixed-dimensional vector space (typically 256 dimensions) where the Euclidean distance or Cosine Similarity between vectors corresponds to speaker similarity. These embeddings are then grouped using clustering algorithms (such as Agglomerative Hierarchical Clustering or Spectral Clustering) to assign unique speaker labels (e.g., Speaker 0, Speaker 1) across the entire recording.3. Containerization Strategy: The Docker ImplementationThe deployment of hardware-accelerated AI pipelines is frequently plagued by environment inconsistencies. The interaction between the host operating system's kernel, the NVIDIA driver, the CUDA toolkit, cuDNN libraries, and the application-level runtime creates a fragile dependency chain. A Docker-based containerization strategy acts as a standardized unit of deployment, encapsulating these dependencies to ensure reproducibility across development, staging, and production environments.3.1 The CUDA Versioning ChallengeOne of the most critical engineering decisions in this pipeline is the selection of the CUDA version. As of early 2026, the ecosystem is in a transition period between CUDA 11.x and CUDA 12.x.Sherpa-ONNX Dependencies: The pre-compiled binary wheels and libraries for Sherpa-ONNX are often strictly coupled to specific CUDA versions. The documentation highlights that mixing libraries compiled for CUDA 11.x with a CUDA 12.x runtime (or vice versa) frequently leads to symbol lookup errors or silent failures where the provider falls back to CPU execution.Host Compatibility: While the Docker container provides the CUDA toolkit and userspace libraries, the host machine must have an NVIDIA driver capable of supporting that CUDA version. A driver installed for CUDA 12.0 is generally backward compatible with CUDA 11.8 containers, but the reverse is not true.Recommendation: For maximum stability and compatibility with the widest range of patched ONNX Runtime libraries provided by the Sherpa project, CUDA 11.8 remains the reference standard for production deployments, although migration paths to CUDA 12.x are actively supported for newer hardware (e.g., H100).3.2 Multi-Stage Build ArchitectureTo balance the conflicting requirements of a comprehensive build environment (compilers, headers, CMake) and a lean production image, a multi-stage Docker build is mandatory. This strategy minimizes the final image size and reduces the attack surface by excluding unnecessary build artifacts.3.2.1 Stage 1: The BuilderThe builder stage utilizes the devel variant of the NVIDIA CUDA image. This image includes the complete compiler toolchain (nvcc), which is necessary if any custom compilation of C++ extensions is required.Key Responsibilities of the Builder Stage:Toolchain Provisioning: Installation of git, cmake, g++, wget, and python3-dev.ONNX Runtime Acquisition: Standard pip install onnxruntime-gpu is often insufficient for Sherpa-ONNX due to C++ ABI incompatibilities. The build must download specific, patched versions of ONNX Runtime (e.g., onnxruntime-linux-x64-gpu-1.17.1-patched.zip) that expose the necessary C++ headers for Sherpa's backend.Source Compilation: Cloning the k2-fsa/sherpa-onnx repository and compiling the Python extension (_sherpa_onnx) from source. This ensures the bindings are linked against the exact versions of the CUDA and ORT libraries present in the environment.CMake Flags: Crucial flags include -DSHERPA_ONNX_ENABLE_GPU=ON and -DSHERPA_ONNX_ENABLE_PYTHON=ON. The SHERPA_ONNXRUNTIME_LIB_DIR environment variable must point to the patched libraries.3.2.2 Stage 2: The RuntimeThe final stage uses the runtime variant of the NVIDIA CUDA image. This image is significantly smaller (often by gigabytes) as it strips out the compiler stack.Artifact Transfer and Configuration:Library Copying: Only the compiled shared objects (.so files) and the Python wheel/egg are copied from the builder stage.Environment Variables: The LD_LIBRARY_PATH must be explicitly set to include the directory containing the patched ONNX Runtime libraries. Failure to do so prevents the dynamic linker from finding libonnxruntime_providers_cuda.so at runtime.Model Management: For strictly offline environments (air-gapped), models are downloaded during the build phase and baked into the image. For internet-connected deployments, they can be mounted via volumes to reduce image size.3.3 Dockerfile Implementation ReferenceThe following Dockerfile specification synthesizes these requirements into a concrete implementation:Dockerfile# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build dependencies
# git/cmake/g++: For compiling sherpa-onnx source
# python3-dev: Headers for Python C extension
# wget/unzip: For downloading models and patched libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    wget \
    unzip \
    ca-certificates \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Download patched ONNX Runtime for GPU support (Critical Step)
# Using v1.17.1 as per current stable Sherpa-ONNX recommendations for CUDA 11.8
RUN wget -q https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1-patched.zip \
    && unzip onnxruntime-linux-x64-gpu-1.17.1-patched.zip \
    && rm onnxruntime-linux-x64-gpu-1.17.1-patched.zip

# Set environment variables for CMake to locate the patched ORT
ENV SHERPA_ONNXRUNTIME_LIB_DIR=/workspace/onnxruntime-linux-x64-gpu-1.17.1-patched/lib
ENV SHERPA_ONNXRUNTIME_INCLUDE_DIR=/workspace/onnxruntime-linux-x64-gpu-1.17.1-patched/include

# Clone Sherpa-ONNX source code
RUN git clone https://github.com/k2-fsa/sherpa-onnx.git

# Compile Sherpa-ONNX Python bindings
WORKDIR /workspace/sherpa-onnx
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_SHARED_LIBS=ON \
          -DSHERPA_ONNX_ENABLE_GPU=ON \
          -DSHERPA_ONNX_ENABLE_PYTHON=ON \
         .. && \
    make -j$(nproc)

# Install the Python package
RUN python3 setup.py install

# -----------------------------------------------------------------------------
# Model Acquisition
# -----------------------------------------------------------------------------
WORKDIR /models

# Download Parakeet-TDT ASR Model (v3 Int8)
RUN wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2 && \
    tar xvf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2 && \
    rm sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2

# Download Pyannote Segmentation Model
RUN wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 && \
    tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 && \
    rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

# Download WeSpeaker Embedding Model
RUN wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx

# Download Punctuation Model
RUN wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2 && \
    tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2 && \
    rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2

# -----------------------------------------------------------------------------
# Stage 2: Production Runtime
# -----------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python runtime and minimal audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Transfer installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Transfer Patched ONNX Runtime Libraries
# These are essential for GPU acceleration
COPY --from=builder /workspace/onnxruntime-linux-x64-gpu-1.17.1-patched/lib /usr/local/lib/sherpa_onnx_runtime

# Configure Dynamic Linker
ENV LD_LIBRARY_PATH=/usr/local/lib/sherpa_onnx_runtime:$LD_LIBRARY_PATH

# Transfer Models
COPY --from=builder /models /models

# Application Setup
WORKDIR /app
COPY app.py.

# Define Entrypoint
ENTRYPOINT ["python3", "app.py"]
4. Pipeline Implementation Logic: The Batched ArchitectureThe core innovation in this implementation is the batched processing pipeline. A naive approach might process audio files sequentially, or process detected segments one by one. Such an approach significantly underutilizes modern GPU architectures, which thrive on parallel computation. The proposed pipeline maximizes throughput by decoupling the diarization stage from the ASR stage and using batching to saturate the GPU.4.1 Audio Ingestion and PreprocessingThe entry point of the pipeline involves loading the raw audio. Meeting recordings often arrive in various formats (MP3, M4A, WAV). The pipeline utilizes ffmpeg (via subprocess or libraries like pydub) to decode these into a raw float32 array normalized to the range [-1, 1]. Crucially, all models in the Sherpa-ONNX ecosystem expect a sampling rate of 16,000 Hz. Resampling must occur immediately upon ingestion to prevent downstream artifacts.4.2 Phase 1: Speaker DiarizationThe first computational phase answers "who is speaking." This is executed using the OfflineSpeakerDiarization class.4.2.1 ConfigurationThe configuration object aggregates the segmentation, embedding, and clustering parameters.Segmentation: The PyannoteModelConfig points to the model.onnx extracted from the downloaded archive.Embedding: The SpeakerEmbeddingExtractorConfig points to the wespeaker ResNet34 model.Clustering: The FastClusteringConfig controls how embeddings are grouped. Setting num_clusters=-1 enables automatic cluster estimation, where the algorithm infers the number of speakers based on the embedding distances. Alternatively, if the number of meeting participants is known, fixing this value improves accuracy.4.2.2 ExecutionThe diarizer.process(audio_samples) method runs the entire diarization logic. It returns a list of segment objects, each containing a start time, end time, and speaker_id. This phase typically runs faster than real-time (e.g., processing 1 hour of audio in a few minutes) because it only requires forward passes of the segmentation and embedding models, which are lighter than the ASR transducer.4.3 Phase 2: Segmentation and Dynamic BatchingOnce the timeline of speaker turns is established, the original audio waveform is sliced according to these timestamps. This results in a collection of audio chunks, each corresponding to a single speaker's utterance.To optimize for GPU inference, these chunks are not processed individually. Instead, they are organized into batches.Batch Size Strategy: A batch size of 10 to 30 is generally optimal for int8 Parakeet models on consumer GPUs (e.g., RTX 3090/4090). This ensures the GPU has enough work to hide memory access latency without exceeding VRAM limits.Padding: Since audio segments vary in length, batching requires padding shorter segments to match the length of the longest segment in the batch. Sherpa-ONNX handles this internally within the OfflineStream management, but awareness of this behavior is important for understanding memory spikes.4.4 Phase 3: Batched ASR InferenceThis phase performs the actual speech-to-text conversion using the OfflineRecognizer.4.4.1 Stream ManagementThe OfflineRecognizer utilizes the concept of "Streams." For a batch of $N$ audio segments, the application creates $N$ OfflineStream objects.Pythonstreams = [recognizer.create_stream() for _ in batch_of_segments]
Each stream is fed its respective audio waveform.4.4.2 Parallel DecodingThe critical call is recognizer.decode_streams(streams). This function triggers the batched forward pass through the Parakeet-TDT neural network on the GPU. The TDT model processes all streams concurrently, predicting tokens and durations for the entire batch. This is where the massive parallelism of the GPU provides a throughput advantage over sequential CPU decoding.4.5 Phase 4: Post-Processing and Output GenerationThe raw text from the ASR engine is retrieved from each stream.Merging: The text is paired back with the speaker ID and timestamps from Phase 1.Punctuation: The OfflinePunctuation class (wrapping the CT-Transformer) takes the raw text and inserts commas, periods, and question marks. This model is computationally lightweight and can typically run on the CPU without bottlenecking the pipeline.Formatting: The final output is structured into a standard format, such as a JSON transcript or a WebVTT/SRT subtitle file, ready for consumption by downstream applications.5. Performance Engineering and BenchmarkingBuilding a functional pipeline is only the first step; optimizing it for production throughput requires a deep understanding of performance metrics and resource constraints.5.1 Real-Time Factor (RTF) AnalysisThe Real-Time Factor (RTF) measures the speed of the system: $RTF = \frac{\text{Processing Time}}{\text{Audio Duration}}$. An RTF of 0.1 means 10 hours of audio can be processed in 1 hour.Parakeet-TDT Advantage: Benchmarks indicate that the Parakeet-TDT model achieves an RTF of roughly 0.015 (approx. 67x faster than real-time) on localized hardware for the ASR component alone. When combined with diarization, end-to-end RTFs of 0.03 - 0.05 are achievable on modern GPUs like the NVIDIA A100 or RTX 4090.TDT vs. RNN-T: The frame-skipping capability of the TDT architecture is the primary driver of this speed. By predicting duration, the decoder avoids the redundant "blank" token emissions that plague standard RNN-T models during silences or steady vowels.5.2 GPU Memory ProfilingEfficient batching is bounded by Video RAM (VRAM).Quantization: Using int8 quantized models is mandatory for high-throughput batching. The int8 Parakeet model occupies ~640MB of static VRAM.Dynamic Usage: The runtime VRAM usage scales with batch_size $\times$ max_segment_length. A batch of 20 segments, each 10 seconds long, might consume 4-6GB of VRAM. It is crucial to monitor VRAM usage to prevent Out-Of-Memory (OOM) crashes. If a specific diarization segment is exceptionally long (e.g., >60 seconds), it should be split or processed in a smaller batch.5.3 VAD Tuning for Meeting AcousticsThe quality of the diarization (and consequently the ASR) depends heavily on the Voice Activity Detection settings. The Pyannote pipeline utilizes VAD implicitly, but Sherpa-ONNX allows tuning via the silero_vad configuration.Thresholding: A standard threshold of 0.5 is a good baseline. However, in noisy meeting rooms, increasing this to 0.6 or 0.7 prevents background noise (e.g., HVAC, typing) from being transcribed as hallucinations. Conversely, for quiet recordings, lowering it to 0.2 ensures detecting soft-spoken participants.Min Duration: Setting min_speech_duration to 0.25s or higher filters out short, non-speech bursts (coughs, mic bumps) that waste compute cycles.6. Comparison of Models and ArchitecturesTo contextualize the selection of Parakeet-TDT, we present a comparative analysis of supported architectures within the Sherpa-ONNX ecosystem.FeatureParakeet-TDT (v3)Zipformer (Transducer)Whisper (OpenAI)SenseVoiceArchitectureFastConformer + TDTZipformer + RNN-TTransformer (Enc-Dec)FunASR (SenseVoice)Decoding StyleFrame-skippingFrame-by-frameAutoregressiveNon-autoregressiveInference SpeedFastest (RTF ~0.015)Fast (RTF ~0.03)Slow (RTF ~0.1 - 0.2)Very FastQuantizationInt8 / FP16Int8 / FP16Int8 / FP16Int8MultilingualYes (25 langs)Model-dependentYes (99+ langs)Yes (5 langs)Timestamp PrecisionHigh (Token-level)High (Token-level)Low (Segment-level)HighRich MetadataText onlyText onlyText onlyEmotion & Event TagsTable 1: Comparison of ASR architectures supported by Sherpa-ONNX.While SenseVoice is a compelling emerging alternative due to its ability to detect emotions (e.g., <|HAPPY|>, <|ANGRY|>) and acoustic events (e.g., <|LAUGHTER|>), Parakeet-TDT remains the superior choice for pure transcription throughput and broad European language support in strict offline corporate settings where emotion detection is secondary to verbatim accuracy.7. Python Implementation GuideThe following Python script demonstrates the integration of the components discussed. It assumes the directory structure and models setup in the Dockerfile.Pythonimport sherpa_onnx
import wave
import numpy as np
import sys

def read_wave(wave_filename):
    """
    Reads a WAV file and returns samples as a normalized float32 numpy array.
    Ensures the sample rate is 16000Hz.
    """
    with wave.open(wave_filename, "rb") as wf:
        if wf.getnchannels()!= 1:
            raise ValueError("Audio must be mono")
        if wf.getframerate()!= 16000:
            raise ValueError("Audio must be 16k sample rate")
        
        num_samples = wf.getnframes()
        samples = wf.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768.0
        return samples_float32

def main():
    if len(sys.argv)!= 2:
        print("Usage: python3 app.py <input.wav>")
        sys.exit(1)
        
    wave_file = sys.argv[1]
    audio_data = read_wave(wave_file)
    
    # -------------------------------------------------------------------------
    # Step 1: Speaker Diarization
    # -------------------------------------------------------------------------
    print("--- Starting Diarization ---")
    diar_config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model="/models/model.onnx" # Segmentation model
            )
        ),
        embedding=sherpa_onnx.OfflineSpeakerEmbeddingExtractorConfig(
            model="/models/wespeaker_en_voxceleb_resnet34.onnx"
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=-1  # Auto-detect number of speakers
        )
    )
    
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(diar_config)
    
    # Process the entire audio file to get segments
    diar_segments = diarizer.process(audio_data)
    
    # Sort segments by start time
    diar_segments.sort(key=lambda x: x.start)
    
    print(f"Detected {len(diar_segments)} segments.")

    # -------------------------------------------------------------------------
    # Step 2: Batched ASR
    # -------------------------------------------------------------------------
    print("--- Starting ASR ---")
    asr_config = sherpa_onnx.OfflineRecognizerConfig(
        model_config=sherpa_onnx.OfflineModelConfig(
            transducer=sherpa_onnx.OfflineTransducerModelConfig(
                encoder_filename="/models/encoder.int8.onnx",
                decoder_filename="/models/decoder.int8.onnx",
                joiner_filename="/models/joiner.int8.onnx",
            ),
            tokens="/models/tokens.txt",
            provider="cuda", 
            num_threads=2,
            debug=False
        )
    )
    recognizer = sherpa_onnx.OfflineRecognizer(asr_config)
    
    # Prepare batching
    BATCH_SIZE = 10
    final_transcript =
    
    # Iterate through segments in batches
    for i in range(0, len(diar_segments), BATCH_SIZE):
        batch_segments = diar_segments
        streams =
        
        # Create streams for the current batch
        for segment in batch_segments:
            start_sample = int(segment.start * 16000)
            end_sample = int(segment.end * 16000)
            
            # Extract audio for this segment
            # Note: Boundary checks omitted for brevity
            segment_audio = audio_data[start_sample:end_sample]
            
            s = recognizer.create_stream()
            s.accept_waveform(16000, segment_audio)
            streams.append(s)
        
        # Parallel Decode
        if streams:
            recognizer.decode_streams(streams)
        
        # Collect results
        for idx, stream in enumerate(streams):
            text = stream.result.text
            if text.strip():
                final_transcript.append({
                    "start": batch_segments[idx].start,
                    "end": batch_segments[idx].end,
                    "speaker": f"Speaker {batch_segments[idx].speaker}",
                    "text": text
                })

    # -------------------------------------------------------------------------
    # Step 3: Punctuation (Post-Processing)
    # -------------------------------------------------------------------------
    print("--- Adding Punctuation ---")
    punct_config = sherpa_onnx.OfflinePunctuationConfig(
        model=sherpa_onnx.OfflinePunctuationModelConfig(
            ct_transformer="/models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/model.int8.onnx"
        )
    )
    punct = sherpa_onnx.OfflinePunctuation(punct_config)
    
    for entry in final_transcript:
        entry["text"] = punct.add_punctuation(entry["text"])
        print(f"[{entry['start']:.2f} - {entry['end']:.2f}] {entry['speaker']}: {entry['text']}")

if __name__ == "__main__":
    main()
8. ConclusionThe architecture presented herein leverages the Sherpa-ONNX framework to construct a high-fidelity, offline meeting transcription service. By utilizing Parakeet-TDT for its breakthrough speed and Pyannote for its robust diarization, the system achieves a level of performance that rivals cloud-based APIs while operating entirely within a local, containerized environment.The success of this deployment hinges on the rigorous management of the CUDA/ONNX Runtime dependency chain and the implementation of intelligent batching strategies. The move towards TDT architectures and quantized inference models signifies a maturation of open-source speech technology, enabling engineers to build sophisticated, privacy-preserving speech intelligence systems on commodity hardware. As models like SenseVoice continue to evolve, integrating multimodal capabilities (emotion, event detection) into this pipeline represents the next frontier for rich meeting analysis.Citations: