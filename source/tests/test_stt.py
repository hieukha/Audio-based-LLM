#!/usr/bin/env python3
"""
Test Speech-to-Text (STT) với RealtimeSTT + Whisper
Hỗ trợ cả microphone và file audio
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    from RealtimeSTT import AudioToTextRecorder
    import numpy as np
    import soundfile as sf
except ImportError as e:
    logger.error(f"❌ Missing dependency: {e}")
    logger.error("Run: pip install RealtimeSTT soundfile numpy")
    sys.exit(1)


class STTTester:
    """Test Speech-to-Text với Whisper"""
    
    def __init__(self, model_size="medium", language="vi"):
        """
        Initialize STT tester.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Language code (vi for Vietnamese)
        """
        self.model_size = model_size
        self.language = language
        self.recorder = None
        
        logger.info(f"🎤 Initializing STT with model: {model_size}, language: {language}")
        
    def test_with_microphone(self):
        """Test STT với microphone real-time"""
        logger.info("🎙️ Starting microphone test...")
        logger.info("💡 Speak into your microphone. Press Ctrl+C to stop.")
        
        config = {
            "model": self.model_size,
            "language": self.language,
            "silero_sensitivity": 0.05,
            "webrtc_sensitivity": 3,
            "post_speech_silence_duration": 0.7,
            "min_length_of_recording": 0.5,
            "enable_realtime_transcription": True,
            "realtime_processing_pause": 0.03,
            "beam_size": 5,
            "initial_prompt": "Xin chào, tôi đang nói tiếng Việt.",
            "device": "cpu",
            "compute_type": "int8",
            "no_log_file": True,
        }
        
        def on_realtime(text):
            """Callback cho realtime transcription"""
            print(f"\r🔄 [Realtime]: {text}", end='', flush=True)
        
        def on_final(text):
            """Callback cho final transcription"""
            print(f"\n✅ [Final]: {text}")
            print("-" * 50)
        
        config["realtime_model_callback"] = on_realtime
        config["on_transcription_finished"] = on_final
        
        try:
            logger.info("⏳ Loading Whisper model...")
            self.recorder = AudioToTextRecorder(**config)
            logger.info("✅ Model loaded successfully!")
            logger.info("🎙️ Recording started. Speak now!")
            
            self.recorder.text()  # Block until stopped
            
        except KeyboardInterrupt:
            logger.info("\n⏹️ Stopped by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            raise
        finally:
            if self.recorder:
                self.recorder.shutdown()
    
    def test_with_file(self, audio_file: str):
        """
        Test STT với audio file.
        
        Args:
            audio_file: Path to audio file (wav, mp3, etc.)
        """
        audio_path = Path(audio_file)
        if not audio_path.exists():
            logger.error(f"❌ File not found: {audio_file}")
            return
        
        logger.info(f"📁 Loading audio file: {audio_file}")
        
        try:
            # Load audio file
            audio_data, sample_rate = sf.read(audio_file)
            logger.info(f"📊 Sample rate: {sample_rate} Hz, Duration: {len(audio_data)/sample_rate:.2f}s")
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Normalize to int16
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # Initialize recorder
            logger.info("⏳ Loading Whisper model...")
            from faster_whisper import WhisperModel
            
            model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8"
            )
            logger.info("✅ Model loaded!")
            
            # Transcribe
            logger.info("🔄 Transcribing...")
            segments, info = model.transcribe(
                audio_data,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt="Xin chào, tôi đang nói tiếng Việt.",
            )
            
            logger.info(f"📝 Detected language: {info.language} (probability: {info.language_probability:.2%})")
            print("\n" + "="*50)
            print("📄 TRANSCRIPTION RESULT:")
            print("="*50)
            
            full_text = []
            for i, segment in enumerate(segments, 1):
                text = segment.text.strip()
                full_text.append(text)
                print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}")
            
            print("="*50)
            print("\n✅ FULL TEXT:")
            print(" ".join(full_text))
            print("="*50)
            
        except Exception as e:
            logger.error(f"❌ Error processing file: {e}")
            raise
    
    def test_batch_files(self, audio_dir: str):
        """
        Test STT với nhiều file trong folder.
        
        Args:
            audio_dir: Path to directory containing audio files
        """
        dir_path = Path(audio_dir)
        if not dir_path.exists():
            logger.error(f"❌ Directory not found: {audio_dir}")
            return
        
        audio_files = list(dir_path.glob("*.wav")) + list(dir_path.glob("*.mp3"))
        if not audio_files:
            logger.error(f"❌ No audio files found in: {audio_dir}")
            return
        
        logger.info(f"📁 Found {len(audio_files)} audio files")
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
            logger.info(f"{'='*50}")
            self.test_with_file(str(audio_file))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test Speech-to-Text with Whisper")
    parser.add_argument(
        "--mode",
        choices=["mic", "file", "batch"],
        default="mic",
        help="Test mode: mic (microphone), file (single file), batch (directory)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Audio file path (for file mode)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Audio directory path (for batch mode)"
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="medium",
        help="Whisper model size"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="vi",
        help="Language code (default: vi for Vietnamese)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "file" and not args.file:
        logger.error("❌ --file is required for file mode")
        sys.exit(1)
    if args.mode == "batch" and not args.dir:
        logger.error("❌ --dir is required for batch mode")
        sys.exit(1)
    
    # Create tester
    tester = STTTester(model_size=args.model, language=args.language)
    
    # Run test
    try:
        if args.mode == "mic":
            tester.test_with_microphone()
        elif args.mode == "file":
            tester.test_with_file(args.file)
        elif args.mode == "batch":
            tester.test_batch_files(args.dir)
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

