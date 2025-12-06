#!/usr/bin/env python3
"""
Test đơn giản Speech-to-Text với Whisper
Chỉ cần faster-whisper, không cần RealtimeSTT
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except ImportError:
    logger.error("❌ Missing faster-whisper. Install: pip install faster-whisper")
    sys.exit(1)


def test_stt_with_text(text_input: str, model_size: str = "medium"):
    """
    Test STT bằng cách generate audio từ text (dùng TTS) rồi transcribe lại.
    Cách này giúp test trên server không cần mic.
    """
    logger.info(f"📝 Input text: {text_input}")
    logger.info(f"⏳ Loading Whisper model: {model_size}...")
    
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )
    logger.info("✅ Model loaded!")
    
    # Tạo audio từ text bằng Piper TTS
    logger.info("🔊 Generating audio from text using Piper TTS...")
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from piper_audio_module import PiperTTSWrapper
        
        tts = PiperTTSWrapper(
            model_path="../models/piper/vi_VN-vais1000-medium.onnx"
        )
        
        # Generate audio
        audio_data = tts.synthesize(text_input)
        logger.info(f"✅ Audio generated: {len(audio_data)} bytes")
        
        # Save to temp file
        import tempfile
        import soundfile as sf
        import numpy as np
        
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_np, tts.sample_rate)
        logger.info(f"💾 Saved to: {temp_file.name}")
        
        # Transcribe
        logger.info("🔄 Transcribing audio back to text...")
        segments, info = model.transcribe(
            temp_file.name,
            language="vi",
            beam_size=5,
            vad_filter=True,
            initial_prompt="Xin chào, tôi đang nói tiếng Việt.",
        )
        
        transcribed_text = " ".join([seg.text.strip() for seg in segments])
        
        print("\n" + "="*60)
        print("📊 STT TEST RESULT:")
        print("="*60)
        print(f"📝 Original text:    {text_input}")
        print(f"🎤 Transcribed text: {transcribed_text}")
        print(f"✅ Match: {text_input.lower() == transcribed_text.lower()}")
        print("="*60)
        
        # Cleanup
        import os
        os.unlink(temp_file.name)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_stt_with_file(audio_file: str, model_size: str = "medium"):
    """Test STT với audio file có sẵn"""
    logger.info(f"📁 Audio file: {audio_file}")
    logger.info(f"⏳ Loading Whisper model: {model_size}...")
    
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )
    logger.info("✅ Model loaded!")
    
    logger.info("🔄 Transcribing...")
    segments, info = model.transcribe(
        audio_file,
        language="vi",
        beam_size=5,
        vad_filter=True,
        initial_prompt="Xin chào, tôi đang nói tiếng Việt.",
    )
    
    print("\n" + "="*60)
    print("📄 TRANSCRIPTION:")
    print("="*60)
    
    for segment in segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text.strip()}")
    
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test STT đơn giản")
    parser.add_argument(
        "--mode",
        choices=["text", "file"],
        default="text",
        help="Test mode: text (TTS->STT), file (từ audio file)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Xin chào, hôm nay trời đẹp quá nhỉ?",
        help="Text để test (chỉ dùng với mode=text)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Audio file path (chỉ dùng với mode=file)"
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="medium",
        help="Whisper model size"
    )
    
    args = parser.parse_args()
    
    if args.mode == "text":
        test_stt_with_text(args.text, args.model)
    elif args.mode == "file":
        if not args.file:
            logger.error("❌ --file is required for file mode")
            sys.exit(1)
        test_stt_with_file(args.file, args.model)

