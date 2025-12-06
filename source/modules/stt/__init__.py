"""Speech-to-Text modules"""
# Import RealtimeSTT with faster-whisper (same as original source code)
from .transcribe import TranscriptionProcessor

# Alternative: OpenAI Whisper API (no VAD, transcribes all audio including background noise)
# from .transcribe_openai import TranscriptionProcessor

__all__ = ['TranscriptionProcessor']

