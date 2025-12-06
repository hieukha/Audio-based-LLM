# piper_audio_module.py - Piper TTS Audio Processor
"""
Text-to-Speech module using Piper via RealtimeTTS for Vietnamese voice synthesis.
Based on RealtimeVoiceChat AudioProcessor with Piper engine.
"""

import asyncio
import logging
import os
import struct
import threading
import time
from collections import namedtuple
from queue import Queue, Full
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import RealtimeTTS
try:
    from RealtimeTTS import TextToAudioStream
    from RealtimeTTS.engines.base_engine import BaseEngine
    REALTIMETTS_AVAILABLE = True
except ImportError as e:
    REALTIMETTS_AVAILABLE = False
    logger.warning(f"🔊⚠️ RealtimeTTS not installed: {e}")

import wave
import tempfile
import subprocess
import pyaudio
from typing import Optional as Opt
from queue import Queue as StdQueue


class PiperVoice:
    """Piper voice configuration for custom engine."""
    def __init__(self, model_file: str, config_file: Opt[str] = None):
        self.model_file = model_file
        if config_file is None:
            possible_json = f"{model_file}.json"
            self.config_file = possible_json if os.path.isfile(possible_json) else None
        else:
            self.config_file = config_file


class CustomPiperEngine(BaseEngine):
    """
    Custom Piper TTS engine that supports 22050 Hz sample rate.
    Based on RealtimeTTS PiperEngine but with flexible sample rate.
    """
    
    def __init__(self, 
                 piper_path: Opt[str] = None,
                 voice: Opt[PiperVoice] = None,
                 sample_rate: int = 22050,
                 debug: bool = False):
        """
        Initialize custom Piper engine.
        
        Args:
            piper_path: Path to piper binary
            voice: PiperVoice configuration
            sample_rate: Expected output sample rate (default 22050 for Vietnamese)
            debug: Enable debug logging
        """
        if piper_path is None:
            env_path = os.environ.get("PIPER_PATH")
            self.piper_path = env_path if env_path else "piper"
        else:
            self.piper_path = piper_path
        
        self.voice = voice
        self.sample_rate = sample_rate
        self.debug = debug
        self.queue = StdQueue()
        self.post_init()
    
    def post_init(self):
        self.engine_name = "piper"
    
    def get_stream_info(self):
        """Returns PyAudio stream configuration."""
        return pyaudio.paInt16, 1, self.sample_rate
    
    def synthesize(self, text: str) -> bool:
        """Synthesize text to audio using Piper binary."""
        if not self.voice:
            logger.error("🔊❌ No voice set for Piper")
            return False
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_wav_path = tmp.name
        
        # Build command
        cmd_list = [
            self.piper_path,
            "-m", self.voice.model_file,
            "-f", output_wav_path
        ]
        
        if self.voice.config_file:
            cmd_list.extend(["-c", self.voice.config_file])
        
        if self.debug:
            logger.debug(f"🔊🔧 Piper command: {cmd_list}")
        
        try:
            result = subprocess.run(
                cmd_list,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                shell=False
            )
            
            # Read WAV file
            with wave.open(output_wav_path, "rb") as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                width = wf.getsampwidth()
                
                if self.debug:
                    logger.debug(f"🔊📊 WAV: channels={channels}, rate={rate}, width={width}")
                
                # Flexible sample rate check
                if channels != 1 or width != 2:
                    logger.warning(f"🔊⚠️ Unexpected WAV: channels={channels}, width={width}")
                
                # Update sample rate if different
                if rate != self.sample_rate:
                    logger.info(f"🔊📊 Sample rate: {rate} Hz (expected {self.sample_rate})")
                    self.sample_rate = rate
                
                audio_data = wf.readframes(wf.getnframes())
                self.queue.put(audio_data)
            
            return True
            
        except FileNotFoundError:
            logger.error(f"🔊❌ Piper not found: {self.piper_path}")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"🔊❌ Piper error: {e.stderr.decode('utf-8', errors='replace')}")
            return False
        finally:
            if os.path.isfile(output_wav_path):
                os.remove(output_wav_path)
    
    def set_voice(self, voice: PiperVoice):
        self.voice = voice
    
    def get_voices(self):
        return []

# Configuration constants (same as source code)
Silence = namedtuple("Silence", ("comma", "sentence", "default"))
ENGINE_SILENCES = {
    "piper": Silence(comma=0.3, sentence=0.6, default=0.3),
}
QUICK_ANSWER_STREAM_CHUNK_SIZE = 8
FINAL_ANSWER_STREAM_CHUNK_SIZE = 30


class PiperAudioProcessor:
    """
    Text-to-Speech processor using Piper engine via RealtimeTTS.
    Based on RealtimeVoiceChat AudioProcessor - EXACT same logic, just Piper engine.
    """
    
    # Audio format constants
    SAMPLE_RATE = 22050  # Piper default
    BYTES_PER_SAMPLE = 2  # 16-bit
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        models_dir: str = "models/piper",
    ):
        """
        Initialize Piper TTS processor with RealtimeTTS.
        
        Args:
            model_path: Path to custom Piper .onnx model. If None, auto-detect.
            models_dir: Directory containing Piper models.
        """
        if not REALTIMETTS_AVAILABLE:
            raise ImportError("RealtimeTTS required. Install with: pip install RealtimeTTS")
        
        self.engine_name = "piper"
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.audio_chunks = asyncio.Queue()
        
        self.silence = ENGINE_SILENCES["piper"]
        self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE
        
        # Find model
        if not model_path:
            model_path = self._find_model(Path(models_dir))
        
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Piper model not found in {models_dir}")
        
        logger.info(f"🔊📂 Loading Piper model: {model_path}")
        
        # Create PiperVoice with custom model path
        # PiperVoice from RealtimeTTS expects model_file (.onnx) and optional config_file (.json)
        voice = PiperVoice(model_file=str(model_path))
        
        # Find piper binary path (same directory as models)
        piper_dir = Path(models_dir).resolve()
        piper_binary = piper_dir / "piper"
        if not piper_binary.exists():
            # Try absolute path from project root
            piper_dir = Path(__file__).parent.parent.parent.parent / "models" / "piper"
            piper_binary = piper_dir / "piper"
        
        if piper_binary.exists():
            logger.info(f"🔊✅ Found Piper binary: {piper_binary}")
            
            # Add piper library path to LD_LIBRARY_PATH for shared libraries
            piper_lib_path = str(piper_dir)
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if piper_lib_path not in current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{piper_lib_path}:{current_ld_path}"
                logger.info(f"🔊📚 Added to LD_LIBRARY_PATH: {piper_lib_path}")
            
            # Use custom engine that supports 22050 Hz sample rate
            self.engine = CustomPiperEngine(
                piper_path=str(piper_binary),
                voice=voice,
                sample_rate=self.SAMPLE_RATE  # 22050 Hz
            )
        else:
            logger.warning(f"🔊⚠️ Piper binary not found at {piper_binary}, using default path")
            self.engine = CustomPiperEngine(voice=voice, sample_rate=self.SAMPLE_RATE)
        
        # Initialize RealtimeTTS stream (EXACT same as source code)
        self.stream = TextToAudioStream(
            self.engine,
            muted=True,
            playout_chunk_size=4096,
            on_audio_stream_stop=self.on_audio_stream_stop,
        )
        
        # Prewarm (EXACT same as source code)
        self.stream.feed("prewarm")
        play_kwargs = dict(
            log_synthesized_text=False,
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,
        )
        self.stream.play(**play_kwargs)
        while self.stream.is_playing():
            time.sleep(0.01)
        self.finished_event.wait()
        self.finished_event.clear()
        
        # Measure TTFA (EXACT same as source code)
        start_time = time.time()
        ttfa = None
        
        def on_audio_chunk_ttfa(chunk: bytes):
            nonlocal ttfa
            if ttfa is None:
                ttfa = time.time() - start_time
        
        self.stream.feed("This is a test sentence to measure the time to first audio chunk.")
        play_kwargs_ttfa = dict(
            on_audio_chunk=on_audio_chunk_ttfa,
            log_synthesized_text=False,
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,
        )
        self.stream.play_async(**play_kwargs_ttfa)
        
        while ttfa is None and (self.stream.is_playing() or not self.finished_event.is_set()):
            time.sleep(0.01)
        self.stream.stop()
        
        if not self.finished_event.is_set():
            self.finished_event.wait(timeout=2.0)
        self.finished_event.clear()
        
        if ttfa is not None:
            self.tts_inference_time = ttfa * 1000
        else:
            logger.warning("👄⚠️ TTFA measurement failed")
            self.tts_inference_time = 0
        
        # Callback
        self.on_first_audio_chunk_synthesize: Optional[Callable[[], None]] = None
        
        # Track chunk timing
        self._quick_prev_chunk_time: float = 0.0
        self._final_prev_chunk_time: float = 0.0
    
    def _find_model(self, models_dir: Path) -> Optional[str]:
        """Find Vietnamese Piper model."""
        if not models_dir.exists():
            return None
        
        patterns = ["vi_VN*.onnx", "vi-*.onnx", "vietnamese*.onnx"]
        for pattern in patterns:
            models = list(models_dir.glob(pattern))
            if models:
                return str(models[0])
        
        # Fallback: any .onnx
        models = list(models_dir.glob("*.onnx"))
        return str(models[0]) if models else None
    
    def on_audio_stream_stop(self) -> None:
        """Callback when audio stream stops (EXACT same as source code)."""
        logger.info("👄🛑 Audio stream stopped.")
        self.finished_event.set()
    
    def synthesize(
        self,
        text: str,
        audio_chunks: Queue,
        stop_event: threading.Event,
        generation_string: str = "",
    ) -> bool:
        """
        Synthesize text and put chunks into queue.
        EXACT SAME LOGIC as source code audio_module.py synthesize().
        """
        self.stream.feed(text)
        self.finished_event.clear()
        
        # Buffering state (EXACT same as source code)
        buffer: list[bytes] = []
        good_streak: int = 0
        buffering: bool = True
        buf_dur: float = 0.0
        SR, BPS = 24000, 2  # Will be overridden by engine sample rate
        start = time.time()
        self._quick_prev_chunk_time = 0.0
        
        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur, start
            
            if stop_event.is_set():
                logger.info(f"👄🛑 {generation_string} Quick interrupted")
                return
            
            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR
            
            # Timing and logging (EXACT same as source code)
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._quick_prev_chunk_time = now
                ttfa_actual = now - start
                logger.info(f"👄🚀 {generation_string} Quick audio start. TTFA: {ttfa_actual:.2f}s. Text: {text[:50]}...")
            else:
                gap = now - self._quick_prev_chunk_time
                self._quick_prev_chunk_time = now
                if gap <= play_duration * 1.1:
                    good_streak += 1
                else:
                    logger.warning(f"👄❌ {generation_string} Quick chunk slow (gap={gap:.3f}s)")
                    good_streak = 0
            
            put_occurred_this_call = False
            
            # Buffering logic (EXACT same as source code)
            buffer.append(chunk)
            buf_dur += play_duration
            
            if buffering:
                if good_streak >= 2 or buf_dur >= 0.5:
                    logger.info(f"👄➡️ {generation_string} Quick Flushing buffer (streak={good_streak}, dur={buf_dur:.2f}s, chunks={len(buffer)})")
                    for c in buffer:
                        try:
                            logger.info(f"👄🔊 {generation_string} Quick Putting {len(c)} bytes into queue (qsize before: {audio_chunks.qsize()})")
                            audio_chunks.put_nowait(c)
                            logger.info(f"👄✅ {generation_string} Quick Put success (qsize after: {audio_chunks.qsize()})")
                            put_occurred_this_call = True
                        except Full:
                            logger.warning(f"👄⚠️ {generation_string} Quick queue full")
                    buffer.clear()
                    buf_dur = 0.0
                    buffering = False
            else:
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except Full:
                    logger.warning(f"👄⚠️ {generation_string} Quick queue full")
            
            # First chunk callback (EXACT same as source code)
            if put_occurred_this_call and not on_audio_chunk.callback_fired:
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.info(f"👄🚀 {generation_string} Quick Firing on_first_audio_chunk_synthesize")
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(f"👄💥 {generation_string} Callback error: {e}", exc_info=True)
                on_audio_chunk.callback_fired = True
        
        # Initialize callback state
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False
        
        play_kwargs = dict(
            log_synthesized_text=True,
            on_audio_chunk=on_audio_chunk,
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,
        )
        
        logger.info(f"👄▶️ {generation_string} Quick Starting synthesis. Text: {text[:50]}...")
        self.stream.play_async(**play_kwargs)
        
        # Wait loop (EXACT same as source code)
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.info(f"👄🛑 {generation_string} Quick aborted")
                buffer.clear()
                self.finished_event.wait(timeout=1.0)
                return False
            time.sleep(0.01)
        
        # Flush remaining buffer (EXACT same as source code)
        if buffering and buffer and not stop_event.is_set():
            logger.info(f"👄➡️ {generation_string} Quick Flushing remaining buffer")
            for c in buffer:
                try:
                    audio_chunks.put_nowait(c)
                except Full:
                    logger.warning(f"👄⚠️ {generation_string} Quick queue full on final flush")
            buffer.clear()
        
        logger.info(f"👄✅ {generation_string} Quick answer synthesis complete")
        return True
    
    def synthesize_generator(
        self,
        generator: Generator[str, None, None],
        audio_chunks: Queue,
        stop_event: threading.Event,
        generation_string: str = "",
    ) -> bool:
        """
        Synthesize from text generator and put chunks into queue.
        EXACT SAME LOGIC as source code audio_module.py synthesize_generator().
        """
        # Feed generator to stream
        self.stream.feed(generator)
        self.finished_event.clear()
        
        # Buffering state (EXACT same as source code)
        buffer: list[bytes] = []
        good_streak: int = 0
        buffering: bool = True
        buf_dur: float = 0.0
        SR, BPS = 24000, 2
        start = time.time()
        self._final_prev_chunk_time = 0.0
        
        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur, start
            
            if stop_event.is_set():
                logger.info(f"👄🛑 {generation_string} Final interrupted")
                return
            
            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR
            
            # Timing and logging (EXACT same as source code)
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._final_prev_chunk_time = now
                ttfa_actual = now - start
                logger.info(f"👄🚀 {generation_string} Final audio start. TTFA: {ttfa_actual:.2f}s")
            else:
                gap = now - self._final_prev_chunk_time
                self._final_prev_chunk_time = now
                if gap <= play_duration * 1.1:
                    good_streak += 1
                else:
                    logger.warning(f"👄❌ {generation_string} Final chunk slow (gap={gap:.3f}s)")
                    good_streak = 0
            
            put_occurred_this_call = False
            
            # Buffering logic (EXACT same as source code)
            buffer.append(chunk)
            buf_dur += play_duration
            
            if buffering:
                if good_streak >= 2 or buf_dur >= 0.5:
                    logger.info(f"👄➡️ {generation_string} Final Flushing buffer (streak={good_streak}, dur={buf_dur:.2f}s)")
                    for c in buffer:
                        try:
                            audio_chunks.put_nowait(c)
                            put_occurred_this_call = True
                        except Full:
                            logger.warning(f"👄⚠️ {generation_string} Final queue full")
                    buffer.clear()
                    buf_dur = 0.0
                    buffering = False
            else:
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except Full:
                    logger.warning(f"👄⚠️ {generation_string} Final queue full")
            
            # First chunk callback (EXACT same as source code)
            if put_occurred_this_call and not on_audio_chunk.callback_fired:
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.info(f"👄🚀 {generation_string} Final Firing on_first_audio_chunk_synthesize")
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(f"👄💥 {generation_string} Callback error: {e}", exc_info=True)
                on_audio_chunk.callback_fired = True
        
        # Initialize callback state
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False
        
        play_kwargs = dict(
            log_synthesized_text=True,
            on_audio_chunk=on_audio_chunk,
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,
        )
        
        logger.info(f"👄▶️ {generation_string} Final Starting synthesis from generator")
        self.stream.play_async(**play_kwargs)
        
        # Wait loop (EXACT same as source code)
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.info(f"👄🛑 {generation_string} Final aborted")
                buffer.clear()
                self.finished_event.wait(timeout=1.0)
                return False
            time.sleep(0.01)
        
        # Flush remaining buffer (EXACT same as source code)
        if buffering and buffer and not stop_event.is_set():
            logger.info(f"👄➡️ {generation_string} Final Flushing remaining buffer")
            for c in buffer:
                try:
                    audio_chunks.put_nowait(c)
                except Full:
                    logger.warning(f"👄⚠️ {generation_string} Final queue full on final flush")
            buffer.clear()
        
        logger.info(f"👄✅ {generation_string} Final answer synthesis complete")
        return True
    
    def _find_model(self, models_dir: Path) -> Optional[str]:
        """Find Vietnamese Piper model."""
        if not models_dir.exists():
            return None
        
        patterns = ["vi_VN*.onnx", "vi-*.onnx", "vietnamese*.onnx"]
        for pattern in patterns:
            models = list(models_dir.glob(pattern))
            if models:
                return str(models[0])
        
        # Fallback
        models = list(models_dir.glob("*.onnx"))
        return str(models[0]) if models else None
