# transcribe.py - Vietnamese Transcription using RealtimeSTT + faster-whisper
"""
Transcription processor using RealtimeSTT with faster-whisper model.
Based on RealtimeVoiceChat source code with custom model cache directory.
"""

import os
import logging
logger = logging.getLogger(__name__)

# Configure HuggingFace cache directory BEFORE importing any models
_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_current_file))))
HF_CACHE_DIR = os.path.join(_project_root, 'models', 'huggingface')
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Set environment variables for HuggingFace cache
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = os.path.join(HF_CACHE_DIR, 'transformers')
os.environ['HF_DATASETS_CACHE'] = os.path.join(HF_CACHE_DIR, 'datasets')

logger.info(f"👂📁 Model cache directory: {HF_CACHE_DIR}")

from difflib import SequenceMatcher
from utils import Colors
from utils.text_similarity import TextSimilarity
from scipy import signal
import numpy as np
import threading
import textwrap
import torch
import json
import copy
import time
import re
from typing import Optional, Callable, Any, Dict, List

# --- Configuration Flags ---
USE_TURN_DETECTION = False  # Disabled (requires turndetect.py)
START_STT_SERVER = False # Set to True to use the client/server version of RealtimeSTT

# --- Recorder Configuration ---
DEFAULT_RECORDER_CONFIG: Dict[str, Any] = {
    "use_microphone": False,
    "spinner": False,
    "model": "large-v3",  # Whisper Large-v3 - Best accuracy, requires GPU
    "realtime_model_type": "large-v3",  # Use large-v3 for realtime too
    "use_main_model_for_realtime": True,  # Use same model for both
    "language": "vi",  # Vietnamese
    "silero_sensitivity": 0.05,
    "webrtc_sensitivity": 3,
    "post_speech_silence_duration": 0.7,
    "min_length_of_recording": 0.8,  # Increased to avoid noise
    "min_gap_between_recordings": 0,
    "enable_realtime_transcription": True,
    "realtime_processing_pause": 0.03,
    "silero_use_onnx": True,
    "silero_deactivity_detection": True,
    "early_transcription_on_silence": 0,
    "beam_size": 5,
    "beam_size_realtime": 3,
    "no_log_file": True,
    "wake_words": "",  # Disabled
    "wakeword_backend": "pvporcupine",
    "allowed_latency_limit": 500,
    "debug_mode": False,
    "initial_prompt": "Xin chào, tôi đang nói tiếng Việt.",
    "initial_prompt_realtime": "Hôm nay tôi muốn hỏi về...",
    "faster_whisper_vad_filter": False,
    "device": "cuda",  # Use GPU for faster inference
    "compute_type": "float16",  # float16 for GPU (best performance/accuracy trade-off)
}


if START_STT_SERVER:
    from RealtimeSTT import AudioToTextRecorderClient
else:
    from RealtimeSTT import AudioToTextRecorder


INT16_MAX_ABS_VALUE: float = 32768.0
SAMPLE_RATE: int = 16000


class TranscriptionProcessor:
    """
    Manages audio transcription using RealtimeSTT with PhoWhisper for Vietnamese.
    """
    # --- Constants for Silence Monitor Logic ---
    _PIPELINE_RESERVE_TIME_MS: float = 0.02
    _HOT_THRESHOLD_OFFSET_S: float = 0.35
    _MIN_HOT_CONDITION_DURATION_S: float = 0.15
    _TTS_ALLOWANCE_OFFSET_S: float = 0.25
    _MIN_POTENTIAL_END_DETECTION_TIME_MS: float = 0.02
    _SENTENCE_CACHE_MAX_AGE_MS: float = 0.2
    _SENTENCE_CACHE_TRIGGER_COUNT: int = 3


    def __init__(
            self,
            source_language: str = "vi",
            realtime_transcription_callback: Optional[Callable[[str], None]] = None,
            full_transcription_callback: Optional[Callable[[str], None]] = None,
            potential_full_transcription_callback: Optional[Callable[[str], None]] = None,
            silence_active_callback: Optional[Callable[[bool], None]] = None,
            on_recording_start_callback: Optional[Callable[[], None]] = None,
            before_final_sentence: Optional[Callable[[Any, Optional[str]], bool]] = None,
            local: bool = True,
            is_orpheus: bool = False,
            pipeline_latency: float = 0.5,
            recorder_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize TranscriptionProcessor."""
        self.source_language = source_language
        self.realtime_transcription_callback = realtime_transcription_callback
        self.full_transcription_callback = full_transcription_callback
        self.potential_full_transcription_callback = potential_full_transcription_callback
        self.silence_active_callback = silence_active_callback
        self.on_recording_start_callback = on_recording_start_callback
        self.before_final_sentence = before_final_sentence
        self.local = local
        self.is_orpheus = is_orpheus
        self.pipeline_latency = pipeline_latency
        self.recorder: Optional[AudioToTextRecorder] = None
        self.realtime_text: Optional[str] = None
        # Cache structure: [{'text': normalized_text, 'timestamps': [t1, t2, ...]}]
        self.sentence_end_cache: List[Dict[str, Any]] = []
        # Yielded structure: [{'text': normalized_text, 'timestamp': t}]
        self.potential_sentences_yielded: List[Dict[str, Any]] = []
        self.stripped_partial_user_text: str = ""
        self.final_transcription: Optional[str] = None
        self.shutdown_performed: bool = False
        self.silence_time: float = 0.0
        self.silence_active: bool = False
        self.last_audio_copy: Optional[np.ndarray] = None
        
        # Initialize TextSimilarity for sentence comparison
        self.text_similarity = TextSimilarity(focus='end', n_words=5)

        # Use provided config or default
        self.recorder_config = copy.deepcopy(recorder_config if recorder_config else DEFAULT_RECORDER_CONFIG)
        self.recorder_config['language'] = self.source_language

        self._create_recorder()
        self._start_silence_monitor()

    # --- Recorder Parameter Abstraction ---

    def _get_recorder_param(self, param_name: str, default: Any = None) -> Any:
        """Get parameter from recorder."""
        if not self.recorder:
            return default
        if START_STT_SERVER:
            return self.recorder.get_parameter(param_name)
        else:
            return getattr(self.recorder, param_name, default)

    def _set_recorder_param(self, param_name: str, value: Any) -> None:
        """Set parameter on recorder."""
        if not self.recorder:
            return
        if START_STT_SERVER:
            self.recorder.set_parameter(param_name, value)
        else:
            setattr(self.recorder, param_name, value)

    def _is_recorder_recording(self) -> bool:
        """Check if recorder is recording."""
        if not self.recorder:
            return False
        if START_STT_SERVER:
            return self.recorder.get_parameter("is_recording")
        else:
            return getattr(self.recorder, "is_recording", False)

    # --- Silence Monitor ---
    def _start_silence_monitor(self) -> None:
        """Start background thread to monitor silence."""
        def monitor_loop():
            hot = False
            potential_sentence_end_triggered = False  # Flag to prevent log spam
            while not self.shutdown_performed:
                time.sleep(0.02)
                
                speech_end_silence_start = self.silence_time
                
                if self.recorder and speech_end_silence_start is not None and speech_end_silence_start != 0:
                    silence_waiting_time = self._get_recorder_param("post_speech_silence_duration", 0.0)
                    time_since_silence = time.time() - speech_end_silence_start
                    
                    start_hot_condition_time = silence_waiting_time - self._HOT_THRESHOLD_OFFSET_S
                    
                    # Potential sentence end
                    potential_sentence_end_time = max(
                        self._MIN_POTENTIAL_END_DETECTION_TIME_MS,
                        silence_waiting_time - self._HOT_THRESHOLD_OFFSET_S
                    )
                    
                    if time_since_silence > potential_sentence_end_time and not potential_sentence_end_triggered:
                        current_text = self.realtime_text if self.realtime_text else ""
                        logger.info(f"👂🔚 Potential sentence end detected (timed out): {current_text}")
                        # Force yield with ellipses on timeout
                        self.detect_potential_sentence_end(current_text, force_yield=True, force_ellipses=True)
                        potential_sentence_end_triggered = True  # Set flag to prevent spam
                    
                    # TTS allowance
                    tts_allowance_time = silence_waiting_time - self._TTS_ALLOWANCE_OFFSET_S
                    if time_since_silence > tts_allowance_time:
                        pass  # Could trigger TTS here
                    
                    hot_condition_met = time_since_silence > start_hot_condition_time
                    
                    if hot_condition_met and not hot:
                        hot = True
                        if start_hot_condition_time >= self._MIN_HOT_CONDITION_DURATION_S:
                            self._trigger_potential_full_transcription()
                    elif not hot_condition_met and hot:
                        hot = False
                elif hot:
                    hot = False
                    potential_sentence_end_triggered = False  # Reset flag when silence ends

        silence_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        silence_monitor_thread.start()

    def on_new_waiting_time(self, waiting_time: float) -> None:
        """Update post_speech_silence_duration."""
        current_duration = self._get_recorder_param("post_speech_silence_duration")
        if current_duration is not None and abs(current_duration - waiting_time) > 0.01:
            self._set_recorder_param("post_speech_silence_duration", waiting_time)

    def _trigger_potential_full_transcription(self) -> None:
        """Trigger potential full transcription callback."""
        if not self.potential_full_transcription_callback:
            return
        
        text = self.realtime_text
        if not text or text.strip() == "":
            return
        
        stripped = text.strip()
        
        # Check if similar to already yielded
        for entry in self.potential_sentences_yielded:
            if self._are_texts_similar(entry['text'], stripped):
                return
        
        # Add to cache
        self.potential_sentences_yielded.append({
            'text': stripped,
            'time': time.time()
        })
        
        # Clean old cache
        current_time = time.time()
        self.potential_sentences_yielded = [
            entry for entry in self.potential_sentences_yielded
            if current_time - entry['time'] <= 5.0
        ]
        
        try:
            self.potential_full_transcription_callback(stripped)
        except Exception as e:
            logger.error(f"Error in potential_full_transcription_callback: {e}")

    def _are_texts_similar(self, text1: str, text2: str, similarity_threshold: float = 0.96) -> bool:
        """Check text similarity."""
        if not text1 or not text2:
            return False
        similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity > similarity_threshold

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison (lowercase, remove punctuation)."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def is_basically_the_same(self, text1: str, text2: str, similarity_threshold: float = 0.96) -> bool:
        """Check if two texts are highly similar using TextSimilarity."""
        similarity = self.text_similarity.calculate_similarity(text1, text2)
        return similarity > similarity_threshold

    def detect_potential_sentence_end(self, text: Optional[str], force_yield: bool = False, force_ellipses: bool = False) -> None:
        """
        Detect potential sentence endings (same logic as source code).
        
        Args:
            text: The real-time transcription text to check.
            force_yield: If True, bypass timing checks and force trigger.
            force_ellipses: If True, allow "..." as sentence end.
        """
        if not text:
            return

        stripped_text_raw = text.strip()
        if not stripped_text_raw:
            return

        # Don't consider ellipses unless forced
        if stripped_text_raw.endswith("...") and not force_ellipses:
            return

        end_punctuations = [".", "!", "?"]
        now = time.time()

        # Check if ends with punctuation
        ends_with_punctuation = any(stripped_text_raw.endswith(p) for p in end_punctuations)
        if not ends_with_punctuation and not force_yield:
            return

        normalized_text = self._normalize_text(stripped_text_raw)
        if not normalized_text:
            return

        # --- Cache Management (track timestamps per normalized text) ---
        entry_found = None
        for entry in self.sentence_end_cache:
            if self.is_basically_the_same(entry['text'], normalized_text):
                entry_found = entry
                break

        if entry_found:
            entry_found['timestamps'].append(now)
            # Keep only recent timestamps
            entry_found['timestamps'] = [t for t in entry_found['timestamps'] if now - t <= self._SENTENCE_CACHE_MAX_AGE_MS]
        else:
            # Add new entry with timestamps list
            entry_found = {'text': normalized_text, 'timestamps': [now]}
            self.sentence_end_cache.append(entry_found)

        # --- Yielding Logic ---
        should_yield = False
        if force_yield:
            should_yield = True
        # Yield if SAME sentence ending appeared multiple times recently
        elif ends_with_punctuation and len(entry_found['timestamps']) >= self._SENTENCE_CACHE_TRIGGER_COUNT:
            should_yield = True

        if should_yield:
            # Check if already yielded (deduplication)
            already_yielded = False
            for yielded_entry in self.potential_sentences_yielded:
                if self.is_basically_the_same(yielded_entry['text'], normalized_text):
                    already_yielded = True
                    break

            if not already_yielded:
                # Add to yielded list
                self.potential_sentences_yielded.append({'text': normalized_text, 'timestamp': now})
                
                logger.info(f"👂➡️ Yielding potential sentence end: {stripped_text_raw}")
                if self.potential_sentence_end:
                    self.potential_sentence_end(stripped_text_raw)

    def set_silence(self, silence_active: bool) -> None:
        """Update silence state."""
        if self.silence_active != silence_active:
            self.silence_active = silence_active
            if self.silence_active_callback:
                self.silence_active_callback(silence_active)

    def get_last_audio_copy(self) -> Optional[np.ndarray]:
        """Get last audio buffer."""
        audio_copy = self.get_audio_copy()
        if audio_copy is not None and len(audio_copy) > 0:
            return audio_copy
        return self.last_audio_copy

    def get_audio_copy(self) -> Optional[np.ndarray]:
        """Copy current audio buffer from recorder."""
        if not self.recorder:
            return self.last_audio_copy
        
        if not hasattr(self.recorder, 'frames'):
            return self.last_audio_copy
        
        try:
            lock = getattr(self.recorder, 'frames_lock', threading.Lock())
            with lock:
                frames_data = list(self.recorder.frames)
            
            if not frames_data or len(frames_data) == 0:
                return self.last_audio_copy
            
            # Convert to numpy array
            audio_data = b''.join(frames_data)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
            
            full_audio = audio_array.copy()
            
            if len(full_audio) > 0:
                self.last_audio_copy = full_audio
            
            return full_audio
            
        except Exception as e:
            logger.error(f"Error getting audio copy: {e}")
            return self.last_audio_copy

    def transcribe_loop(self) -> None:
        """Run transcription loop."""
        
        def on_final(text: Optional[str]):
            if text is None or text == "":
                return
            
            self.final_transcription = text
            logger.info(f"👂✅ Final: {text}")
            
            if self.full_transcription_callback:
                self.full_transcription_callback(text)
        
        if self.recorder:
            if hasattr(self.recorder, 'text'):
                self.recorder.text(on_final)

    def abort_generation(self) -> None:
        """Clear sentence cache and yielded list."""
        self.sentence_end_cache.clear()
        self.potential_sentences_yielded.clear()
        logger.info("👂⏹️ Potential sentence yield cache cleared (generation aborted)")

    def feed_audio(self, chunk: bytes, audio_meta_data: Optional[Dict[str, Any]] = None) -> None:
        """Feed audio chunk to recorder."""
        if self.recorder and not self.shutdown_performed:
            try:
                if START_STT_SERVER:
                    self.recorder.feed_audio(chunk)
                else:
                    self.recorder.feed_audio(chunk)
            except Exception as e:
                logger.error(f"Error feeding audio: {e}")

    def shutdown(self) -> None:
        """Shutdown processor."""
        if not self.shutdown_performed:
            logger.info("👂🔌 Shutting down...")
            self.shutdown_performed = True
            
            if self.recorder:
                try:
                    if hasattr(self.recorder, 'shutdown'):
                        self.recorder.shutdown()
                    self.recorder = None
                except Exception as e:
                    logger.error(f"Error shutting down recorder: {e}")

    def _create_recorder(self) -> None:
        """Create RealtimeSTT recorder."""
        
        def start_silence_detection():
            """Callback when silence starts."""
            self.set_silence(True)
            recorder_silence_start = self._get_recorder_param("speech_end_silence_start", None)
            self.silence_time = recorder_silence_start if recorder_silence_start else time.time()

        def stop_silence_detection():
            """Callback when silence stops."""
            self.set_silence(False)
            self.silence_time = 0.0

        def start_recording():
            """Callback when recording starts."""
            logger.info("👂▶️ Recording started")
            self.set_silence(False)
            self.silence_time = 0.0
            # Clear yielded cache on new recording to allow re-detection
            self.potential_sentences_yielded.clear()
            if self.on_recording_start_callback:
                self.on_recording_start_callback()

        def stop_recording() -> bool:
            """Callback when recording stops - CRITICAL for TTS!"""
            logger.info("👂⏹️ Recording stopped.")
            audio_copy = self.get_last_audio_copy()
            
            # CRITICAL: Call before_final_sentence callback to enable TTS streaming!
            if self.before_final_sentence:
                logger.info("👂➡️ Calling before_final_sentence callback...")
                try:
                    result = self.before_final_sentence(audio_copy, self.realtime_text)
                    return result if isinstance(result, bool) else False
                except Exception as e:
                    logger.error(f"👂💥 Error in before_final_sentence callback: {e}", exc_info=True)
                    return False
            return False

        def on_partial(text: Optional[str]):
            """Callback for realtime transcription."""
            if text is None:
                return
            
            self.realtime_text = text
            
            # Detect potential sentence ends based on punctuation stability (CRITICAL!)
            self.detect_potential_sentence_end(text)
            
            if self.realtime_transcription_callback:
                try:
                    self.realtime_transcription_callback(text)
                except Exception as e:
                    logger.error(f"Error in realtime_transcription_callback: {e}")

        # Setup config with callbacks
        active_config = self.recorder_config.copy()
        active_config["on_realtime_transcription_update"] = on_partial  # Use _update not _stabilized!
        active_config["on_recording_start"] = start_recording
        active_config["on_recording_stop"] = stop_recording
        active_config["on_turn_detection_start"] = start_silence_detection  # Triggered when silence starts
        active_config["on_turn_detection_stop"] = stop_silence_detection  # Triggered when silence stops

        try:
            logger.info("👂⚙️ Creating AudioToTextRecorder...")
            if START_STT_SERVER:
                self.recorder = AudioToTextRecorderClient(**active_config)
            else:
                self.recorder = AudioToTextRecorder(**active_config)
            
            self.recorder.use_wake_words = False
            logger.info("👂✅ AudioToTextRecorder created successfully")
        except Exception as e:
            logger.exception(f"👂🔥 Failed to create recorder: {e}")
            self.recorder = None


if __name__ == "__main__":
    # Test
    processor = TranscriptionProcessor(source_language="vi")
    print("Transcription processor initialized")
