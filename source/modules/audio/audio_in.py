# audio_in.py - Audio Input Processor
"""
Manages audio input, processes it for transcription, and handles callbacks.
"""

import asyncio
import logging
from typing import Optional, Callable

import numpy as np
from scipy.signal import resample_poly

from modules.stt import TranscriptionProcessor  # Using PhoWhisper + Silero VAD for Vietnamese

logger = logging.getLogger(__name__)


class AudioInputProcessor:
    """
    Manages audio input processing for real-time transcription.
    
    Receives raw audio chunks, resamples them to 16kHz,
    and feeds them to the TranscriptionProcessor.
    """

    _RESAMPLE_RATIO = 3  # 48kHz -> 16kHz

    def __init__(
        self,
        language: str = "vi",
        is_orpheus: bool = False,
        silence_active_callback: Optional[Callable[[bool], None]] = None,
        pipeline_latency: float = 0.5,
    ) -> None:
        """
        Initialize the AudioInputProcessor.
        
        Args:
            language: Target language code (default: "vi" for Vietnamese).
            is_orpheus: Flag for specific model variant.
            silence_active_callback: Callback for silence state changes.
            pipeline_latency: Estimated pipeline latency in seconds.
        """
        self.last_partial_text: Optional[str] = None
        self.transcriber = TranscriptionProcessor(
            language,
            on_recording_start_callback=self._on_recording_start,
            silence_active_callback=self._silence_active_callback,
            is_orpheus=is_orpheus,
            pipeline_latency=pipeline_latency,
        )
        
        self._transcription_failed = False
        self.transcription_task = asyncio.create_task(self._run_transcription_loop())

        self.realtime_callback: Optional[Callable[[str], None]] = None
        self.recording_start_callback: Optional[Callable[[], None]] = None
        self.silence_active_callback: Optional[Callable[[bool], None]] = silence_active_callback
        self.interrupted = False

        self._setup_callbacks()
        logger.info("👂🚀 AudioInputProcessor initialized")

    def _silence_active_callback(self, is_active: bool) -> None:
        """Internal callback for silence detection."""
        if self.silence_active_callback:
            self.silence_active_callback(is_active)

    def _on_recording_start(self) -> None:
        """Internal callback when recording starts."""
        if self.recording_start_callback:
            self.recording_start_callback()

    def abort_generation(self) -> None:
        """Signal the transcriber to abort."""
        logger.info("👂🛑 Aborting generation")
        self.transcriber.abort_generation()

    def _setup_callbacks(self) -> None:
        """Set up callbacks for the TranscriptionProcessor."""
        def partial_transcript_callback(text: str) -> None:
            if text != self.last_partial_text:
                self.last_partial_text = text
                if self.realtime_callback:
                    self.realtime_callback(text)

        self.transcriber.realtime_transcription_callback = partial_transcript_callback

    async def _run_transcription_loop(self) -> None:
        """Run the transcription loop in background."""
        task_name = "TranscriptionTask"
        logger.info(f"👂▶️ Starting transcription task ({task_name})")
        
        while True:
            try:
                await asyncio.to_thread(self.transcriber.transcribe_loop)
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                logger.info(f"👂🚫 Transcription loop cancelled")
                break
            except Exception as e:
                logger.error(f"👂💥 Transcription error: {e}", exc_info=True)
                self._transcription_failed = True
                break
        
        logger.info(f"👂⏹️ Transcription task finished")

    def process_audio_chunk(self, raw_bytes: bytes) -> np.ndarray:
        """
        Convert raw audio bytes to 16kHz 16-bit PCM.
        
        Args:
            raw_bytes: Raw audio data (int16, 48kHz).
            
        Returns:
            Resampled audio as int16 numpy array (16kHz).
        """
        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)

        if np.max(np.abs(raw_audio)) == 0:
            expected_len = int(np.ceil(len(raw_audio) / self._RESAMPLE_RATIO))
            return np.zeros(expected_len, dtype=np.int16)

        audio_float32 = raw_audio.astype(np.float32)
        resampled_float = resample_poly(audio_float32, 1, self._RESAMPLE_RATIO)
        resampled_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)

        return resampled_int16

    async def process_chunk_queue(self, audio_queue: asyncio.Queue) -> None:
        """
        Process audio chunks from queue.
        
        Args:
            audio_queue: Queue containing audio data dictionaries.
        """
        logger.info("👂▶️ Starting audio chunk processing")
        
        while True:
            try:
                if self._transcription_failed:
                    logger.error("👂🛑 Transcription failed, stopping")
                    break

                if self.transcription_task and self.transcription_task.done():
                    task_exception = self.transcription_task.exception()
                    if task_exception and not isinstance(task_exception, asyncio.CancelledError):
                        logger.error(f"👂🛑 Transcription task error: {task_exception}")
                        self._transcription_failed = True
                        break
                    else:
                        logger.warning("👂⏹️ Transcription task finished")
                        break

                audio_data = await audio_queue.get()
                if audio_data is None:
                    logger.info("👂🔌 Received termination signal")
                    break

                pcm_data = audio_data.pop("pcm")
                processed = self.process_audio_chunk(pcm_data)
                
                if processed.size == 0:
                    continue

                if not self.interrupted and not self._transcription_failed:
                    self.transcriber.feed_audio(processed.tobytes(), audio_data)

            except asyncio.CancelledError:
                logger.info("👂🚫 Audio processing cancelled")
                break
            except Exception as e:
                logger.error(f"👂💥 Audio processing error: {e}", exc_info=True)
        
        logger.info("👂⏹️ Audio chunk processing finished")

    def shutdown(self) -> None:
        """Shutdown the audio processor."""
        logger.info("👂🛑 Shutting down AudioInputProcessor...")
        
        if hasattr(self.transcriber, 'shutdown'):
            self.transcriber.shutdown()
        
        if self.transcription_task and not self.transcription_task.done():
            self.transcription_task.cancel()
        
        logger.info("👂👋 AudioInputProcessor shutdown complete")

