import base64
import numpy as np
from scipy.signal import resample_poly
from typing import Optional

class UpsampleOverlap:
    """
    Manages chunk-wise audio upsampling with overlap handling.

    This class processes sequential audio chunks, upsamples them from 22.05kHz (Piper) to 48kHz
    using `scipy.signal.resample_poly`, and manages overlap between chunks to
    mitigate boundary artifacts. The processed, upsampled audio segments are
    returned as Base64 encoded strings. It maintains internal state to handle
    the overlap correctly across calls.
    """
    def __init__(self, input_rate: int = 22050, output_rate: int = 48000):
        """
        Initializes the UpsampleOverlap processor.

        Args:
            input_rate: Input sample rate (default 22050 for Piper TTS)
            output_rate: Output sample rate (default 48000 for browser)
        """
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.previous_chunk: Optional[np.ndarray] = None
        self.resampled_previous_chunk: Optional[np.ndarray] = None

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        Processes an incoming audio chunk, upsamples it, and returns the relevant segment as Base64.

        Converts the raw PCM bytes (assumed 16-bit signed integer) chunk to a
        float32 numpy array, normalizes it, and upsamples to target rate.
        It uses the previous chunk's data to create an overlap, resamples the
        combined audio, and extracts the central portion corresponding primarily
        to the current chunk, using overlap to smooth transitions.

        Args:
            chunk: Raw audio data bytes (PCM 16-bit signed integer format expected).

        Returns:
            A Base64 encoded string representing the upsampled audio segment
            corresponding to the input chunk, adjusted for overlap. Returns an
            empty string if the input chunk is empty.
        """
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        # Handle potential empty chunks gracefully
        if audio_int16.size == 0:
             return "" # Return empty string for empty input chunk

        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Upsample the current chunk independently first
        upsampled_current_chunk = resample_poly(audio_float, self.output_rate, self.input_rate)

        if self.previous_chunk is None:
            # First chunk: Output the first half of its upsampled version
            half = len(upsampled_current_chunk) // 2
            part = upsampled_current_chunk[:half]
        else:
            # Subsequent chunks: Combine previous float chunk with current float chunk
            combined = np.concatenate((self.previous_chunk, audio_float))
            # Upsample the combined chunk
            up = resample_poly(combined, self.output_rate, self.input_rate)

            # Calculate lengths and indices for extracting the middle part
            assert self.resampled_previous_chunk is not None
            prev_len = len(self.resampled_previous_chunk)
            h_prev = prev_len // 2

            # Calculate the end index for the part corresponding to the current chunk's main contribution
            h_cur = (len(up) - prev_len) // 2 + prev_len

            part = up[h_prev:h_cur]

        # Update state for the next iteration
        self.previous_chunk = audio_float
        self.resampled_previous_chunk = upsampled_current_chunk

        # Convert the extracted part back to PCM16 bytes and encode
        pcm = (part * 32767).astype(np.int16).tobytes()
        return base64.b64encode(pcm).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        Returns the final remaining segment of upsampled audio after all chunks are processed.

        Returns:
            A Base64 encoded string containing the final upsampled audio chunk,
            or None if no chunks were processed or if flush has already been called.
        """
        if self.resampled_previous_chunk is not None:
            # Return the entire last upsampled chunk
            pcm = (self.resampled_previous_chunk * 32767).astype(np.int16).tobytes()

            # Clear state after flushing
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            return base64.b64encode(pcm).decode('utf-8')
        return None

