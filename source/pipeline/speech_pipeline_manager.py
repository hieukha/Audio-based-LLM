# speech_pipeline_manager.py - Speech Pipeline Manager
"""
Orchestrates the text-to-speech pipeline: STT -> LLM -> TTS.
Integrates Gemini LLM with Piper TTS for Vietnamese voice chat.
"""

import logging
import threading
import time
from queue import Queue, Empty
from typing import Optional, Callable, List, Dict

from modules.tts import PiperAudioProcessor
from modules.llm import GeminiLLM
from utils import TextSimilarity, TextContext, Colors

logger = logging.getLogger(__name__)

# Load system prompt
DEFAULT_SYSTEM_PROMPT = """Bạn là một trợ lý AI thân thiện nói tiếng Việt.

Quy tắc:
- Trả lời NGẮN GỌN, tối đa 2-3 câu
- Nói tự nhiên như đang trò chuyện
- KHÔNG dùng markdown hay ký tự đặc biệt
- Trả lời trực tiếp, đi thẳng vào vấn đề"""

try:
    with open("config/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    logger.info("🗣️📄 System prompt loaded from file")
except FileNotFoundError:
    system_prompt = DEFAULT_SYSTEM_PROMPT
    logger.info("🗣️📄 Using default system prompt")


class PipelineRequest:
    """Represents a request to be processed by the pipeline."""
    
    def __init__(self, action: str, data: Optional[any] = None):
        self.action = action
        self.data = data
        self.timestamp = time.time()


class RunningGeneration:
    """Holds state for a single text-to-speech generation."""
    
    def __init__(self, gen_id: int):
        self.id = gen_id
        self.text: Optional[str] = None
        self.timestamp = time.time()
        
        # LLM state
        self.llm_generator = None
        self.llm_finished: bool = False
        self.llm_aborted: bool = False
        
        # Quick answer (first sentence)
        self.quick_answer: str = ""
        self.quick_answer_provided: bool = False
        self.quick_answer_first_chunk_ready: bool = False
        self.quick_answer_overhang: str = ""
        
        # TTS state
        self.tts_quick_started: bool = False
        self.tts_quick_allowed_event = threading.Event()
        self.audio_chunks = Queue()
        self.audio_quick_finished: bool = False
        self.audio_quick_aborted: bool = False
        
        # Final answer state
        self.tts_final_started: bool = False
        self.audio_final_finished: bool = False
        self.audio_final_aborted: bool = False
        self.final_answer: str = ""
        
        # General state
        self.abortion_started: bool = False
        self.completed: bool = False


class SpeechPipelineManager:
    """
    Orchestrates STT -> LLM -> TTS pipeline for Vietnamese voice chat.
    
    Uses Gemini for LLM and Piper for TTS.
    """
    
    def __init__(
        self,
        tts_models_dir: str = "models/piper",
        gemini_model: str = "gemini-2.5-flash",
    ):
        """
        Initialize the SpeechPipelineManager.
        
        Args:
            tts_models_dir: Directory containing Piper TTS models.
            gemini_model: Gemini model to use.
        """
        logger.info("🗣️🚀 Initializing SpeechPipelineManager...")
        
        # Initialize TTS
        logger.info("🗣️🔊 Initializing Piper TTS...")
        self.audio = PiperAudioProcessor(models_dir=tts_models_dir)
        self.audio.on_first_audio_chunk_synthesize = self.on_first_audio_chunk_synthesize
        
        # Initialize LLM
        logger.info("🗣️🧠 Initializing Gemini LLM...")
        self.llm = GeminiLLM(
            model_name=gemini_model,
            system_prompt=system_prompt,
        )
        self.llm.prewarm()
        self.llm_inference_time = self.llm.measure_inference_time() or 500.0
        
        # Utilities
        self.text_similarity = TextSimilarity(focus='end', n_words=5)
        self.text_context = TextContext()
        
        # State
        self.history: List[Dict[str, str]] = []
        self.requests_queue = Queue()
        self.running_generation: Optional[RunningGeneration] = None
        self.generation_counter: int = 0
        
        # Threading
        self.shutdown_event = threading.Event()
        self.abort_lock = threading.Lock()
        self.generator_ready_event = threading.Event()
        self.llm_answer_ready_event = threading.Event()
        self.stop_llm_request_event = threading.Event()
        self.stop_tts_quick_request_event = threading.Event()
        self.stop_tts_final_request_event = threading.Event()
        self.abort_block_event = threading.Event()
        self.abort_block_event.set()
        
        # Flags
        self.llm_generation_active = False
        self.tts_quick_generation_active = False
        self.tts_final_generation_active = False
        
        # Callback
        self.on_partial_assistant_text: Optional[Callable[[str], None]] = None
        
        # Start worker threads
        self._start_workers()
        
        # Calculate latency
        self.full_output_pipeline_latency = self.llm_inference_time + self.audio.tts_inference_time
        logger.info(f"🗣️⏱️ Pipeline latency: {self.full_output_pipeline_latency:.2f}ms "
                   f"(LLM: {self.llm_inference_time:.2f}ms, TTS: {self.audio.tts_inference_time:.2f}ms)")
        
        logger.info("🗣️✅ SpeechPipelineManager initialized")

    def _start_workers(self):
        """Start background worker threads."""
        threads = [
            ("RequestProcessor", self._request_processing_worker),
            ("LLMWorker", self._llm_inference_worker),
            ("TTSQuickWorker", self._tts_quick_inference_worker),
            ("TTSFinalWorker", self._tts_final_inference_worker),
        ]
        
        for name, target in threads:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            logger.debug(f"🗣️▶️ Started {name} thread")

    def is_valid_gen(self) -> bool:
        """Check if there's a valid running generation."""
        return self.running_generation is not None and not self.running_generation.abortion_started

    def on_first_audio_chunk_synthesize(self):
        """Callback when first TTS audio chunk is ready."""
        logger.info("🗣️🎶 First audio chunk synthesized")
        if self.running_generation:
            self.running_generation.quick_answer_first_chunk_ready = True

    def preprocess_chunk(self, chunk: str) -> str:
        """Preprocess text chunk before TTS."""
        return chunk.replace("—", "-").replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'").replace("…", "...")

    # === Worker Threads ===

    def _request_processing_worker(self):
        """Process requests from the queue."""
        logger.info("🗣️🚀 Request Processor started")
        
        while not self.shutdown_event.is_set():
            try:
                request = self.requests_queue.get(block=True, timeout=1)
                
                # Drain queue to get most recent request
                while not self.requests_queue.empty():
                    skipped = self.requests_queue.get(False)
                    request = skipped
                
                self.abort_block_event.wait()
                
                if request.action == "prepare":
                    self.process_prepare_generation(request.data)
                
            except Empty:
                continue
            except Exception as e:
                logger.exception(f"🗣️💥 Request Processor error: {e}")
        
        logger.info("🗣️🏁 Request Processor stopped")

    def _llm_inference_worker(self):
        """Handle LLM inference."""
        logger.info("🗣️🧠 LLM Worker started")
        
        while not self.shutdown_event.is_set():
            if not self.generator_ready_event.wait(timeout=1.0):
                continue
            
            if self.stop_llm_request_event.is_set():
                self.stop_llm_request_event.clear()
                self.llm_generation_active = False
                continue
            
            self.generator_ready_event.clear()
            current_gen = self.running_generation
            
            if not current_gen or not current_gen.llm_generator:
                self.llm_generation_active = False
                continue
            
            gen_id = current_gen.id
            logger.info(f"🗣️🧠🔄 [Gen {gen_id}] LLM processing...")
            
            self.llm_generation_active = True
            start_time = time.time()
            token_count = 0
            
            try:
                for chunk in current_gen.llm_generator:
                    if self.stop_llm_request_event.is_set():
                        logger.info(f"🗣️🧠❌ [Gen {gen_id}] LLM stopped")
                        current_gen.llm_aborted = True
                        break
                    
                    chunk = self.preprocess_chunk(chunk)
                    token_count += 1
                    current_gen.quick_answer += chunk
                    
                    if token_count == 1:
                        ttft = time.time() - start_time
                        logger.info(f"🗣️🧠⏱️ [Gen {gen_id}] TTFT: {ttft:.3f}s")
                    
                    # Check for quick answer boundary
                    if not current_gen.quick_answer_provided:
                        context, overhang = self.text_context.get_context(current_gen.quick_answer)
                        if context:
                            logger.info(f"🗣️🧠✔️ [Gen {gen_id}] Quick answer: {context[:50]}...")
                            current_gen.quick_answer = context
                            current_gen.quick_answer_overhang = overhang
                            current_gen.quick_answer_provided = True
                            
                            if self.on_partial_assistant_text:
                                self.on_partial_assistant_text(current_gen.quick_answer)
                            
                            self.llm_answer_ready_event.set()
                            break
                
                # If no boundary found, use full response
                if not current_gen.llm_aborted and not current_gen.quick_answer_provided:
                    logger.info(f"🗣️🧠✔️ [Gen {gen_id}] Using full response as quick answer")
                    current_gen.quick_answer_provided = True
                    if self.on_partial_assistant_text:
                        self.on_partial_assistant_text(current_gen.quick_answer)
                    self.llm_answer_ready_event.set()
                
            except Exception as e:
                logger.exception(f"🗣️🧠💥 [Gen {gen_id}] LLM error: {e}")
                current_gen.llm_aborted = True
            finally:
                self.llm_generation_active = False
                current_gen.llm_finished = True
                
                if current_gen.llm_aborted:
                    self.stop_tts_quick_request_event.set()
                    self.stop_tts_final_request_event.set()
                    self.llm_answer_ready_event.set()
                
                logger.info(f"🗣️🧠🏁 [Gen {gen_id}] LLM finished")

    def _tts_quick_inference_worker(self):
        """Handle quick TTS synthesis."""
        logger.info("🗣️👄🚀 Quick TTS Worker started")
        
        while not self.shutdown_event.is_set():
            if not self.llm_answer_ready_event.wait(timeout=1.0):
                continue
            
            if self.stop_tts_quick_request_event.is_set():
                self.stop_tts_quick_request_event.clear()
                self.tts_quick_generation_active = False
                continue
            
            self.llm_answer_ready_event.clear()
            current_gen = self.running_generation
            
            if not current_gen or not current_gen.quick_answer:
                self.tts_quick_generation_active = False
                continue
            
            if current_gen.audio_quick_aborted or current_gen.abortion_started:
                continue
            
            gen_id = current_gen.id
            logger.info(f"🗣️👄🔄 [Gen {gen_id}] Quick TTS processing...")
            
            self.tts_quick_generation_active = True
            current_gen.tts_quick_started = True
            
            try:
                if self.stop_tts_quick_request_event.is_set() or current_gen.abortion_started:
                    current_gen.audio_quick_aborted = True
                else:
                    logger.info(f"🗣️👄🎶 [Gen {gen_id}] Synthesizing: '{current_gen.quick_answer[:50]}...'")
                    completed = self.audio.synthesize(
                        current_gen.quick_answer,
                        current_gen.audio_chunks,
                        self.stop_tts_quick_request_event,
                        f"[Gen {gen_id}]"
                    )
                    
                    if not completed:
                        current_gen.audio_quick_aborted = True
                    else:
                        logger.info(f"🗣️👄✅ [Gen {gen_id}] Quick TTS complete")
                
            except Exception as e:
                logger.exception(f"🗣️👄💥 [Gen {gen_id}] Quick TTS error: {e}")
                current_gen.audio_quick_aborted = True
            finally:
                self.tts_quick_generation_active = False
                current_gen.audio_quick_finished = True
                self.stop_tts_quick_request_event.clear()

    def _tts_final_inference_worker(self):
        """Handle final TTS synthesis."""
        logger.info("🗣️👄🚀 Final TTS Worker started")
        
        while not self.shutdown_event.is_set():
            time.sleep(0.01)
            
            current_gen = self.running_generation
            if not current_gen:
                continue
            if current_gen.tts_final_started:
                continue
            if not current_gen.tts_quick_started:
                continue
            if not current_gen.audio_quick_finished:
                continue
            if current_gen.audio_quick_aborted:
                continue
            if not current_gen.quick_answer_provided:
                continue
            if current_gen.abortion_started:
                continue
            
            gen_id = current_gen.id
            logger.info(f"🗣️👄🔄 [Gen {gen_id}] Final TTS processing...")
            
            def get_generator():
                """Yields remaining text for final TTS."""
                # Yield overhang first
                if current_gen.quick_answer_overhang:
                    overhang = self.preprocess_chunk(current_gen.quick_answer_overhang)
                    current_gen.final_answer += overhang
                    if self.on_partial_assistant_text:
                        self.on_partial_assistant_text(current_gen.quick_answer + current_gen.final_answer)
                    yield overhang
                
                # Yield remaining from LLM
                try:
                    for chunk in current_gen.llm_generator:
                        if self.stop_tts_final_request_event.is_set():
                            current_gen.audio_final_aborted = True
                            break
                        
                        processed = self.preprocess_chunk(chunk)
                        current_gen.final_answer += processed
                        if self.on_partial_assistant_text:
                            self.on_partial_assistant_text(current_gen.quick_answer + current_gen.final_answer)
                        yield processed
                except Exception as e:
                    logger.exception(f"🗣️👄💥 [Gen {gen_id}] Final TTS gen error: {e}")
                    current_gen.audio_final_aborted = True
            
            self.tts_final_generation_active = True
            current_gen.tts_final_started = True
            
            try:
                completed = self.audio.synthesize_generator(
                    get_generator(),
                    current_gen.audio_chunks,
                    self.stop_tts_final_request_event,
                    f"[Gen {gen_id}]"
                )
                
                if not completed:
                    current_gen.audio_final_aborted = True
                
            except Exception as e:
                logger.exception(f"🗣️👄💥 [Gen {gen_id}] Final TTS error: {e}")
                current_gen.audio_final_aborted = True
            finally:
                self.tts_final_generation_active = False
                current_gen.audio_final_finished = True
                self.stop_tts_final_request_event.clear()

    # === Public Methods ===

    def prepare_generation(self, txt: str):
        """Queue a new generation request."""
        logger.info(f"🗣️📥 Queueing request: '{txt[:50]}...'")
        self.requests_queue.put(PipelineRequest("prepare", txt))

    def process_prepare_generation(self, txt: str):
        """Process a prepare generation request."""
        # Check if we need to abort existing generation
        self.check_abort(txt, wait_for_finish=True, abort_reason="new request")
        
        # Create new generation
        self.generation_counter += 1
        gen_id = self.generation_counter
        logger.info(f"🗣️✨ [Gen {gen_id}] Preparing for: '{txt[:50]}...'")
        
        # Reset state
        self.llm_generation_active = False
        self.tts_quick_generation_active = False
        self.tts_final_generation_active = False
        self.generator_ready_event.clear()
        self.llm_answer_ready_event.clear()
        self.stop_llm_request_event.clear()
        self.stop_tts_quick_request_event.clear()
        self.stop_tts_final_request_event.clear()
        self.abort_block_event.set()
        
        # Create generation object
        self.running_generation = RunningGeneration(gen_id)
        self.running_generation.text = txt
        
        try:
            logger.info(f"🗣️🧠🚀 [Gen {gen_id}] Calling LLM...")
            self.running_generation.llm_generator = self.llm.generate(
                text=txt,
                history=self.history,
                use_system_prompt=True,
            )
            logger.info(f"🗣️🧠✔️ [Gen {gen_id}] LLM generator created")
            self.generator_ready_event.set()
        except Exception as e:
            logger.exception(f"🗣️🧠💥 [Gen {gen_id}] Failed to create LLM generator: {e}")
            self.running_generation = None

    def check_abort(self, txt: str, wait_for_finish: bool = True, abort_reason: str = ""):
        """Check if current generation should be aborted."""
        if self.running_generation:
            if self.running_generation.abortion_started:
                return True
            
            # Check similarity
            if self.running_generation.text:
                similarity = self.text_similarity.calculate_similarity(
                    self.running_generation.text, txt
                )
                if similarity >= 0.95:
                    logger.info(f"🗣️🛑 Text too similar ({similarity:.2f}), ignoring")
                    return False
            
            # Abort existing generation
            self.abort_generation(wait_for_completion=wait_for_finish, reason=abort_reason)
            return True
        
        return False

    def abort_generation(self, wait_for_completion: bool = False, timeout: float = 5.0, reason: str = ""):
        """Abort the current generation."""
        with self.abort_lock:
            current_gen = self.running_generation
            
            if not current_gen or current_gen.abortion_started:
                return
            
            gen_id = current_gen.id
            logger.info(f"🗣️🛑🚀 [Gen {gen_id}] Aborting (reason: {reason})...")
            
            current_gen.abortion_started = True
            self.abort_block_event.clear()
            
            # Stop all workers
            self.stop_llm_request_event.set()
            self.stop_tts_quick_request_event.set()
            self.stop_tts_final_request_event.set()
            self.generator_ready_event.set()
            self.llm_answer_ready_event.set()
            
            # Cancel LLM
            self.llm.cancel_generation()
            
            # Wait a bit for workers to stop
            time.sleep(0.1)
            
            # Clear generation
            self.running_generation = None
            self.generator_ready_event.clear()
            self.llm_answer_ready_event.clear()
            self.abort_block_event.set()
            
            logger.info(f"🗣️🛑✅ [Gen {gen_id}] Abort complete")

    def reset(self):
        """Reset the pipeline state."""
        logger.info("🗣️🔄 Resetting pipeline...")
        self.abort_generation(wait_for_completion=True, reason="reset")
        self.history = []
        logger.info("🗣️🧹 Reset complete")

    def shutdown(self):
        """Shutdown the pipeline manager."""
        logger.info("🗣️🔌 Shutting down...")
        self.shutdown_event.set()
        self.abort_generation(wait_for_completion=True, reason="shutdown")
        
        # Wake up waiting threads
        self.generator_ready_event.set()
        self.llm_answer_ready_event.set()
        
        logger.info("🗣️🔌✅ Shutdown complete")

