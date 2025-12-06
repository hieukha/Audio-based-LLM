# gemini_module.py - Google Gemini LLM Client
"""
LLM module using Google Gemini API for Vietnamese voice chat.
Provides streaming text generation with support for conversation history.
"""

import os
import re
import logging
import time
import uuid
from typing import Generator, List, Dict, Optional, Any
from threading import Lock

logger = logging.getLogger(__name__)

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("🤖⚠️ google-generativeai not installed. Gemini backend will not function.")


class GeminiLLM:
    """
    Provides streaming text generation using Google Gemini API.
    
    Optimized for Vietnamese voice chat with low-latency streaming.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        max_output_tokens: int = 150,
        temperature: float = 0.7,
    ):
        """
        Initialize the Gemini LLM client.
        
        Args:
            model_name: Gemini model to use (e.g., "gemini-2.5-flash").
            system_prompt: System instruction for the model.
            api_key: Google API key. If None, reads from GEMINI_API_KEY env var.
            max_output_tokens: Maximum tokens in response.
            temperature: Creativity parameter (0.0 to 1.0).
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai is required. Install with: pip install google-generativeai")
        
        self.model_name = model_name
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        
        # Get API key (support both GEMINI_API_KEY and GOOGLE_API_KEY)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY not set. LLM features will be disabled.")
            logger.warning("⚠️ Get API key at: https://aistudio.google.com/apikey")
            self.model = None
            self._inference_time_ms = 500.0
            return
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Safety settings - Allow all content for voice chat
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Create model with system instruction
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
            ),
            safety_settings=safety_settings,
        )
        
        # State
        self._active_requests: Dict[str, Any] = {}
        self._requests_lock = Lock()
        self._inference_time_ms: Optional[float] = None
        
        logger.info(f"🤖✅ GeminiLLM initialized with model: {model_name}")
    
    def _get_default_system_prompt(self) -> str:
        """Returns the default system prompt for Vietnamese voice chat."""
        return """Bạn là một trợ lý AI thân thiện, hữu ích, nói tiếng Việt tự nhiên.

Quy tắc quan trọng:
- Trả lời NGẮN GỌN, tối đa 2-3 câu
- Nói tự nhiên như đang trò chuyện
- KHÔNG dùng markdown, bullet points, hay ký tự đặc biệt
- KHÔNG liệt kê dạng danh sách
- Trả lời trực tiếp, đi thẳng vào vấn đề
- Dùng ngôn ngữ đơn giản, dễ hiểu
- Thân thiện nhưng không quá formal"""
    
    def _convert_history_to_gemini(self, history: List[Dict[str, str]]) -> List[Dict]:
        """Convert chat history to Gemini format."""
        gemini_history = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Gemini uses "user" and "model" roles
            if role == "assistant":
                role = "model"
            elif role == "system":
                continue  # System prompt is handled separately
            
            gemini_history.append({
                "role": role,
                "parts": [{"text": content}]
            })
        
        return gemini_history
    
    def prewarm(self, max_retries: int = 1) -> bool:
        """
        Prewarm the model by making a simple test request.
        
        Returns:
            True if prewarm succeeded, False otherwise.
        """
        logger.info("🤖🔥 Prewarming Gemini model...")
        
        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content(
                    "Xin chào",
                    generation_config=genai.GenerationConfig(max_output_tokens=10)
                )
                if response.text:
                    logger.info(f"🤖🔥✅ Prewarm successful: '{response.text[:30]}...'")
                    return True
            except Exception as e:
                logger.warning(f"🤖🔥⚠️ Prewarm attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(2)
        
        logger.error("🤖🔥❌ Prewarm failed after all retries")
        return False
    
    def measure_inference_time(self, num_tokens: int = 10) -> Optional[float]:
        """
        Measure inference time for the first few tokens.
        
        Returns:
            Time in milliseconds to generate tokens, or None on failure.
        """
        logger.info(f"🤖⏱️ Measuring inference time for {num_tokens} tokens...")
        
        try:
            start_time = time.time()
            token_count = 0
            
            response = self.model.generate_content(
                "Đếm từ 1 đến 10",
                stream=True,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=50,
                    temperature=0.1
                )
            )
            
            first_token_time = None
            for chunk in response:
                if chunk.text:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    token_count += 1
                    if token_count >= num_tokens:
                        break
            
            if first_token_time:
                self._inference_time_ms = first_token_time * 1000
                logger.info(f"🤖⏱️✅ TTFT: {self._inference_time_ms:.2f}ms")
                return self._inference_time_ms
            
        except Exception as e:
            logger.error(f"🤖⏱️❌ Failed to measure inference time: {e}")
        
        return None
    
    def generate(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_system_prompt: bool = True,
        request_id: Optional[str] = None,
        **kwargs: Any
    ) -> Generator[str, None, None]:
        """
        Generate text response with streaming.
        
        Args:
            text: User input text.
            history: Previous conversation messages.
            use_system_prompt: Whether to use system prompt (handled at model level).
            request_id: Optional unique ID for this request.
            **kwargs: Additional generation config parameters.
            
        Yields:
            Text chunks as they are generated.
        """
        req_id = request_id or f"gemini-{uuid.uuid4().hex[:8]}"
        logger.info(f"🤖💬 [{req_id}] Starting generation for: '{text[:50]}...'")
        
        try:
            # Build messages
            gemini_history = []
            if history:
                gemini_history = self._convert_history_to_gemini(history)
            
            # Start chat with history
            chat = self.model.start_chat(history=gemini_history)
            
            # Register request
            with self._requests_lock:
                self._active_requests[req_id] = {
                    "start_time": time.time(),
                    "chat": chat
                }
            
            # Generate with streaming
            response = chat.send_message(text, stream=True)
            
            token_count = 0
            start_time = time.time()
            
            for chunk in response:
                # Check if cancelled
                with self._requests_lock:
                    if req_id not in self._active_requests:
                        logger.info(f"🤖🗑️ [{req_id}] Request cancelled, stopping generation")
                        break
                
                if chunk.text:
                    token_count += 1
                    if token_count == 1:
                        ttft = time.time() - start_time
                        logger.info(f"🤖⏱️ [{req_id}] TTFT: {ttft:.3f}s")
                    
                    yield chunk.text
            
            logger.info(f"🤖✅ [{req_id}] Generation complete. Tokens: {token_count}")
            
        except Exception as e:
            logger.error(f"🤖💥 [{req_id}] Generation error: {e}")
            raise
        finally:
            with self._requests_lock:
                self._active_requests.pop(req_id, None)
    
    def cancel_generation(self, request_id: Optional[str] = None) -> bool:
        """
        Cancel active generation(s).
        
        Args:
            request_id: Specific request to cancel, or None for all.
            
        Returns:
            True if any request was cancelled.
        """
        with self._requests_lock:
            if request_id:
                if request_id in self._active_requests:
                    del self._active_requests[request_id]
                    logger.info(f"🤖🗑️ Cancelled request: {request_id}")
                    return True
                return False
            else:
                count = len(self._active_requests)
                self._active_requests.clear()
                logger.info(f"🤖🗑️ Cancelled {count} active requests")
                return count > 0
    
    @property
    def inference_time_ms(self) -> float:
        """Returns the measured inference time in milliseconds."""
        return self._inference_time_ms or 500.0  # Default estimate


# Context manager for safe generation
class GeminiGenerationContext:
    """Context manager for safe Gemini text generation."""
    
    def __init__(
        self,
        llm: GeminiLLM,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
    ):
        self.llm = llm
        self.prompt = prompt
        self.history = history
        self.kwargs = kwargs
        self.generator = None
        self.request_id = f"ctx-gemini-{uuid.uuid4().hex[:8]}"
    
    def __enter__(self) -> Generator[str, None, None]:
        self.generator = self.llm.generate(
            self.prompt,
            self.history,
            request_id=self.request_id,
            **self.kwargs
        )
        return self.generator
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.llm.cancel_generation(self.request_id)
        if self.generator and hasattr(self.generator, 'close'):
            try:
                self.generator.close()
            except Exception:
                pass
        return False


# Example usage
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    
    try:
        llm = GeminiLLM()
        
        # Prewarm
        llm.prewarm()
        
        # Measure inference time
        llm.measure_inference_time()
        
        # Test generation
        print("\n🤖 Testing Gemini generation:")
        print("-" * 40)
        
        for chunk in llm.generate("Xin chào, bạn khỏe không?"):
            print(chunk, end="", flush=True)
        print("\n")
        
        # Test with context manager
        print("🤖 Testing with context manager:")
        print("-" * 40)
        
        with GeminiGenerationContext(llm, "Kể cho tôi một câu chuyện ngắn") as gen:
            for chunk in gen:
                print(chunk, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")

