# server.py - FastAPI WebSocket Server for Vietnamese Voice Chat
"""
Real-time voice chat server using FastAPI and WebSockets.
Integrates: RealtimeSTT (Whisper) + Gemini LLM + Piper TTS
"""

from queue import Queue, Empty
# Empty is already imported from queue module
import logging
from utils.logsetup import setup_logging
setup_logging(logging.INFO)  # Back to INFO level
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🖥️👋 Welcome to Vietnamese Real-Time Voice Chat")

import uvicorn
import asyncio
import struct
import json
import time
import base64
import threading
import sys
import os
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, Response, FileResponse

from modules.audio import AudioInputProcessor
from pipeline import SpeechPipelineManager
from utils import Colors
from utils.upsample_overlap import UpsampleOverlap

# Configuration
LANGUAGE = "vi"
MAX_AUDIO_QUEUE_SIZE = 50

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class NoCacheStaticFiles(StaticFiles):
    """Serves static files without caching."""
    
    async def get_response(self, path: str, scope: Dict[str, Any]) -> Response:
        response: Response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        if "etag" in response.headers:
            response.headers.__delitem__("etag")
        if "last-modified" in response.headers:
            response.headers.__delitem__("last-modified")
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🖥️▶️ Server starting up")
    
    # Initialize components
    app.state.SpeechPipelineManager = SpeechPipelineManager(
        tts_models_dir="../models/piper",  # Relative to code folder
        gemini_model="gemini-2.0-flash",  # Use 2.0-flash (free tier, stable)
    )
    
    app.state.AudioInputProcessor = AudioInputProcessor(
        LANGUAGE,
        pipeline_latency=app.state.SpeechPipelineManager.full_output_pipeline_latency / 1000,
    )
    
    # Initialize Upsampler for TTS audio (22.05kHz -> 48kHz)
    app.state.Upsampler = UpsampleOverlap(input_rate=22050, output_rate=48000)
    
    yield
    
    logger.info("🖥️⏹️ Server shutting down")
    app.state.AudioInputProcessor.shutdown()


# FastAPI app
app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")


@app.get("/")
async def get_index() -> HTMLResponse:
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


def parse_json_message(text: str) -> dict:
    """Parse JSON message safely."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("🖥️⚠️ Invalid JSON message")
        return {}


def format_timestamp_ns(timestamp_ns: int) -> str:
    """Format nanosecond timestamp."""
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000
    dt = datetime.fromtimestamp(seconds)
    time_str = dt.strftime("%H:%M:%S")
    milliseconds = remainder_ns // 1_000_000
    return f"{time_str}.{milliseconds:03d}"


class TranscriptionCallbacks:
    """Manages callbacks for a WebSocket connection."""
    
    def __init__(self, app: FastAPI, message_queue: asyncio.Queue):
        self.app = app
        self.message_queue = message_queue
        self.final_transcription = ""
        
        # State flags
        self.tts_to_client: bool = False
        self.user_interrupted: bool = False
        self.tts_chunk_sent: bool = False
        self.tts_client_playing: bool = False
        self.interruption_time: float = 0.0
        self.is_hot: bool = False
        self.synthesis_started: bool = False
        self.assistant_answer: str = ""
        self.final_assistant_answer_sent: bool = False
        self.partial_transcription: str = ""
        
        self.reset_state()

    def reset_state(self):
        """Reset connection state."""
        self.tts_to_client = False
        self.user_interrupted = False
        self.tts_chunk_sent = False
        self.interruption_time = 0.0
        self.is_hot = False
        self.synthesis_started = False
        self.assistant_answer = ""
        self.final_assistant_answer_sent = False
        self.partial_transcription = ""
        self.app.state.AudioInputProcessor.abort_generation()

    def on_partial(self, txt: str):
        """Handle partial transcription."""
        self.final_assistant_answer_sent = False
        self.final_transcription = ""
        self.partial_transcription = txt
        self.message_queue.put_nowait({"type": "partial_user_request", "content": txt})
        
        # Trigger check_abort for potential sentence
        self.app.state.SpeechPipelineManager.check_abort(txt, False, "on_partial")

    def on_potential_sentence(self, txt: str):
        """Handle potential sentence end."""
        logger.debug(f"🖥️🧠 Potential sentence: '{txt}'")
        self.app.state.SpeechPipelineManager.prepare_generation(txt)

    def on_before_final(self, audio: bytes, txt: str):
        """Handle before final transcription."""
        logger.info(Colors.apply('🖥️🏁 USER TURN END').light_gray())
        
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            self.app.state.SpeechPipelineManager.running_generation.tts_quick_allowed_event.set()
        
        if not self.app.state.AudioInputProcessor.interrupted:
            self.app.state.AudioInputProcessor.interrupted = True
            self.interruption_time = time.time()
        
        # CRITICAL: Enable TTS streaming to client
        logger.info("🖥️🔊 TTS STREAM RELEASED")
        self.tts_to_client = True
        
        self.tts_to_client = True
        
        user_request = self.final_transcription if self.final_transcription else self.partial_transcription
        self.message_queue.put_nowait({
            "type": "final_user_request",
            "content": user_request
        })
        
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            if self.app.state.SpeechPipelineManager.running_generation.quick_answer:
                self.assistant_answer = self.app.state.SpeechPipelineManager.running_generation.quick_answer
                self.message_queue.put_nowait({
                    "type": "partial_assistant_answer",
                    "content": self.assistant_answer
                })
        
        self.app.state.SpeechPipelineManager.history.append({"role": "user", "content": user_request})

    def on_final(self, txt: str):
        """Handle final transcription."""
        logger.info(f"\n{Colors.apply('🖥️✅ FINAL USER: ').green()}{txt}")
        if not self.final_transcription:
            self.final_transcription = txt

    def on_recording_start(self):
        """Handle recording start."""
        logger.info(f"{Colors.ORANGE}🖥️🎙️ Recording started{Colors.RESET}")
        
        if self.tts_client_playing:
            self.tts_to_client = False
            self.user_interrupted = True
            
            self.send_final_assistant_answer(forced=True)
            self.tts_chunk_sent = False
            
            self.message_queue.put_nowait({"type": "stop_tts", "content": ""})
            self.app.state.SpeechPipelineManager.abort_generation(reason="user interrupted")
            self.message_queue.put_nowait({"type": "tts_interruption", "content": ""})

    def on_partial_assistant_text(self, txt: str):
        """Handle partial assistant response."""
        logger.info(f"{Colors.apply('🖥️💬 ASSISTANT: ').green()}{txt}")
        if not self.user_interrupted:
            self.assistant_answer = txt
            # Always send partial answer to show bot is responding
            self.message_queue.put_nowait({
                "type": "partial_assistant_answer",
                "content": txt
            })

    def send_final_assistant_answer(self, forced=False):
        """Send final assistant answer to client."""
        final_answer = ""
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            gen = self.app.state.SpeechPipelineManager.running_generation
            final_answer = gen.quick_answer + gen.final_answer
        
        if not final_answer:
            if forced and self.assistant_answer:
                final_answer = self.assistant_answer
            else:
                return
        
        if not self.final_assistant_answer_sent and final_answer:
            import re
            cleaned = re.sub(r'[\r\n]+', ' ', final_answer)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            if cleaned:
                logger.info(f"\n{Colors.apply('🖥️✅ FINAL ASSISTANT: ').green()}{cleaned}")
                self.message_queue.put_nowait({
                    "type": "final_assistant_answer",
                    "content": cleaned
                })
                self.app.state.SpeechPipelineManager.history.append({
                    "role": "assistant",
                    "content": cleaned
                })
                self.final_assistant_answer_sent = True


async def process_incoming_data(ws: WebSocket, app: FastAPI, incoming_chunks: asyncio.Queue, callbacks: TranscriptionCallbacks):
    """Process incoming WebSocket data."""
    try:
        while True:
            msg = await ws.receive()
            
            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                
                if len(raw) < 8:
                    continue
                
                timestamp_ms, flags = struct.unpack("!II", raw[:8])
                client_sent_ns = timestamp_ms * 1_000_000
                
                metadata = {
                    "client_sent_ms": timestamp_ms,
                    "client_sent": client_sent_ns,
                    "isTTSPlaying": bool(flags & 1),
                    "server_received": time.time_ns(),
                    "pcm": raw[8:]
                }
                
                if incoming_chunks.qsize() < MAX_AUDIO_QUEUE_SIZE:
                    await incoming_chunks.put(metadata)
            
            elif "text" in msg and msg["text"]:
                data = parse_json_message(msg["text"])
                msg_type = data.get("type")
                
                if msg_type == "tts_start":
                    callbacks.tts_client_playing = True
                elif msg_type == "tts_stop":
                    callbacks.tts_client_playing = False
                elif msg_type == "clear_history":
                    app.state.SpeechPipelineManager.reset()
    
    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect:
        logger.warning("🖥️⚠️ WebSocket disconnected")
    except Exception as e:
        logger.exception(f"🖥️💥 Error processing data: {e}")


async def send_text_messages(ws: WebSocket, message_queue: asyncio.Queue):
    """Send text messages to client."""
    try:
        while True:
            await asyncio.sleep(0.001)
            data = await message_queue.get()
            msg_type = data.get("type")
            if msg_type != "tts_chunk":
                logger.info(f"🖥️📤 →→Client: {data}")
            await ws.send_json(data)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.exception(f"🖥️💥 Error sending messages: {e}")


async def send_tts_chunks(app: FastAPI, message_queue: asyncio.Queue, callbacks: TranscriptionCallbacks):
    """Send TTS audio chunks to client."""
    try:
        logger.info("🖥️🔊 Starting TTS chunk sender")
        
        while True:
            await asyncio.sleep(0.001)
            
            # Reset interruption after timeout
            if app.state.AudioInputProcessor.interrupted and callbacks.interruption_time:
                if time.time() - callbacks.interruption_time > 2.0:
                    app.state.AudioInputProcessor.interrupted = False
                    callbacks.interruption_time = 0
            
            if not callbacks.tts_to_client:
                continue
            
            if not app.state.SpeechPipelineManager.running_generation:
                continue
            
            gen = app.state.SpeechPipelineManager.running_generation
            
            if gen.abortion_started:
                continue
            
            # Signal TTS worker to start if not finished
            if not gen.audio_quick_finished:
                gen.tts_quick_allowed_event.set()
            
            if not gen.quick_answer_first_chunk_ready:
                continue
            
            # Get audio chunk
            try:
                chunk = gen.audio_chunks.get_nowait()
                if chunk:
                    # Upsample and encode as base64 for WebSocket
                    b64_chunk = app.state.Upsampler.get_base64_chunk(chunk)
                    message_queue.put_nowait({
                        "type": "tts_chunk",
                        "content": b64_chunk
                    })
                    
                    if not callbacks.tts_chunk_sent:
                        logger.info("🖥️🔊 First TTS chunk sent")
                        # Reset interruption flag after first chunk
                        async def reset_interrupt():
                            await asyncio.sleep(1)
                            if app.state.AudioInputProcessor.interrupted:
                                app.state.AudioInputProcessor.interrupted = False
                                callbacks.interruption_time = 0
                        asyncio.create_task(reset_interrupt())
                    
                    callbacks.tts_chunk_sent = True
            
            except Empty:
                # Check if generation is done - queue is empty, that's normal
                if gen.quick_answer_provided and gen.audio_final_finished:
                    logger.info("🖥️🏁 TTS complete")
                    callbacks.send_final_assistant_answer()
                    app.state.SpeechPipelineManager.running_generation = None
                    callbacks.tts_chunk_sent = False
                    callbacks.reset_state()
    
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.exception(f"🖥️💥 Error sending TTS: {e}")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Main WebSocket endpoint."""
    await ws.accept()
    logger.info("🖥️✅ Client connected")
    
    message_queue = asyncio.Queue()
    audio_chunks = asyncio.Queue()
    
    callbacks = TranscriptionCallbacks(app, message_queue)
    
    # Set up callbacks
    app.state.AudioInputProcessor.realtime_callback = callbacks.on_partial
    app.state.AudioInputProcessor.transcriber.potential_sentence_end = callbacks.on_potential_sentence
    app.state.AudioInputProcessor.transcriber.full_transcription_callback = callbacks.on_final
    app.state.AudioInputProcessor.transcriber.before_final_sentence = callbacks.on_before_final
    app.state.AudioInputProcessor.recording_start_callback = callbacks.on_recording_start
    app.state.SpeechPipelineManager.on_partial_assistant_text = callbacks.on_partial_assistant_text
    
    # Create tasks
    tasks = [
        asyncio.create_task(process_incoming_data(ws, app, audio_chunks, callbacks)),
        asyncio.create_task(app.state.AudioInputProcessor.process_chunk_queue(audio_chunks)),
        asyncio.create_task(send_text_messages(ws, message_queue)),
        asyncio.create_task(send_tts_chunks(app, message_queue, callbacks)),
    ]
    
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception as e:
        logger.error(f"🖥️💥 WebSocket error: {e}")
    finally:
        logger.info("🖥️🧹 Cleaning up...")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("🖥️❌ WebSocket session ended")


if __name__ == "__main__":
    logger.info("🖥️▶️ Starting server on http://localhost:45679")
    uvicorn.run("server:app", host="0.0.0.0", port=45679, log_config=None)

