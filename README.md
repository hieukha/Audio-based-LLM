# Vietnamese Real-Time Voice Chat System

**Hệ thống trò chuyện giọng nói tiếng Việt thời gian thực**

Dự án này sử dụng:
- **STT**: OpenAI Whisper API (speech-to-text)
- **LLM**: Google Gemini API
- **TTS**: Piper TTS cho tiếng Việt

## 🎬 Demo

Xem video demo hệ thống:

<video width="100%" controls>
  <source src="Demo_Audio-based_LLM.mp4" type="video/mp4">
  Trình duyệt của bạn không hỗ trợ video tag.
</video>

**Hoặc xem trực tiếp:** [Demo_Audio-based_LLM.mp4](./Demo_Audio-based_LLM.mp4)

## 🏗️ Cấu trúc dự án

```
RealtimeSystem/
├── code/
│   ├── server.py              # FastAPI WebSocket server
│   ├── audio_in.py            # Audio input processor
│   ├── transcribe.py          # STT với RealtimeSTT
│   ├── gemini_module.py       # Gemini LLM client
│   ├── piper_audio_module.py  # Piper TTS processor
│   ├── speech_pipeline_manager.py  # Pipeline orchestrator
│   ├── text_similarity.py     # Text comparison utilities
│   ├── text_context.py        # Context extraction
│   ├── colors.py              # Terminal colors
│   ├── logsetup.py            # Logging configuration
│   ├── system_prompt.txt      # AI system prompt
│   └── static/
│       ├── index.html         # Web UI
│       └── app.js             # Client JavaScript
├── models/
│   └── piper/                 # Piper TTS models (download separately)
├── requirements.txt
└── README.md
```

## 🚀 Cài đặt

### 1. Tạo môi trường ảo

```bash
cd /workspace/khanh/Audio_System/RealtimeSystem
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình API Key

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

### 4. Tải Piper TTS Binary và Model

#### 4.1. Tải Piper binary (bắt buộc)

```bash
mkdir -p models/piper
cd models/piper

# Tải Piper binary cho Linux x86_64
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
mv piper/* .
rm -rf piper piper_amd64.tar.gz

# Kiểm tra
./piper --help
```

> **Lưu ý**: Cho các platform khác, tải từ [Piper Releases](https://github.com/rhasspy/piper/releases):
> - Linux ARM64: `piper_arm64.tar.gz`
> - macOS: `piper_macos_x64.tar.gz`
> - Windows: `piper_windows_amd64.zip`

#### 4.2. Tải model tiếng Việt

```bash
# Tải model tiếng Việt (trong thư mục models/piper)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json
```

Hoặc copy model đã fine-tune:
```bash
cp /path/to/your/finetuned/model.onnx models/piper/vi_VN-finetuned.onnx
cp /path/to/your/finetuned/model.onnx.json models/piper/vi_VN-finetuned.onnx.json
```

#### 4.3. Cấu trúc thư mục models/piper sau khi cài đặt

```
models/piper/
├── piper                        # Piper binary executable
├── espeak-ng-data/              # eSpeak data
├── libespeak-ng.so*             # Shared libraries
├── libonnxruntime.so*
├── libpiper_phonemize.so*
├── vi_VN-vais1000-medium.onnx   # Vietnamese model
└── vi_VN-vais1000-medium.onnx.json
```

## ▶️ Chạy hệ thống

### Chạy server

```bash
# Set environment variables
export LD_LIBRARY_PATH="/path/to/RealtimeSystem/models/piper:$LD_LIBRARY_PATH"
export GEMINI_API_KEY="your-gemini-api-key"

# Nếu dùng GPU cho Whisper STT
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/audio_system/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

# Chạy server
cd code
python server.py
```

Server sẽ chạy tại: http://localhost:45678

### Sử dụng

1. Mở trình duyệt và truy cập http://localhost:8000
2. Nhấn "Bắt đầu" để bắt đầu trò chuyện
3. Nói vào microphone
4. AI sẽ trả lời bằng giọng nói tiếng Việt

## 🔧 Cấu hình

### System Prompt

Chỉnh sửa file `code/system_prompt.txt` để thay đổi cách AI trả lời.

### Gemini Model

Trong `code/server.py`, thay đổi:
```python
gemini_model="gemini-2.0-flash"
```

### Piper TTS Model

Model được tự động tìm trong `models/piper/`. Ưu tiên:
1. `vi_VN-finetuned.onnx`
2. `vi_VN-vais1000-medium.onnx`

## 📊 Pipeline Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Browser    │───▶│  FastAPI    │───▶│  STT        │
│  (Audio)    │    │  WebSocket  │    │  (Whisper)  │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Browser    │◀───│  WebSocket  │◀───│  LLM        │
│  (TTS Audio)│    │  (Streaming)│    │  (Gemini)   │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
                                      ┌─────────────┐
                                      │  TTS        │
                                      │  (Piper)    │
                                      └─────────────┘
```

## 🎯 Features

- **Real-time streaming**: LLM và TTS streaming để giảm latency
- **Barge-in**: Có thể ngắt AI khi đang nói
- **Vietnamese optimized**: Tối ưu cho tiếng Việt
- **Web-based**: Không cần cài đặt thêm phần mềm

## 🐛 Troubleshooting

### Lỗi microphone
- Kiểm tra quyền truy cập microphone trong trình duyệt
- Sử dụng HTTPS nếu chạy trên server từ xa

### Lỗi Gemini API
- Kiểm tra `GEMINI_API_KEY` environment variable
- Kiểm tra quota tại https://aistudio.google.com/

### Lỗi TTS / Piper

**Lỗi "Piper executable not found":**
```bash
# Tải Piper binary (xem mục 4.1)
cd models/piper
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz && mv piper/* . && rm -rf piper piper_amd64.tar.gz
```

**Lỗi shared library không tìm thấy:**
```bash
export LD_LIBRARY_PATH="/path/to/models/piper:$LD_LIBRARY_PATH"
```

**Kiểm tra Piper hoạt động:**
```bash
cd models/piper
echo "Xin chào" | ./piper -m vi_VN-vais1000-medium.onnx -f test.wav
aplay test.wav  # hoặc mở file test.wav
```

**Các lỗi khác:**
- Kiểm tra model Piper đã được tải về đúng thư mục
- Kiểm tra file `.onnx` và `.onnx.json` cùng tên

## 📝 License

MIT License

