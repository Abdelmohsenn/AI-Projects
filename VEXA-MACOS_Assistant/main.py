import os
import json
import base64
import time
import pyaudio
import wave
import websocket
from Execution import execute
import threading
from pydub import AudioSegment
from dotenv import load_dotenv
import Client
from pydub.playback import play
import random
from rapidfuzz import fuzz

load_dotenv()

vexa = Client.Vexa(client=None, modelName="gpt-5")
# vexa.initializeWAVs()
Chunk, Format, channels, rate = 1024, pyaudio.paInt16, 1, 24000  # Changed to 24000 for realtime API
endVoices = os.listdir("VEXA-MACOS_Assistant/WAVs/vexa_end")
startVoices = os.listdir("VEXA-MACOS_Assistant/WAVs/vexa_intro")
ErrorVoices = os.listdir("VEXA-MACOS_Assistant/WAVs/vexa_error")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUDIO_PATH = "VEXA-MACOS_Assistant/WAVs/vexa_error/3.wav"
transcript_buf = []
lock = threading.Lock()
recording_enabled = threading.Event()
URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
HEADERS = [
    "Authorization: Bearer " + OPENAI_API_KEY,
    "OpenAI-Beta: realtime=v1"
]

# Global audio objects
p = None
mic = None

def setup_audio():
    global p, mic
    p = pyaudio.PyAudio()
    mic = p.open(
        channels=channels, 
        format=Format, 
        rate=rate, 
        input=True, 
        frames_per_buffer=Chunk,
        stream_callback=audio_callback
    )

def audio_callback(in_data, frame_count, time_info, status):
    global ws
    if recording_enabled.is_set() and ws:
        audio_b64 = base64.b64encode(in_data).decode('utf-8')
        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }
        ws.send(json.dumps(message))
    return (in_data, pyaudio.paContinue)

def on_open(ws):
    print("Connected!")
    recording_enabled.set()
    
    # Fixed session update message
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": "You are a helpful assistant.",
            "voice": "shimmer",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "whisper-1",
                "language": "en"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.7,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200
            }
        }
    }
    ws.send(json.dumps(session_update))
    
    # Start audio recording
    setup_audio()
    mic.start_stream()

def wakeUp(wakeUpWord, final_transcript):
    print(final_transcript)
    randomStartSound = random.choice(startVoices)
    sound = AudioSegment.from_file(f"VEXA-MACOS_Assistant/WAVs/vexa_intro/{randomStartSound}")
    if fuzz.ratio(wakeUpWord, final_transcript) > 40:
        print("Wake word detected!")
        play(sound)
        error = execute(client = vexa)
        print("The error is:", error)
        if error:
            randomErrorVoice = random.choice(ErrorVoices)
            sound = AudioSegment.from_file(f"VEXA-MACOS_Assistant/WAVs/vexa_error/{randomErrorVoice}")
        else:
            randomEndVoice = random.choice(endVoices)
            sound = AudioSegment.from_file(f"VEXA-MACOS_Assistant/WAVs/vexa_end/{randomEndVoice}")
        play(sound)

def on_message(ws, message):   
    try:
        data = json.loads(message)
    except Exception:
        print("Non-JSON message:", message)
        return

    t = data.get("type")
    if t == "error":
        print("ERROR:", json.dumps(data, indent=2))

    # elif t == "conversation.item.input_audio_transcription.delta":
    #     with lock:
    #         delta = data.get("delta", "")
    #         transcript_buf.append(delta)

    elif t == "conversation.item.input_audio_transcription.completed":
        delta = data.get("transcript", "")
        transcript_buf.append(delta)
        with lock:
            final_transcript = "".join(transcript_buf)
            transcript_buf.clear()
            if final_transcript.strip():  # Only process if there's actual content
                recording_enabled.clear()
                wakeUp("Hello, Vexa", final_transcript)
                recording_enabled.set()

def on_error(ws, err):
    print("WebSocket error:", err)

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")
    cleanup()

def cleanup():
    global p, mic
    recording_enabled.clear()
    if mic:
        mic.stop_stream()
        mic.close()
    if p:
        p.terminate()

ws = websocket.WebSocketApp(
    url=URL,
    header=HEADERS,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

try:
    ws.run_forever()
except KeyboardInterrupt:
    print("\nShutting down...")
    cleanup()