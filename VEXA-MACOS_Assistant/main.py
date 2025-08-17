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


vexa = Client.Vexa(client=None, modelName="gpt-4o")
# vexa.initializeWAVs()
Chunk, Format, channels, rate, seconds = 1024, pyaudio.paInt16, 1, 16000, 2
endVoices = os.listdir("VEXA-MACOS_Assistant/WAVs/vexa_end")
startVoices = os.listdir("VEXA-MACOS_Assistant/WAVs/vexa_intro")
ErrorVoices = os.listdir("VEXA-MACOS_Assistant/WAVs/vexa_error")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUDIO_PATH = "VEXA-MACOS_Assistant/WAVs/vexa_error/3.wav"
transcript_buf = []
lock = threading.Lock()
recording_enabled = threading.Event()
URL = "wss://api.openai.com/v1/realtime?intent=transcription"
HEADERS = [
    "Authorization: Bearer " + OPENAI_API_KEY,
    "OpenAI-Beta: realtime=v1"
]

def realTimeRecord(ws):
    p = pyaudio.PyAudio()
    mic = p.open(channels=channels, format=Format, rate=rate, input=True, frames_per_buffer=Chunk)
    try:
        audiobytes = []
        print("Recording...")
        for i in range(0, int(rate / Chunk * seconds)):
            data = mic.read(Chunk)
            audiobytes.append(data)

        audiobytes = b''.join(audiobytes)
        ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audiobytes).decode("ascii")
        }))
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    finally:
        mic.stop_stream()
        mic.close()
        p.terminate()

def load_audio_as_pcm16(path):

    audio = realTimeRecord()
    return audio

def on_open(ws):
    print("Connected!")
    recording_enabled.set()  # allow recording at start
    session_update = {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "gpt-4o-mini-transcribe",
                "language": "en"
            }
        }
    }
    ws.send(json.dumps(session_update))

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
            print("xx")
            randomErrorVoice = random.choice(ErrorVoices)
            sound = AudioSegment.from_file(f"VEXA-MACOS_Assistant/WAVs/vexa_error/{randomErrorVoice}")
        else:
            print("yy")
            randomEndVoice = random.choice(endVoices)
            sound = AudioSegment.from_file(f"VEXA-MACOS_Assistant/WAVs/vexa_end/{randomEndVoice}")
        play(sound)

def on_message(ws, message):   

    try:
        data = json.loads(message)
    except Exception:
        print("Non-JSON message:", message)

    t = data.get("type")
    if t == "error":
        # print("ERROR:", json.dumps(data, indent=2))
        pass

    if t == "conversation.item.input_audio_transcription.delta":
        with lock:
            delta = data.get("delta", "")
            transcript_buf.append(delta)

    if t == "conversation.item.input_audio_transcription.completed":
        with lock:
            final_transcript = "".join(transcript_buf)
            transcript_buf.clear()
            recording_enabled.clear()
            wakeUp("Hello, Vexa", final_transcript)
            recording_enabled.set()
    # print(final_transcript)

def on_error(ws, err):
    print("WebSocket error:", err)

ws = websocket.WebSocketApp(
    url=URL,
    header=HEADERS,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error
)
def record_loop(ws):
    while True:
        recording_enabled.wait()
        realTimeRecord(ws)
        time.sleep(0.1)  # slight delay before next recording
threading.Thread(target=ws.run_forever).start()

record_loop(ws)

