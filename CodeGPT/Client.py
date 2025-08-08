from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import speech_recognition as sr

load_dotenv()
apikey = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Set up file paths
speech_file_path = Path(__file__).parent / "speech.wav"
output_file_path = Path(__file__).parent / "CodeGPT.wav"

# Text to speech using open ai
def TTS(output, outpath):
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="echo",  
        input=output,
        speed=1  # Normal speed
    )
    response.stream_to_file(outpath)

# Speech to text using open ai
def STT():
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=speech_file_path
    )
    # print(response.text)
    return response.text

#local Speech to text
def localSTT():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("CodeGPT is Listening...")
        r.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = r.listen(source)  # Listen in real-time
                text = r.recognize_google(audio)  # Convert to text
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Could not understand.")
            except sr.RequestError:
                print("API unavailable.")
