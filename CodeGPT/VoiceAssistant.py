from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import speech_recognition as sr


class VoiceAssistant:
    def __init__(self):
        load_dotenv()
        self.apikey = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.apikey)

    # Text to speech using open ai
    def TTS(self, output, outpath):
        response = self.client.audio.speech.create(
            model = "tts-1-hd",
            voice = "echo",  
            input = output,
            speed = 1  # Normal speed
        )
        response.stream_to_file(outpath)

    # Speech to text using open ai
    def STT(self, speech_file_path):
        response = self.client.audio.transcriptions.create(
            model = "whisper-1",
            file = speech_file_path
        )
        # print(response.text)
        return response.text

    #local Speech to text
    def localSTT(self):
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
