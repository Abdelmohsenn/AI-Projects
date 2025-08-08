import os
from dotenv import load_dotenv
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import io, json
from systemPrompt import systemPrompt

class Kira:
    def __init__(self, client, modelName, temperature=0.7, max_tokens=2048):
        self.client = client
        self.modelName = modelName
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __repr__(self):
        if self.apikey is None:
            return f"Kira(client={self.client}, apikey={self.apikey}, modelName={self.modelName}, temperature={self.temperature}, max_tokens={self.max_tokens})"
        else:
            return f"Kira(client={self.client}, modelName={self.modelName}, temperature={self.temperature}, max_tokens={self.max_tokens})"

    def initEnv(self):
        load_dotenv()
        self.apikey = os.getenv("OPENAI_API_KEY", "")
        if not self.apikey:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    def setClient(self):
        self.initEnv()
        self.client = OpenAI(api_key=self.apikey)

    def response(self, query, sysPrompt):
        self.setClient()
        unstructured_response = self.client.chat.completions.create(
            model=self.modelName,
            messages=[
                {"role": "user", "content": query},
                {"role": "system", "content": sysPrompt}
            ]
        )
        structured_response = unstructured_response.choices[0].message.content.strip()
        return structured_response

    def STT(self, audioBytes):

        if self.client is None:
            self.setClient()
        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audioBytes,
            response_format="text",
            language="en"
        )
        return response

    def recordAudio(self, seconds, fs=44100):
        print("Recording...")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  
        buffer = io.BytesIO()
        sf.write(buffer, myrecording, fs, format='mp3')
        buffer.seek(0)  
        buffer.name = "audio.mp3"  
        return buffer 