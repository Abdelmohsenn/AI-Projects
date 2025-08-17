import os
from dotenv import load_dotenv
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import numpy
import io, json, time
from systemPrompt import systemPrompt

class Vexa:
    def __init__(self, client, modelName, temperature=0.7, max_tokens=2048):
        self.client = client
        self.modelName = modelName
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __repr__(self):
        return f"Kira(client={self.client},\n modelName={self.modelName},\n temperature={self.temperature},\n max_tokens={self.max_tokens})"

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
    
    # def recordAudio(self, seconds, fs=44100):
    #     print("Vexa is Listening...")
    #     myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    #     sd.wait()  
    #     buffer = io.BytesIO()
    #     sf.write(buffer, myrecording, fs, format='mp3')
    #     buffer.seek(0)  
    #     buffer.name = "audio.mp3"  
    #     return buffer 

    def recordAudio(self, fs=44100, silence_thresh=0.7, silence_duration=2.0):
        print("Vexa is Listening... (speak now)")

        recording = []
        silence_start = None
        chunk_size = int(fs * 0.2)

        with sd.InputStream(samplerate=fs, channels=1) as stream:
            while True:
                # read small chunk
                chunk, _ = stream.read(chunk_size)
                recording.append(chunk)

                # compute volume (RMS)
                rms = numpy.sqrt(numpy.mean(chunk**2))

                if rms < silence_thresh:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        print("Silence detected, stopping...")
                        break
                else:
                    silence_start = None  # reset silence timer

        # concatenate chunks
        myrecording = numpy.concatenate(recording, axis=0)

        # save to buffer as MP3
        buffer = io.BytesIO()
        sf.write(buffer, myrecording, fs, format='mp3')
        buffer.seek(0)
        buffer.name = "audio.mp3"

        return buffer

    def talk(self, text, path):
        if self.client is None:
            self.setClient()
        response = self.client.audio.speech.create(
        model="tts-1-hd",
        voice="coral",  
        input=text,
        speed=1  # Normal speed 
        )
        response.stream_to_file(path)

    def initializeWAVs(self):
        start_phrases = [
            "I am Listening",
            "Go ahead, I'm all ears.",
            "Ready when you are.",
            "Listening closely.",
            "What can I do for you?",
            "Awaiting your instructions.",
            "You speak, I act.",
            "Ears on, brain ready.",
            "Command center online."
        ]

        end_phrases = [
            "Got it.",
            "Done and dusted.",
            "Consider it Done.",
            "All set.",
            "Task completed.",
            "Understood loud and clear.",
            "Mission Completed.",
            "Execution successful.",
            "Done. What's next?"
        ]
        errorPhrases = [
            "Sorry, I didn't catch that.",
            "Could you please repeat that?",
            "I'm having trouble understanding.",
            "Can you say that again?",
            "I didn't quite get that.",
            "I'm not sure I understand."
        ]

        if not os.path.exists("VEXA-MACOS_Assistant/WAVs/vexa_intro")\
            or not os.path.exists("VEXA-MACOS_Assistant/WAVs/vexa_end")\
            or not os.path.exists("VEXA-MACOS_Assistant/WAVs/vexa_error"):
            os.makedirs("VEXA-MACOS_Assistant/WAVs/vexa_intro")
            os.makedirs("VEXA-MACOS_Assistant/WAVs/vexa_end")
            os.makedirs("VEXA-MACOS_Assistant/WAVs/vexa_error")

        for i, phrase in enumerate(start_phrases):
            self.talk(phrase, f"VEXA-MACOS_Assistant/WAVs/vexa_intro/{i+1}.wav")
        for i, phrase in enumerate(end_phrases):
            self.talk(phrase, f"VEXA-MACOS_Assistant/WAVs/vexa_end/{i+1}.wav")
        for i, phrase in enumerate(errorPhrases):
            self.talk(phrase, f"VEXA-MACOS_Assistant/WAVs/vexa_error/{i+1}.wav")