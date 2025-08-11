import speech_recognition as sr
import subprocess
from Execution import execute
from rapidfuzz import fuzz
from pydub.playback import play
from pydub import AudioSegment
import os, random
import websocket
import Client

vexa = Client.Vexa(client=None, modelName="gpt-4o")
# vexa.initializeWAVs()
print(vexa)
endVoices = os.listdir("VEXA-MACOS_Assistant/WAVs/vexa_end")
startVoices = os.listdir("VEXA-MACOS_Assistant/WAVs/vexa_intro")
ErrorVoices = os.listdir("VEXA-MACOS_Assistant/WAVs/vexa_error")
print(startVoices)

r = sr.Recognizer()
mic = sr.Microphone()

def wakeUp(wakeUpWord):
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        text = "Vexa"
        print(text)
        randomStartSound = random.choice(startVoices)   

        sound = AudioSegment.from_file(f"VEXA-MACOS_Assistant/WAVs/vexa_intro/{randomStartSound}")
        if fuzz.ratio(wakeUpWord, text) > 50:
            print("Wake word detected!")
            play(sound)
            error = execute(client = vexa)
            if error:
                randomErrorVoice = random.choice(ErrorVoices)
                sound = AudioSegment.from_file(f"VEXA-MACOS_Assistant/WAVs/vexa_error/{randomErrorVoice}")
            else:
                randomEndVoice = random.choice(endVoices)
                sound = AudioSegment.from_file(f"VEXA-MACOS_Assistant/WAVs/vexa_end/{randomEndVoice}")
            play(sound)

    except sr.RequestError as e:
        pass
    except sr.UnknownValueError:
        pass

while True:
    wakeUp("Vexa")
