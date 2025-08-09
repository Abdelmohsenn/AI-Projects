import speech_recognition as sr
import subprocess
from main import execute
from rapidfuzz import fuzz
from pydub.playback import play
from pydub import AudioSegment
import os
from Client import Vexa

vexa = Vexa(client=None, modelName="gpt-4o")
r = sr.Recognizer()
mic = sr.Microphone()
with mic as source:
    r.adjust_for_ambient_noise(source)

def wakeUp(wakeUpWord):
    with mic as source:
        audio = r.listen(source)
    try:
        text = sr.recognize_google(audio)
        print(text)
        sound = AudioSegment.from_file("VEXA-MACOS_Assistant/WAVs/vexa_intro.wav")
        if fuzz.ratio(wakeUpWord, text) > 50:
            print("Wake word detected!")
            play(sound)
        sound = AudioSegment.from_file("VEXA-MACOS_Assistant/WAVs/vexa_end.wav")
        execute(client = vexa)
        play(sound)
    except sr.RequestError as e:
        pass
    except sr.UnknownValueError:
        pass


while True:
    wakeUp("Vexa")    
    