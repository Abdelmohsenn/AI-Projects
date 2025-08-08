import speech_recognition as sr
import subprocess
from main import execute
from rapidfuzz import fuzz

wakeUpWord = "Vexa"
r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    r.adjust_for_ambient_noise(source)

def wakeUp():
    with mic as source:
        audio = r.listen(source)
    try:
        text = "Vexa"
        print(text)
        if fuzz.ratio(wakeUpWord, text) > 50:
            print("Wake word detected!")
            execute()
    except sr.RequestError as e:
        pass
    except sr.UnknownValueError:
        pass


while True:
    wakeUp()    
