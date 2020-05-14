import speech_recognition as sr
import pyaudio
import audioop
import os
import math
from os import system
import threading


def transcribe_audio():
    with sr.Microphone() as source:
        # read the audio data from the default microphone
        r = sr.Recognizer()
        #print("Chatbot : Hey! How can I help?")
        print("recording ...")
        audio_data = r.record(source, duration=5) 
        # convert speech to text
        text = r.recognize_google(audio_data, show_all = True)
        try:
            text = text['alternative'][0]['transcript']
            #print(text)
            return text
        except:
            print("can't hear you, speak up")
            pass
    return None

    
    # with open("recorded.wav", "wb") as f:
    #     f.write(audio_data.get_wav_data())


