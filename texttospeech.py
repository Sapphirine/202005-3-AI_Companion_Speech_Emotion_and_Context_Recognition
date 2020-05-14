import speech_recognition as sr
import playsound # to play saved mp3 file 
from gtts import gTTS # google text to speech 
import os # to save/open files 

# suppress warning for certificate verification
import requests
requests.packages.urllib3.disable_warnings() 

#num = 1
def assistant_speaks(output): 
    #global num 
  
    # num to rename every audio file  
    # with different name to remove ambiguity 
    #num += 1
    print("Chatbot : ", output) 
  
    toSpeak = gTTS(text = output, lang ='en', slow = False) 
    # saving the audio file given by google text to speech 
    file = str('audfile')+".mp3"
    toSpeak.save(file) 
      
    # playsound package is used to play the same file. 
    playsound.playsound(file, True)  
    os.remove(file) 
    
# if __name__ == "__main__": 
#     assistant_speaks("Hello, what's your name?") 
#     