# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:23:42 2020

@author: KAVITA
"""

import pyaudio
import wave
import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import pickle as pkl
import librosa as lib
import numpy as np
from keras.models import load_model
#import speech_recognition as spr
import sys
#from plot_accuracy import *
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


# def predict_emotion():
#     model=load_model('model_best.h5',compile=False)
#     print("debug model load")
#     with open('test_file.pkl', 'rb') as f:
#         audio_element = pkl.load(f)
#     x_test=np.transpose(audio_element['audio'])
#     x_test=x_test.reshape([1,x_test.shape[0],x_test.shape[1]])
#     #print(x_test.shape)
#     y_pred=model.predict_classes(x_test)
#     #print(y_pred)
#     emotions = ['anger', 'disgust','fear', 'happiness', 'sadness', 'surprise','neutral','calm']
#     #emo_ravdess=['neutral','calm', 'happy', 'sad','angry', 'fearful', 'disgust', 'surprised']
#     #print(y_pred)
#     print('Emotion class=',emotions[y_pred[0]])
#     #print('Ravdess emotion class=',emo_ravdess[y_pred[0]])

def get_audio_features(queue_stream):
    destination_filepath="output.wav"
    #feature_files= 'D:\Facial_Recognition'

    #video = VideoFileClip(source_filepath) 
    #audio_part = video.audio
    #audio_part.write_audiofile(destination_filepath)
    
    #sample_rate, samples = wavfile.read(audio)
    #frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    
    ##### read audio as vector
    
    audio_input,sr=lib.load(destination_filepath)
    #non_silence=lib.effects.split(y=audio_input, frame_length=32, top_db=40)
    #print(non_silence.shape)
    #non_silence=np.asarray([float(x) for x in non_silence[0]])
    #print(non_silence.shape)

    #print(sr)
    #print(audio_input)
    #print(non_silence)
    ##### Hamming Window and stft
    #audio_stft = lib.stft(non_silence,window='hamm',n_fft=1024)
    #audio_db = librosa.amplitude_to_db(abs(audio_stft))
    #audio_stft = lib.stft(non_silence,window='hamm',n_fft=1024)
    audio_mfcc=lib.feature.mfcc(abs(audio_input),n_mfcc=20)
    #print(audio_mfcc.shape)
 
    output_filename = 'test_file.pkl' 
    out = {'audio': audio_mfcc,
       }
    # with open(output_filename, 'wb') as w:
    #     pkl.dump(out, w)

    queue_stream.put(out)


def remove_sil():
    #sound = AudioSegment.from_file(path_in, format=format)
    sound = AudioSegment.from_wav("output.wav")
    output_loc='audio_wav'
    non_sil_times = detect_nonsilent(sound, min_silence_len=1, silence_thresh=sound.dBFS * 1.5)
    if len(non_sil_times) > 0:
        non_sil_times_concat = [non_sil_times[0]]
        if len(non_sil_times) > 1:
            for t in non_sil_times[1:]:
                if t[0] - non_sil_times_concat[-1][-1] < 200:
                    non_sil_times_concat[-1][-1] = t[1]
                else:
                    non_sil_times_concat.append(t)
        non_sil_times = [t for t in non_sil_times_concat if t[1] - t[0] > 350]
        #sound[non_sil_times[0][0]: non_sil_times[-1][1]].export(output_loc)
        #print('audio_exported')
        #print('content audio')
        return sound[non_sil_times[0][0]: non_sil_times[-1][1]]
    else:
        print('silence audio')
        return 0


def adjust_volume(audio_file):
    #audio_file = AudioSegment.from_wav("output.wav")
    audio_file=audio_file+10
    return audio_file
    
    
    
def get_live_input(queue_stream):
    #current_dir='D:\Facial_Recognition\test_audio_wav'
    x=0
    while(True):

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 22050
        RECORD_SECONDS =5
        #save_name="output{}".format(x)+".wav"
        #av_file_directory = os.path.join(current_dir,save_name)
        #av_file_directory = ".\test_audio_wav\output{}".format(x)+".wav"
        output_file_name='output.wav'
        x+=1
        p = pyaudio.PyAudio()
        
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        #print("* recording{}".format(x))
        
        frames = []
        
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            #print(i)
            data = stream.read(CHUNK)
            frames.append(data)
        
        #print("* processing")
        
        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(output_file_name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        #adjust_for_ambient_noise(frames)
        #non_silence_audio=remove_sil()
        #if non_silence_audio!=0:
        #    vol_adjusted_audio_file=adjust_volume(non_silence_audio)
        #with open('output_adjusted.wav', 'wb') as out_f:
        #    vol_adjusted_audio_file.export(out_f, format='wav')
        get_audio_features(queue_stream)
        #predict_emotion()
        
        #final_audio=remove_sil(vol_adjusted_audio_file,"wav")
        #print(type(final_audio))


def get_data(file_list):
    def load_into(_filename, x, y):
        base_dir='D:\Facial_Recognition\Afew_features_encoded\Train'

        with open(os.path.join(base_dir, _filename), 'rb') as f:
            # print(f)
            audio_element = pkl.load(f)
            x.append(np.transpose(audio_element['audio']))
            # print(audio_element['audio'])
            y.append((audio_element['class_label']))

    x, y = [], []
    for filename in file_list:
        # print(filename)
        load_into(filename, x, y)

    return x, y



def get_files(av_file_directory):
    file_list = []
    #file_dir='D:\Facial_Recognition\Afew_features_encoded\Train'
    file_dir=av_file_directory
    for files in os.listdir(file_dir):
        file_list.append(files)

    #fetch_data = Data_preparation(file_list)
    #print(file_list)
    x_test, y_test = get_data(file_list)
    #x_test=np.asarray(x_test)
    le = LabelEncoder()
    y_test=(np.asarray(y_test)).reshape([np.asarray(y_test).shape[0],1])
    y_test=to_categorical(le.fit_transform(y_test))
    x_max=max(len(x_test[x]) for x in range(len(x_test)))
    x_1=x_max
    x_2=len(x_test[0][0])
    x_0=len(x_test)
    x_conv=np.zeros((x_0,x_1,x_2))
    for l in range(len(x_test)):
        if len(x_test[l]) < x_max:
            x_test[l] = np.concatenate((x_test[l], np.zeros(shape=(x_max - len(x_test[l]), x_2))))
        x_conv[l] = x_test[l]

    x_test=np.asarray(x_test)
    print(x_test.shape)
    model = load_model('model_best.h5', compile=False)
    y_pred = model.predict_classes(x_test)
    rounded_labels2=np.argmax(y_test, axis=1)
    plot_confusionMatrix(rounded_labels2,y_pred)


# if __name__ == "__main__":
#     cmdln_input=int(sys.argv[1])
#     if cmdln_input==0:
#         get_live_input()
#     else:
#         get_files(cmdln_input)

#python test_audio.py