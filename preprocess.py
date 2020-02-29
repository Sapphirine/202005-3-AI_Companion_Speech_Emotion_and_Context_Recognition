import os
import numpy as np
import soundfile
import pickle
import librosa
import scipy.io.wavfile
from scipy.fftpack import dct
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sounddevice as sd


def splitSample(X, y, window = 0.1, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input sample array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the sample and create new ones on windows
    spsample = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsample:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)

def getSignal(file_name):
	X, sample_rate = librosa.load(file_name)	
	return X, sample_rate
	
def addNoise(data):
	"""
	Add White noise to the signal
	"""
	noise_amp = 0.005*np.random.uniform()*np.amax(data)
	data = data.astype('float64') + noise_amp*np.random.normal(size=data.shape[0])
	return data

def shift(data):
	"""
	Adding random shift to data
	"""
	s_range = int(np.random.uniform(low=-5, high=5)*500)
	return np.roll(data,s_range)
	
def getFeatures(X, sample_rate, mfcc, chroma, mel):
	# Compute features from sound file
	# mfcc = Mel Frequency Cepstral Coefficeints (short-term power spectrum of a sound)
	# chroma = 12 different pitch classes
	# mel = Mel Spectrogram Frequency

	
	result=np.array([])
	
	if mfcc:
		mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, hop_length=512, n_mfcc=40).T, axis=0)
		result = np.hstack((result, mfccs))
		
	if chroma:
		stft = np.abs(librosa.stft(X))
		chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
		result = np.hstack((result, chroma))
		
	if mel: 
		mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
		result = np.hstack((result, mel))
	return result
	
def plotFreq(list):
	df = pd.DataFrame({'freq': list})
	df.groupby('freq', as_index=False).size().plot(kind='bar')
	plt.show()
	
def playAudio(data, sample_rate):
	sd.play(data, sample_rate)
	
def saveAudio(filepath, data, sample_rate):
	scipy.io.wavfile.write(filepath, sample_rate, data)
	
#def readData(emotions, num_samples, dir, n_fft = 1024, hop_length = 512):

def readData(emotions, dir, test_size=0.2):
	# currently configured for RAVDESS dataset

	#check that directory to data is correct
	#print(os.listdir(dir)) # check what files are available
	#emotionlst = []
	#mfcclst = []
	#pitchlst = []
	
	x = []
	y = []
	
	num_actors = 24 #total number in RAVDESS is 24
	
	for i in range(num_actors):
		if i < 9:
			path = dir+'/Actor_0'+str(i+1)
		else:
			path = dir+'/Actor_'+str(i+1)
			
		print(path)
		count = 0
		for filename in os.listdir(path):
			#print(filename)
			if filename.endswith('.wav'):
				count=count+1
				
		print('wav files found = ',count)
	
		for root, subdirs, files in os.walk(path):
			for file in files:
				# Read the audio file
				file_name = path + "/" + file
				
				# get info on file
				emotion = file[6:8]
				
				X, sample_rate = getSignal(file_name)
				
				feature=getFeatures(X, sample_rate, mfcc=True, chroma=True, mel=True)
				
				x.append(feature)
				y.append(emotion)
				
				# data augmentation
				X_noise = addNoise(X) # add white noise
				#playAudio(X_noise, sample_rate)
				
				X_shift = shift(X) # add random shift
				
				X_NandS = shift(X_noise) # noise and shift
				
				feature1=getFeatures(X_noise, sample_rate, mfcc=True, chroma=True, mel=True)
				
				feature2=getFeatures(X_shift, sample_rate, mfcc=True, chroma=True, mel=True)
				
				feature3=getFeatures(X_NandS, sample_rate, mfcc=True, chroma=True, mel=True)
				
				x.append(feature1)
				y.append(emotion)
				
				x.append(feature2)
				y.append(emotion)
				
				x.append(feature3)
				y.append(emotion)

	return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
	
	
if __name__ == '__main__':
	num_samples = 660000
	dir_path = 'data_orig'
	emotions = {'01':'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
	
	#features, pitch_list, labels = readData(emotions, num_samples, dir_path)
	
	x_train, x_test, y_train, y_test = readData(emotions, dir_path, test_size=0.2)
	
	print((x_train.shape[0], x_test.shape[0]))
	print('Features extracted: ', x_train.shape[1])
	
	#plotFreq(labels)