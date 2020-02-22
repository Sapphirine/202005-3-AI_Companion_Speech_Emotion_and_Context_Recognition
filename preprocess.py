import os
import numpy as np
import librosa
import scipy.io.wavfile
from scipy.fftpack import dct
import pandas as pd
import matplotlib.pyplot as plt

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

def getMFCC(samples, sr, hop_length=512):
	# Compute MFCC features from the raw signal
	mfcc = librosa.feature.mfcc(y=samples, sr=sr, hop_length=hop_length, n_mfcc=13)
	return mfcc
	
def getPitch(samples, sr, hop_length=512):
	# Compute MFCC features from the raw signal
	pitches, magnitudes = librosa.core.piptrack(y=samples, sr=sr, fmin=75, fmax=1600)
	
	np.set_printoptions(threshold=np.nan)
	#print(pitches[np.nonzero(pitches)])
	return pitches

	
def plotFreq(list):
	df = pd.DataFrame({'freq': list})
	df.groupby('freq', as_index=False).size().plot(kind='bar')
	plt.show()
	

def readData(emotions, num_samples, dir, n_fft = 1024, hop_length = 512):
	# currently configured for RAVDESS dataset

	#check that directory to data is correct
	#print(os.listdir(dir)) # check what files are available
	emotionlst = []
	mfcclst = []
	pitchlst = []
	
	num_actors = 2 #total number in RAVDESS is 24
	
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
				#emotionlst.append(emotions[emotion])
				
				#print(file_name)

				signal, sr = librosa.load(file_name) # signal is 1-dimensional array, sr is sampling rate
				#signal = signal[:num_samples]
				
				# Convert to dataset of spectograms/melspectograms
				signals, y = splitSample(signal, emotion)
				
				for emote in y:
					emotionlst = np.append(emotionlst, emotions[emote])
				
				for s in signals:
					# Get MFCC of signal sample
					mfcc = getMFCC(s, sr)
					mfcclst.append(mfcc)
					pitch = getPitch(s, sr)
					pitchlst.append(pitch)
				# Convert to "spec" representation
				#specs = to_melspectrogram(signals, n_fft, hop_length)
		
		#print(mfcc)
		print(emotionlst.shape)
		print(len(mfcclst))  #may need to play around with how this is being stored
		print(len(pitchlst))
	#plotFreq(emotionlst)
	
	return(mfcclst, pitchlst, emotionlst)
	
	
if __name__ == '__main__':
	num_samples = 660000
	dir_path = 'data'
	emotions = {'01':'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
	
	features, pitch_list, labels = readData(emotions, num_samples, dir_path)
	
	plotFreq(labels)
	
	
	# to do:
	# other features - energy, pitch (frequency), LPCC