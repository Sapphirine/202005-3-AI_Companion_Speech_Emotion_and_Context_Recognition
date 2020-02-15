import os
import numpy as np
import librosa
import scipy.io.wavfile
from scipy.fftpack import dct

# short-time Fourier Transformation(STFT)
def stft(sig, frame_size, overlap_factor=0.5, window=np.hanning):
    win = window(frame_size)
    hop_size = int(frame_size - np.floor(overlap_factor * frame_size))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frame_size / 2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frame_size) / float(hop_size)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frame_size), strides=(samples.strides[0] * hop_size, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames) 

def splitSample(X, y, window = 0.1, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)
	
	
def to_melspectrogram(songs, n_fft = 1024, hop_length = 512):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft = n_fft,
        hop_length = hop_length)[:,:,np.newaxis]

    # map transformation of input speech to melspectrogram using log-scale
    tspeech = map(melspec, speech)
    return np.array(list(tspeech))

def readData(num_samples, dir, n_fft = 1024, hop_length = 512, debug = True):
	# currently configured for RAVDESS dataset

	#check that directory to data is correct
	#print(os.listdir(dir)) # check what files are available
	
	num_actors = 1
	
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

				#print(file_name)

				signal, sr = librosa.load(file_name) # signal is 1-dimensional array, sr is sampling rate
				#signal = signal[:num_samples]
				
				print(signal, file)
		
	
	
	
if __name__ == '__main__':
	num_samples = 660000
	dir_path = 'data'
	readData(num_samples, dir_path)