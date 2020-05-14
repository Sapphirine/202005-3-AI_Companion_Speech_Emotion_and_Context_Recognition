
import tensorflow as tf
import re
import tensorflow_datasets as tfds
import os
import numpy as np 
import time
import cv2

import keras
from keras.models import Model, load_model
import matplotlib
from keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import img_to_array, load_img
import pickle as pkl 
from collections import deque

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM, BatchNormalization
import dlib

import pyaudio
import wave
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import librosa as lib
import sys
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from multiprocessing import Process, Queue, Pool, Event, Pipe, Lock, Value, Array, Manager
#import multiprocessing

import test_audio
import test_video
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

from chatbot import *
from speechtotext import *
from texttospeech import *
from collections import deque

# os.environ['KMP_DUPLICATE_LIB_OK']='True'


# NUM_PARALLEL_EXEC_UNITS = 8
# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)
# os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
# os.environ["KMP_BLOCKTIME"] = "30"
# os.environ["KMP_SETTINGS"] = "1"
# os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"


def preprocess_sentence(sentence):
	sentence = sentence.lower().strip()
	# creating a space between a word and the punctuation following it
	# eg: "he is a boy." => "he is a boy ."
	sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
	sentence = re.sub(r'[" "]+', " ", sentence)
	# replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
	sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
	sentence = sentence.strip()
	# adding a start and an end token to the sentence
	return sentence    
	
def evaluate(model, sentence, emotion, tokenizer, START_TOKEN, END_TOKEN):
	MAX_LENGTH = 40
	
	sentence = preprocess_sentence(sentence)

	sentence = tf.expand_dims(
	  START_TOKEN + tokenizer.encode(sentence) + END_TOKEN + tokenizer.encode(emotion), axis=0)

	output = tf.expand_dims(START_TOKEN, 0)

	for i in range(MAX_LENGTH):
		predictions = model(inputs=[sentence, output], training=False)

		# select the last word from the seq_len dimension
		predictions = predictions[:, -1:, :]
		predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

		# return the result if the predicted_id is equal to the end token
		if tf.equal(predicted_id, END_TOKEN[0]):
			break

		# concatenated the predicted_id to the output which is given to the decoder
		# as its input.
		output = tf.concat([output, predicted_id], axis=-1)

	return tf.squeeze(output, axis=0)

def preprocess_tokenize():
	emotions_lst, clean_questions, clean_answers, emotions_val_lst, clean_questions_val, clean_answers_val = preprocess()
	tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(emotions_lst+clean_questions + clean_answers, target_vocab_size=2**13)
	return tokenizer

def predict(model, sentence, emotion):
	
	emotion_list = ['neutral', 'angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised']
	#emotion_list, clean_questions, clean_answers = preprocess()
	# Build tokenizer using tfds for both questions and answers
	# Define start and end token to indicate the start and end of a sentence
	
	START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
	prediction = evaluate(model, sentence, emotion, tokenizer, START_TOKEN, END_TOKEN)

	predicted_sentence = tokenizer.decode(
	  [i for i in prediction if i < tokenizer.vocab_size])

	#print('Input: {}'.format(sentence))
	#print('Output: {}'.format(predicted_sentence))

	return predicted_sentence

def chatbot_predict(sentence, emotion):
	#video_emotion= str(predicted_emotion.get())
	#print("emotion detected is", video_emotion)
	predicted_sentence = predict(model, sentence, emotion)
	# time.sleep(0.5)
	# predict(model, 'I have been feeling sad.', predicted_emotion.get())
	# time.sleep(0.5)
	# predict(model, 'I have been feeling sad.', 'neutral')
	# time.sleep(0.5)
	return predicted_sentence

# def process_video_onlyCNN(queue, predicted_emotion, new_model):
# 	#try:
# 	frame = queue.get()
# 	pred = new_model.predict(np.expand_dims(frame, axis=0))
# 	i = np.argmax(pred)
# 	#labels_mapping = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4: "Sad", 5: "Surprise"}
# 	labels_mapping = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
# 	#print("emotion from video", labels_mapping[i])
# 	predicted_emotion.put(labels_mapping[i])
# 	#pred_emotion2.put(labels_mapping[i])
# 	return pred, labels_mapping[i]
# 	# except:
# 	# 	print("queue empty")
# 	# 	pass

def process_video_CNN_frame(frames):
	pred = 0
	for frame in frames:
		pred += new_model.predict(np.expand_dims(frame, axis=0))
	pred = pred/len(frames)
	i = np.argmax(pred)
	#labels_mapping = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4: "Sad", 5: "Surprise"}
	#labels_mapping = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
	labels_mapping = {0:"angry", 1:"disgust", 2:"fearful", 3:"happy", 4: "neutral", 5: "sad", 6: "surprised"}
	
	#predicted_emotion.put(labels_mapping[i])
	#pred_emotion2.put(labels_mapping[i])
	return pred, labels_mapping[i]

if __name__ == "__main__":
	
	model = get_model(NUM_LAYERS = 2, D_MODEL = 256, NUM_HEADS = 8, UNITS = 512, DROPOUT = 0.1)
	model_path = "audvid_models/custom_vgg_model6.h5"
	new_model = keras.models.load_model(model_path, compile = False)

	#t0 = time.time()
	#tokenizer = preprocess_tokenize()
	#t1 = time.time()
	#print("tokenizationt time", t1 - t0)

	with open("chatbot_models/model/tokenizer.pickle", "rb") as input_file:
		tokenizer = pkl.load(input_file)
   
	#predicted_emotion = Queue()
	#predicted_emotion.put("neutral")

	queue = Queue() 

	#e = Event()
	#p1 = Process(target = test_video.get_live_video, args = (queue,e))
	#p1.start()
	#time.sleep(0.1)
	assistant_speaks("Hi Prutha, What's up?") 

	# while True:
	# 	recent_frames = deque()
	# 	#pred, emotion = process_video_onlyCNN(queue, predicted_emotion, new_model)
	# 	e.set()
	# 	text = transcribe_audio()
	# 	e.clear()
	# 	if text != None:
	# 		if text == "okay bye":
	# 			#exit out of program! 
	# 			assistant_speaks("okay bye, talk soon!") 
	# 			break

	# 		#take the most recent frame from queue -> fetch from queue and dump into a stack and pop
	# 		queue_not_empty = True
	# 		time.sleep(1)
	# 		while queue_not_empty:
	# 			try:
	# 				frame = queue.get(False)
	# 				#print(np.shape(frame))
	# 				recent_frames.append(frame)
	# 			except:
	# 				queue_not_empty = False
	# 		print(len(recent_frames))
	# 		#frame = recent_frames.pop()
	# 		frames = [recent_frames.pop() for _ in range(len(recent_frames))]
	# 		pred, emotion = process_video_CNN_frame(frames)

	# 		print(text)
	# 		print("detected emotion: "+ str(emotion))

	# 		response = chatbot_predict(text,emotion)
	# 		assistant_speaks(response) 

	video = cv2.VideoCapture(0)
	detector = dlib.get_frontal_face_detector()
	while True:
		recent_frames = deque()
		for i in range(5):
			(grabbed,frame) = video.read()
			if not grabbed :
				print("not grabbed")
				break
		
			faces = detector(frame)
			frame_copy = frame.copy()

			for face in faces:
				x1 = face.left()
				y1 = face.top()
				x2 = face.right()
				y2 = face.bottom()
				face = frame[y1:y2, x1:x2]

				cv2.rectangle(frame_copy,(x1,y1),(x2,y2),(255,0,0),thickness=7)

				cv2.imshow('Frame',frame_copy)
				frame = cv2.resize(face, (224, 224)).astype("float32")
				norm_image = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
				recent_frames.append(norm_image)
			if cv2.waitKey(10) == ord('q') : #wait until keyboard interrupt
				break
				cv2.destoryAllWindows()
			
		text = transcribe_audio()
		if text != None:
			if text == "okay bye":
	 			#exit out of program! 
				assistant_speaks("okay bye, talk soon!") 
				break

			pred, emotion = process_video_CNN_frame(recent_frames)
			cv2.putText(frame_copy, emotion, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
			print(text)
			print("detected emotion: "+ str(emotion))

			response = chatbot_predict(text,emotion)
			assistant_speaks(response) 


	#p1.join()
	#p2.join()
	cv2.destroyAllWindows()
	
