import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
import os



def clean_text(text):
	'''Clean text by removing unnecessary characters and altering the format of words.'''

	text = text.lower()
	
	text = re.sub(r"i'm", "i am", text)
	text = re.sub(r"i'll", "i will", text)
	text = re.sub(r"he's", "he is", text)
	text = re.sub(r"she's", "she is", text)
	text = re.sub(r"it's", "it is", text)
	text = re.sub(r"that's", "that is", text)
	text = re.sub(r"what's", "that is", text)
	text = re.sub(r"where's", "where is", text)
	text = re.sub(r"how's", "how is", text)
	text = re.sub(r"\’ ll ", " will", text)
	text = re.sub(r"\’ll ", " will", text)
	text = re.sub(r"\’ ve", " have", text)
	text = re.sub(r"\'re", " are", text)
	text = re.sub(r"\'d", " would", text)
	text = re.sub(r"\’ re", " are", text)
	text = re.sub(r"won ' t", "will not", text)
	text = re.sub(r"can ' t", "cannot", text)
	text = re.sub(r"don ’ t", "do not", text)
	text = re.sub(r"’ s", "is", text)
	text = re.sub(r"n ' t", " not", text)
	text = re.sub(r"n'", "ng", text)
	text = re.sub(r"'bout", "about", text)
	text = re.sub(r"'til", "until", text)
	text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
	
	return text

def preprocess():
	# Load the data
	path = "data_chatbot/daily_dialogue"
	# for filename in os.listdir(path):
	# 	print(filename)
	
	lines = open('data_chatbot/daily_dialogue/dialogues_text.txt', encoding='utf-8', errors='ignore').read().split('\n')
	emotions = open('data_chatbot/daily_dialogue/dialogues_emotion.txt', encoding='utf-8', errors='ignore').read().split('\n')
	lines_val = open('data_chatbot/daily_dialogue/dialogues_validation.txt', encoding='utf-8', errors='ignore').read().split('\n')
	emotions_val = open('data_chatbot/daily_dialogue/dialogues_emotion_validation.txt', encoding='utf-8', errors='ignore').read().split('\n')
		
	emotions_dic = {'0':'neutral', '1': 'angry', '2': 'disgust', '3': 'fearful', '4': 'happy', '5': 'sad', '6': 'surprised'}
	convs = [ ]
	for line in lines:
		_line = line.split(' __eou__')
		convs.append(_line[:-1])
		
	convs_val = [ ]
	for line in lines_val:
		_line = line.split(' __eou__')
		convs_val.append(_line[:-1])

	emotes = [ ]
	for emotion in emotions:
		_line = emotion.split(' ')
		emotes.append(_line[:-1])
	
	emotes_val = [ ]
	for emotion_val in emotions_val:
		_line = emotion_val.split(' ')
		emotes_val.append(_line[:-1])
		
	# Sort the sentences into questions (inputs) and answers (targets)
	questions = []
	answers = []
	emotions_lst = []
	count = 0
	for conv in convs:
		for i in range(len(conv)-1):
			questions.append(conv[i])
			answers.append(conv[i+1])
			# CHANGE THIS to append actual emotion int
			
	for emot in emotes:
		for i in range(len(emot)-1):
			emotions_lst.append(emotions_dic[emot[i]])


	questions_val = []
	answers_val = []
	emotions_val_lst = []
	count = 0
	for conv in convs_val:
		for i in range(len(conv)-1):
			questions_val.append(conv[i])
			answers_val.append(conv[i+1])
			
	for emot in emotes_val:
		for i in range(len(emot)-1):
			emotions_val_lst.append(emotions_dic[emot[i]])
		
	# Clean the data
	clean_questions = []
	for question in questions:
		clean_questions.append(clean_text(question))
		
	clean_answers = []    
	for answer in answers:
		clean_answers.append(clean_text(answer))

	emotions = []
	emotions_key = emotions_dic.keys()
	for i in range(len(emotions_key)):
		emotions.append(emotions_dic[str(i)])

	clean_questions_val = []
	for question in questions_val:
		clean_questions_val.append(clean_text(question))
		
	clean_answers_val = []    
	for answer in answers_val:
		clean_answers_val.append(clean_text(answer))
	
	return emotions_lst, clean_questions, clean_answers, emotions_val_lst, clean_questions_val, clean_answers_val


def scaled_dot_product_attention(query, key, value, mask):
	"""Calculate the attention weights. """
	matmul_qk = tf.matmul(query, key, transpose_b=True)

	# scale matmul_qk
	depth = tf.cast(tf.shape(key)[-1], tf.float32)
	logits = matmul_qk / tf.math.sqrt(depth)

	# add the mask to zero out padding tokens
	if mask is not None:
		logits += (mask * -1e9)

	# softmax is normalized on the last axis (seq_len_k)
	attention_weights = tf.nn.softmax(logits, axis=-1)

	output = tf.matmul(attention_weights, value)

	return output

class MultiHeadAttention(tf.keras.layers.Layer):

	def __init__(self, d_model, num_heads, name="multi_head_attention"):
		super(MultiHeadAttention, self).__init__(name=name)
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.query_dense = tf.keras.layers.Dense(units=d_model)
		self.key_dense = tf.keras.layers.Dense(units=d_model)
		self.value_dense = tf.keras.layers.Dense(units=d_model)

		self.dense = tf.keras.layers.Dense(units=d_model)

	def split_heads(self, inputs, batch_size):
		inputs = tf.reshape(
			inputs, shape=(batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(inputs, perm=[0, 2, 1, 3])

	def call(self, inputs):
		query, key, value, mask = inputs['query'], inputs['key'], inputs[
			'value'], inputs['mask']
		batch_size = tf.shape(query)[0]

		# linear layers
		query = self.query_dense(query)
		key = self.key_dense(key)
		value = self.value_dense(value)

		# split heads
		query = self.split_heads(query, batch_size)
		key = self.split_heads(key, batch_size)
		value = self.split_heads(value, batch_size)

		# scaled dot-product attention
		scaled_attention = scaled_dot_product_attention(query, key, value, mask)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

		# concatenation of heads
		concat_attention = tf.reshape(scaled_attention,
									  (batch_size, -1, self.d_model))

		# final linear layer
		outputs = self.dense(concat_attention)

		return outputs

def create_padding_mask(x):
	mask = tf.cast(tf.math.equal(x, 0), tf.float32)
	# (batch_size, 1, 1, sequence length)
	return mask[:, tf.newaxis, tf.newaxis, :]
	
def create_look_ahead_mask(x):
	seq_len = tf.shape(x)[1]
	look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
	padding_mask = create_padding_mask(x)
	return tf.maximum(look_ahead_mask, padding_mask)

class PositionalEncoding(tf.keras.layers.Layer):

	def __init__(self, position, d_model):
		super(PositionalEncoding, self).__init__()
		self.pos_encoding = self.positional_encoding(position, d_model)

	def get_angles(self, position, i, d_model):
		angles = 1 / tf.pow(float(10000), (2 * (i // 2)) / tf.cast(d_model, tf.float32))
		return position * angles


	def positional_encoding(self, position, d_model):
		angle_rads = self.get_angles(
			position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
			#i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
			i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
			d_model=d_model)
		# apply sin to even index in the array
		sines = tf.math.sin(angle_rads[:, 0::2])
		# apply cos to odd index in the array
		cosines = tf.math.cos(angle_rads[:, 1::2])

		pos_encoding = tf.concat([sines, cosines], axis=-1)
		pos_encoding = pos_encoding[tf.newaxis, ...]
		return tf.cast(pos_encoding, tf.float32)

	def call(self, inputs):
		return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
		

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
	inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
	padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

	attention = MultiHeadAttention(
	  d_model, num_heads, name="attention")({
		  'query': inputs,
		  'key': inputs,
		  'value': inputs,
		  'mask': padding_mask
	  })
	attention = tf.keras.layers.Dropout(rate=dropout)(attention)
	attention = tf.keras.layers.LayerNormalization(
	  epsilon=1e-6)(inputs + attention)

	outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
	outputs = tf.keras.layers.Dense(units=d_model)(outputs)
	outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
	outputs = tf.keras.layers.LayerNormalization(
	  epsilon=1e-6)(attention + outputs)

	return tf.keras.Model(
	  inputs=[inputs, padding_mask], outputs=outputs, name=name)
	  
def encoder(vocab_size,
			num_layers,
			units,
			d_model,
			num_heads,
			dropout,
			name="encoder"):
	inputs = tf.keras.Input(shape=(None,), name="inputs")
	padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

	embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
	embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
	embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

	outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

	for i in range(num_layers):
		outputs = encoder_layer(
			units=units,
			d_model=d_model,
			num_heads=num_heads,
			dropout=dropout,
			name="encoder_layer_{}".format(i),
		)([outputs, padding_mask])

	return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
	
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
	inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
	enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
	look_ahead_mask = tf.keras.Input(
	  shape=(1, None, None), name="look_ahead_mask")
	padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

	attention1 = MultiHeadAttention(
	  d_model, num_heads, name="attention_1")(inputs={
		  'query': inputs,
		  'key': inputs,
		  'value': inputs,
		  'mask': look_ahead_mask
	  })
	attention1 = tf.keras.layers.LayerNormalization(
	  epsilon=1e-6)(attention1 + inputs)

	attention2 = MultiHeadAttention(
	  d_model, num_heads, name="attention_2")(inputs={
		  'query': attention1,
		  'key': enc_outputs,
		  'value': enc_outputs,
		  'mask': padding_mask
	  })
	attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
	attention2 = tf.keras.layers.LayerNormalization(
	  epsilon=1e-6)(attention2 + attention1)

	outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
	outputs = tf.keras.layers.Dense(units=d_model)(outputs)
	outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
	outputs = tf.keras.layers.LayerNormalization(
	  epsilon=1e-6)(outputs + attention2)

	return tf.keras.Model(
	  inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
	  outputs=outputs,
	  name=name)


def decoder(vocab_size,
			num_layers,
			units,
			d_model,
			num_heads,
			dropout,
			name='decoder'):
	inputs = tf.keras.Input(shape=(None,), name='inputs')
	enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
	look_ahead_mask = tf.keras.Input(
	  shape=(1, None, None), name='look_ahead_mask')
	padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

	embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
	embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
	embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

	outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

	for i in range(num_layers):
		outputs = decoder_layer(
			units=units,
			d_model=d_model,
			num_heads=num_heads,
			dropout=dropout,
			name='decoder_layer_{}'.format(i),
		)(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

	return tf.keras.Model(
	  inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
	  outputs=outputs,
	  name=name)
 
def seq2seq(vocab_size,
				num_layers,
				units,
				d_model,
				num_heads,
				dropout,
				name="transformer"):
	inputs = tf.keras.Input(shape=(None,), name="inputs")
	dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

	enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),name='enc_padding_mask')(inputs)
	# mask the future tokens for decoder inputs at the 1st attention block
	look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask,output_shape=(1, None, None),name='look_ahead_mask')(dec_inputs)
	# mask the encoder outputs for the 2nd attention block
	dec_padding_mask = tf.keras.layers.Lambda(
	create_padding_mask, output_shape=(1, 1, None),
	name='dec_padding_mask')(inputs)

	enc_outputs = encoder(vocab_size=vocab_size,
	num_layers=num_layers,
	units=units,
	  d_model=d_model,
	  num_heads=num_heads,
	  dropout=dropout,
	)(inputs=[inputs, enc_padding_mask])

	dec_outputs = decoder(
	  vocab_size=vocab_size,
	  num_layers=num_layers,
	  units=units,
	  d_model=d_model,
	  num_heads=num_heads,
	  dropout=dropout,
	)(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

	outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

	return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
	

def get_model(NUM_LAYERS = 2, D_MODEL = 256, NUM_HEADS = 8, UNITS = 512, DROPOUT = 0.1):
	tf.keras.backend.clear_session()

	# Hyper-parameters
	NUM_LAYERS = 2
	D_MODEL = 256
	NUM_HEADS = 8
	UNITS = 512
	DROPOUT = 0.1
	VOCAB_SIZE = 8203

	model = seq2seq(
		vocab_size=VOCAB_SIZE,
		num_layers=NUM_LAYERS,
		units=UNITS,
		d_model=D_MODEL,
		num_heads=NUM_HEADS,
		dropout=DROPOUT)
		
	model.load_weights("chatbot_models/model/weight_new.h5")
	
	return model    

# def preprocess_sentence(sentence):
#     sentence = sentence.lower().strip()
#     # creating a space between a word and the punctuation following it
#     # eg: "he is a boy." => "he is a boy ."
#     sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
#     sentence = re.sub(r'[" "]+', " ", sentence)
#     # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
#     sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
#     sentence = sentence.strip()
#     # adding a start and an end token to the sentence
#     return sentence    
	
# def evaluate(model, sentence, emotion, tokenizer, START_TOKEN, END_TOKEN):
#     MAX_LENGTH = 40
	
#     sentence = preprocess_sentence(sentence)

#     sentence = tf.expand_dims(
#       START_TOKEN + tokenizer.encode(sentence) + END_TOKEN + tokenizer.encode(emotion), axis=0)

#     output = tf.expand_dims(START_TOKEN, 0)

#     for i in range(MAX_LENGTH):
#         predictions = model(inputs=[sentence, output], training=False)

#         # select the last word from the seq_len dimension
#         predictions = predictions[:, -1:, :]
#         predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

#         # return the result if the predicted_id is equal to the end token
#         if tf.equal(predicted_id, END_TOKEN[0]):
#             break

#         # concatenated the predicted_id to the output which is given to the decoder
#         # as its input.
#         output = tf.concat([output, predicted_id], axis=-1)

#     return tf.squeeze(output, axis=0)


# def predict(model, sentence, emotion):
	
#     emotion_list = ['neutral', 'angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised']
#     emotion_list, clean_questions, clean_answers = preprocess()
#     # Build tokenizer using tfds for both questions and answers
#     tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#         emotion_list+clean_questions + clean_answers, target_vocab_size=2**13)

#     # Define start and end token to indicate the start and end of a sentence
	
#     START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
	
#     prediction = evaluate(model, sentence, emotion, tokenizer, START_TOKEN, END_TOKEN)

#     predicted_sentence = tokenizer.decode(
#       [i for i in prediction if i < tokenizer.vocab_size])

#     print('Input: {}'.format(sentence))
#     print('Output: {}'.format(predicted_sentence))

#     return predicted_sentence
	
# 	