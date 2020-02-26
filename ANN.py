# What type of framework? -- Keras

import preprocess
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("done")


if __name__ == '__main__':
	dir_path = 'data_orig'
	emotions = {'01':'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
	
	x_train, x_test, y_train, y_test = preprocess.readData(emotions, dir_path, test_size=0.2)
	
	print((x_train.shape[0], x_test.shape[0]))
	print('Features extracted: ', x_train.shape[1])
	
	#defining a Multilayer perceptron classifier
	model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
	
	# Train the model
	model.fit(x_train,y_train)
	print("done training")
	
	# Predict on the test dataset
	y_pred=model.predict(x_test)
	
	
	accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
	
	print("Accuracy: {:.2f}%".format(accuracy*100))