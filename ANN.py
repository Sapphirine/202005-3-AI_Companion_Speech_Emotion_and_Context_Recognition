# What type of framework? -- Keras

import preprocess
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

#print("done")

import itertools

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
	
if __name__ == '__main__':
	dir_path = 'data_orig'
	emotions = {'01':'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
	emotion_arr = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
	
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
	
	print(y_pred)
	
	accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
	
	print("Accuracy: {:.2f}%".format(accuracy*100))
	
	test_cm = confusion_matrix(y_test, y_pred)
	test_cm = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]
	
	plot_confusion_matrix(cm=test_cm,target_names=emotion_arr, normalize=True, title = "ANN Confusion Matrix")