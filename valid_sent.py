#Test IMDB model on sentences.

import pandas as pd
import numpy as np

dataset=pd.read_csv('valid_data_sent.csv')
text_data=dataset['Text'].tolist()
classes=dataset['classess'].tolist()
print dataset.head()
print type(dataset)

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

vocab_size=20000
encoded_docs1 = [one_hot(d, vocab_size) for d in text_data]
#print(encoded_docs1)

# pad documents to a max length of 4 words
max_length = 400
padded_docs1 = pad_sequences(encoded_docs1, maxlen=max_length, padding='post')

from keras.models import model_from_json

from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
#print("Loaded model from disk")


submit=loaded_model.predict(padded_docs1)
print "The probabilities are:\n",submit

predictions=submit.argmax(axis=-1)

print "\n"
print "class_1_probability:\n",submit[:,1]

print "\n"
for i in submit[:,1]:
	i=i*10
	if i>7.6:
		print "Rating for this example is: Best"
	elif i<=7.6 and i>6.5:
		print "Rating for this example is: Better"
	elif i<=6.5 and i>=5.2:
		print "Rating for this example is: Good"
	elif i<5.0 and i>4.5:
		print "Rating for this example is: Negative"
	elif i<=4.5 and i>3.5:
		print "Rating for this example is: Bad"
	elif i<=3.5:
		print "Rating for this example is: worst"
	else:
		print "Coin_toss.........."

print "\n"
print "Move_to_precision_and_recall........................."
from sklearn.metrics import accuracy_score,classification_report
print "The accuracy_score is:",accuracy_score(classes,predictions)*100
print classification_report(classes,predictions)

print "The actual_class labels are:",classes
print "The predicted_class labels are:",predictions


