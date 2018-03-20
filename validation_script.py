#====================validation_dataset===========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Laprint "Saving_model................"
from keras.models import model_from_json
# serialize model to Jencoded_docs = [one_hot(d, vocab_size) for d in X_train]
#print(encoded_docs)SON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")belEncoder
encoder=LabelEncoder()
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

dataset=pd.read_csv('validation_dataset.csv')
text_data=dataset['text'].tolist()
label=dataset['label'].tolist()

encoder.fit(label)
y_test=encoder.fit_transform(label)
temp=y_test
from keras.utils.np_utils import to_categorical
y_test=to_categorical(y_test)

vocab_size=20000
encoded_docs1 = [one_hot(d, vocab_size) for d in text_data]
#print(encoded_docs1)

# pad documents to a max length of 4 words
max_length = 400
padded_docs1 = pad_sequences(encoded_docs1, maxlen=max_length, padding='post')

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
predictions=submit.argmax(axis=-1)

from sklearn.metrics import accuracy_score,classification_report

print "The accuracy_score is:",accuracy_score(temp,predictions)*100
print classification_report(temp,predictions)


#================================manuel_testing===============================


text1=""" An extremely dark and brooding show with an excellent cast. One of the few shows that I try to watch on a regular basis. Glad to see Bebe Neuwirth in a recurring role, but feel Andre Braugher is underutilized. He is one intense actor! Hope CBS gives it a better time slot next season."""

text_set=[]
text_set.append(text1)
vocab_size=20000
encoded_docs1 = [one_hot(d, vocab_size) for d in text_set]
#print(encoded_docs1)

# pad documents to a max length of 4 words
max_length = 400
padded_docs1 = pad_sequences(encoded_docs1, maxlen=max_length, padding='post')

submit=loaded_model.predict(padded_docs1)
predictions=submit.argmax(axis=-1)


print "True class label is: Pos"
print "\n"
print text1

if predictions==0:
	print "The predicted_class label is:Neg"
else:
	print "The predicted_class label is:Pos"



