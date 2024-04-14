import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Load data from pickle file
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Calculate the maximum length of arrays in data_dict['data']
max_length = max(len(d) for d in data_dict['data'])

# Pad each array in data_dict['data'] to the maximum length
data_padded = [np.pad(d, (0, max_length - len(d)), mode='constant')
               for d in data_dict['data']]

# Convert the padded data to a numpy array and flatten it
data = np.array([np.ravel(np.asarray(d)) for d in data_padded])
labels = np.array(data_dict['labels'])
# labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.35, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

# Compute validation accuracy
y_val_predict = model.predict(x_test)
val_score = accuracy_score(y_val_predict, y_test)
print("Validation accuracy:", val_score*100)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
