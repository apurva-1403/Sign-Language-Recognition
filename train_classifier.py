import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.35, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)


# Predict labels for test data
y_predict = model.predict(x_test)

# Compute accuracy
score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly!'.format(score * 100))


# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
