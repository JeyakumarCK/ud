import pickle
import csv

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_test))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Visualizations will be shown in the notebook.
# %matplotlib inline
import csv

class_dict = {}
class_ids = []
sign_names = []
with open('signnames.csv', 'rt') as f:
    csvreader = csv.DictReader(f)
    for row in csvreader:
        class_ids.append(row['ClassId'])
        sign_names.append(row['SignName'])

class_dict['ClassId'] = class_ids
class_dict['SignName'] = sign_names

# index = random.randint(0, len(X_train))
# image = X_train[index].squeeze()
# print('label = ', y_train[index])
# print(class_ids.index(str(y_train[index])), ' = ', sign_names[class_ids.index(str(y_train[index]))])
# plt.figure(figsize=(1,1))
# plt.imshow(image, cmap="color")
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

n_train = len(X_train)
n_validation = len(X_validation)
n_test = len(X_test)

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Total Number of test examples =", n_test)

'''
gs1 = gridspec.GridSpec(5, 10)
gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
fig = plt.figure(figsize=(12,12))

for i in range(50):
    idx = random.randint(0, len(X_train))

    ax1 = plt.subplot(gs1[i])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    img = X_train[idx].squeeze()

    plt.subplot(5,10,i+1)
    plt.imshow(img)
    plt.axis('off')

plt.show()
'''

fig, axes = plt.subplots(nrows=5, ncols=8)

for i, ax in enumerate(axes.flat, start=1):
    idx = random.randint(0, len(X_train))
    img = X_train[idx].squeeze()
    print(y_train[idx], '=', sign_names[class_ids.index(str(y_train[idx]))])
    #ax.set_title('Test Axes {}'.format(i))
    ax.set_xlabel(y_train[idx])
    #ax.set_ylabel('Y axis')
    ax.imshow(img)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    #ax.axis('off')

fig.tight_layout()

plt.show()




