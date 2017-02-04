import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from model import normalizeImage, adjustBrightness, cropAndResize

def viewFewImages(X_train, y_train, visidx=[]):
    #print(len(X_train), len(y_train))
    plt.close('all')
    fig, axes = plt.subplots(nrows=4, ncols=4)
    for i, ax in enumerate(axes.flat, start=1):
        idx = random.randint(0, len(X_train)-1)
        if (len(visidx)>0): 
            idx = visidx[(i-1)]
        img = X_train[idx].squeeze()
        ax.imshow(img)
        #print(y_train[idx], '=', sign_names[class_ids.index(str(y_train[idx]))])
        ax.set_xlabel(y_train[idx])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    fig.tight_layout()
    plt.show()

CSV_FILE_NAME = 'driving_log.csv'
df_a = pd.read_csv(CSV_FILE_NAME, header=0, usecols=[0, 1, 2, 3], names=['CenterImg', 'LeftImg', 'RightImg', 'SteerAngle'])
print('df_a record count', df_a.shape[0])
train_rows_count = int(df_a.shape[0]*0.8)
df_a = df_a.sample(frac=1).reset_index(drop=True)
df_t = df_a.loc[0:train_rows_count-1]
df_v = df_a.loc[train_rows_count:]
df_v = df_v.reset_index(drop=True)
print('df_t record count', df_t.shape[0])
print('df_v record count', df_v.shape[0])
# df_a = None


X_batch = []
y_batch = []

for index, row in df_a.iterrows():
	y_batch.append(row['SteerAngle'])

y = np.asarray(y_batch)
plt.hist(y, 25)
plt.title('Histogram of original steering angles')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# for index, row in df_t.loc[0:200].iterrows():
# 	options = ['CenterImg', 'LeftImg', 'RightImg']
# 	imgToRead = random.choice(options)
# 	image = ndimage.imread(row[imgToRead].strip())
# 	image = cropAndResize(image)
# 	image = adjustBrightness(image)
# 	image = normalizeImage(image)
# 	X_batch.append(image)
# 	y_batch.append(row['SteerAngle'])

# viewFewImages(X_batch, y_batch)