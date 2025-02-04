import os
import shutil
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img
from keras.models import Model
from sklearn.cluster import KMeans

# Load VGG16 model and remove the last layer
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Function to extract features using VGG16 model
def extract_features(file, model):
    img = load_img(file, target_size=(224,224))
    img = np.array(img)
    reshaped_img = img.reshape(1,224,224,3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=True)
    return features

# Path where images are stored
path = r"file path"
os.chdir(path)

# Create a list to hold image names
list1 = []
with os.scandir(path) as files:
    for file in files:
        if file.name.endswith('.jpg'):
            list1.append(file.name)

# Feature extraction for each image
data = {}
for xx in list1:
    feat = extract_features(xx, model)
    data[xx] = feat

# Get filenames and features
filenames = np.array(list(data.keys()))
feat = np.array(list(data.values())).reshape(-1, 4096)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=15, random_state=22)
kmeans.fit(feat)

# Organize images into clusters
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
    groups[cluster].append(file)

# Move images to respective folders based on clusters
for n in range(15):
    os.mkdir("H:\\final\\c%d" % n)
    l = len(groups[n])
    for i in range(l):
        shutil.move(groups[n][i], "H:\\final\\c%d" % n)