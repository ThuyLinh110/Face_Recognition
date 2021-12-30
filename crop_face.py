import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv
import os
from keras_facenet import FaceNet
from matplotlib import pyplot as plt


# crop face from image and get embedding vector
def extract_face(filename, required_size=(160, 160)):
    embedd = FaceNet()
    detections = embedd.extract(filename, threshold=0.95)
    return detections[0]['embedding']


def load_face(dir):
    faces = list()
    # enumerate files
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(dir):
    # list for faces and labels
    X, y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" % (len(faces),subdir) ) # print progress
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

# load train dataset
trainX, trainY = load_dataset('./image/')
np.savez_compressed('image.npz', trainX, trainY)
