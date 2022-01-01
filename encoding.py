#  Get ClassName and Encoding Vector of Training_images

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList




encodeListKnown = findEncodings(images)
print(encodeListKnown)
np.savez_compressed('encoding.npz', classNames, encodeListKnown)


