from crop_face import load_dataset
import numpy as np # linear algebra
from keras.models import load_model
data = np.load('image.npz')
trainX, trainY = data['arr_0'], data['arr_1']
# load the facenet model
model = load_model('./facenet-keras/facenet_keras.h5')
print('Loaded Model')