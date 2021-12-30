from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from keras_facenet import FaceNet
import numpy as np # linear algebra

from crop_face import extract_face

# normalize input vectors
data = np.load('image.npz')
emdTrainX, trainY = data['arr_0'], data['arr_1']
in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)
print(emdTrainX_norm)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainy_enc = out_encoder.transform(trainY)
print(trainy_enc)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(emdTrainX_norm, trainy_enc)

# predict
yhat_train = model.predict(emdTrainX_norm)
print(yhat_train)


# test
testX ='./image_test/sontung.jpg'
embedd = FaceNet()
detections = embedd.extract(testX, threshold=0.95)

emdTestX = detections[0]['embedding']
# emdTestX_norm = in_encoder.transform(emdTestX)

samples = np.expand_dims(emdTestX, axis=0)
yhat_class = model.predict(samples)
print(yhat_class)
