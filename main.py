from keras_facenet import FaceNet
embedder = FaceNet()
image = './got-mat-trai-xoan-2.jpg'
detections = embedder.extract(image, threshold=0.95)
print(detections)
# embeddings = embedder.embeddings(image)