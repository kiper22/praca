import cv2 as cv
import os
import numpy as np
from keras_facenet import FaceNet
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Face:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160) # taki jest rozmiar zdjęć wejściowych, wykorzystywane zdjęcia już są przycięte do pożądanej wielkości
        self.X, self.Y = [], []
        
    def extract_face(self, filename):
        img = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
        return cv.resize(img[0:img.shape[0], 0:img.shape[1]], self.target_size)
    
    def load_faces(self, dir):
        return [self.extract_face(os.path.join(dir, im_name)) for im_name in os.listdir(dir)]

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)
            labels = [sub_dir] * len(FACES)
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

faces = Face(directory=r'C:\Projekty\PracaInz\Programs\images')
X, Y = faces.load_classes()

embedder = FaceNet()

def get_embedding(face_image):
    return embedder.embeddings(np.expand_dims(face_image.astype('float32'), axis=0))[0]

EMBEDDED_X = np.array([get_embedding(img) for img in X])

np.savez_compressed('faces_embeddings_full_4.npz', EMBEDDED_X, Y)

encoder = LabelEncoder().fit(Y)
Y = encoder.transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)

model = SVC(kernel='linear', probability=True).fit(X_train, Y_train)

with open('svm_model_160x160_4.pkl', 'wb') as f:
    pickle.dump(model, f)
