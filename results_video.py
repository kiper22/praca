# Copyright 2023 Kacper Wieleba
# Licensed under the MIT License.
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import datetime
from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CameraApp:
    def __init__(self, window, window_title):
        self.yolo_model = YOLO(r'C:\Projekty\PracaInz\Programs\runs\detect\train3\weights\best.pt')
        self.window = window
        self.window_title = window_title
        
        self.fps_times = [datetime.datetime.now()] * 10
        self.fps_index = 0
        self.frame_count = 0
        
        self.window.title(window_title)
        self.create_menu()
        self.start_video()
        self.facenet = FaceNet()
        self.faces_embeddings = np.load(r'C:\Projekty\PracaInz\Programs\FaceNet_part\faces_embeddings_done_4classes_1.npz')
        self.Y = self.faces_embeddings['arr_1']
        self.encoder = LabelEncoder().fit(self.Y)
        self.yolo_model = YOLO(r'C:\Projekty\PracaInz\Programs\runs\detect\train3\weights\best.pt')
        self.model = pickle.load(open(r'C:\Projekty\PracaInz\Programs\FaceNet_part\svm_model_160x160_1.pkl', 'rb'))

        self.window.bind("<Configure>", self.on_resize)

        self.delay = 5
        self.update()
        self.window.mainloop()
    
    def calculate_fps(self, time_frame_1):
        # Oblicz różnicę czasu od poprzedniej klatki
        time_diff = (time_frame_1 - self.fps_times[self.fps_index]).total_seconds()
        fps = 0

        if time_diff > 0:
            fps = 1 / time_diff

        # Zaktualizuj tablicę z czasami
        self.fps_times[self.fps_index] = time_frame_1
        self.fps_index = (self.fps_index + 1) % 10

        return fps
    
    def start_video(self):
        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()
    
    def create_menu(self):
        menubar = tk.Menu(self.window)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.window.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_checkbutton(label="Turn On/Off", variable=tk.BooleanVar(value=True), command=self.toggle_stream)
        self.window.config(menu=menubar)
    
    def update(self):
        ret, frame = self.vid.read()

        if ret:
            self.frame_count += 1
                
            frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Użyj modelu YOLO do wykrycia twarzy
            yolo_results = self.yolo_model.predict(frame)
            frame_with_faces = self.draw_faces(frame, yolo_results)

            img = Image.fromarray(frame_with_faces)
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.canvas.update()
            
            time_frame_1 = datetime.datetime.now()
            # Oblicz różnicę czasu od poprzedniej klatki
            fps = self.calculate_fps(time_frame_1)
            new_title = f"{self.window_title} - Frames: {self.frame_count} - FPS: {fps:.2f}"
            self.window.title(new_title)
                
        self.window.after(self.delay, self.update)

    def draw_faces(self, frame, yolo_results):
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                tlx = int(box.xyxy.tolist()[0][0])
                tly = int(box.xyxy.tolist()[0][1])
                brx = int(box.xyxy.tolist()[0][2])
                bry = int(box.xyxy.tolist()[0][3])

                # Crop and resize the face
                img = rgb_img[tly:bry, tlx:brx]
                img = cv.resize(img, (160, 160))
                img = np.expand_dims(img, axis=0)

                # Get embeddings from FaceNet
                ypred = self.facenet.embeddings(img)

                # Predict using the SVM model
                face_name = self.model.predict(ypred)
                final_name = self.encoder.inverse_transform(face_name)[0]

                # Probability calculation
                yhat_prob = self.model.predict_proba(ypred)
                class_index = face_name[0]
                class_probability = yhat_prob[0, class_index] * 100
                predict_names = self.encoder.inverse_transform(face_name)
                
                final_name = predict_names[0]
                cv.rectangle(frame, (tlx, tly), (brx, bry), (255, 0, 255), 10)
                cv.putText(frame, f"{final_name} ({class_probability:.2f}%)", (tlx, tly - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)
        return frame
    
    def toggle_stream(self):            
        self.update()
    
    def on_resize(self, event):
        self.width = event.widget.winfo_width()
        self.height = event.widget.winfo_height()

window = tk.Tk()
app = CameraApp(window, "Tkinter Camera")