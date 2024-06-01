import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QGridLayout, QFrame
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer
import cv2
import mediapipe as mp
import qdarkstyle

from copy import copy
import sys
import numpy as np
import onnxruntime
from data.post_process import Preprocessing, interpolate_or_pad
import sys
sys.path.insert(0, 'F:\6.Spring_24\VIetNamese_sign_language')
from dataset.extract_landmark import POINT_LANDMARKS
import pandas as pd
import json
preprocessLayer = Preprocessing()
sess = onnxruntime.InferenceSession("output/model.onnx")
with open('dataset/sign_to_prediction_index_map.json', 'r', encoding='utf-8') as json_file:
    label_map = json.load(json_file)
def predict(input_data, session, label_map=label_map,threshold=0.5):
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    # Chạy mô hình trên dữ liệu đầu vào
    output = session.run(None, {'input': input_data})[0][0]
    predictions = np.argmax(output, axis=0)
    probabilities = (np.exp(output) / np.sum(np.exp(output), axis=0))
    confidence = probabilities[np.argmax(probabilities, axis=0)]
    predicted_labels = list(label_map.keys())[predictions]
    if confidence < threshold :
        return "Uncertain"
    else:
        return predicted_labels


preprocess = Preprocessing()

class ConvertFileToParquet():
    def __init__(self, data, folder_path, save_path=None):
        self.folder_path = folder_path
        self.save_path = save_path
        self.data = data

    @staticmethod
    def column_name():
        coordinates = ['x', 'y', 'z']
        col_name = ['_face_', '_left_hand_', '_pose_', '_right_hand_']
        column_name_list = []
        for coordinate in coordinates:
            for name in col_name:
                if name == '_face_':
                    for i in range(0, 468):
                        column_name_list.append(coordinate+name+str(i))
                elif name == '_left_hand_':
                    for i in range(0, 21):
                        column_name_list.append(coordinate+name+str(i))
                elif name == '_pose_':
                    for i in range(0, 33):
                        column_name_list.append(coordinate+name+str(i))
                elif name == '_right_hand_':
                    for y in range(0, 21):
                        column_name_list.append(coordinate+name+str(y))
        return column_name_list

    def convert_to_dataframe(self):
        data_list = []
        frame = []
        for i in range(self.data.shape[0]):
            frame.append(i)
            data_list.append(self.data[i])
        data_df = pd.DataFrame(data=data_list, columns=ConvertFileToParquet.column_name())
        data_df.insert(0, 'frame', frame)
        return data_df

class TrackingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand and Face Landmark Tracking App")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        button_layout1 = QVBoxLayout()
        self.exit_button = QPushButton("Exit Application", self)
        self.exit_button.clicked.connect(self.close)
        self.exit_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.exit_button)

        self.run_webcam_button = QPushButton("Run Webcam", self)
        self.run_webcam_button.clicked.connect(self.run_webcam)
        self.run_webcam_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.run_webcam_button)

        self.import_video_button = QPushButton("Import Video", self)
        self.import_video_button.clicked.connect(self.import_video)
        self.import_video_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.import_video_button)
        
        self.stop_camera_button = QPushButton("Stop Webcam", self)
        self.stop_camera_button.clicked.connect(self.stop_webcam)
        self.stop_camera_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.stop_camera_button)

        self.export_csv_button = QPushButton("Export CSV", self)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_csv_button.setFixedSize(200, 50)  # Set fixed size for the button
        button_layout1.addWidget(self.export_csv_button)
        
        self.layout = QGridLayout()
        self.layout.addLayout(button_layout1, 0, 0)

        # Add a line between the buttons and the video label
        line = QFrame(self)
        line.setFrameShape(QFrame.VLine)
        self.layout.addWidget(line, 0, 1)

        self.layout.addWidget(self.video_label, 0, 2)

        widget = QWidget(self)
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.camera = None
        self.mp_hands = mp.solutions.hands.Hands()
        self.mp_face = mp.solutions.face_mesh.FaceMesh()
        self.mp_pose = mp.solutions.pose.Pose()
        self.mp_holistic = mp.solutions.holistic.Holistic()


        self.hands_results = None
        self.face_results = None
        self.pose_results = None
        
        self.landmark_dataframe = pd.DataFrame()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.landmark_dataframe = pd.DataFrame(columns=["sequence_id", "frame"])

        self.res = []
        self.is_video_finished = False
        self.threshold = 40
        self.num_frame_space = 10
        self.list_frame = []
        self.predicted = None

    def run_webcam(self):
        if self.camera is not None:
            self.camera.release()  # Release the camera
        self.camera = cv2.VideoCapture(0)  # Open the default camera
        self.is_video_finished = False
        self.res = []
        self.timer.start(30)  # Update frame every 30 milliseconds
        
    def import_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Select Video File")
        if video_path:
            if self.camera is not None:
                self.camera.release()
            self.camera = cv2.VideoCapture(video_path)
            self.is_video_finished = False
            self.res = []
            self.timer.start(30)  # Update frame every 30 milliseconds
            

    def stop_webcam(self):
        if self.camera is not None:
            self.camera.release()  # Release the camera
            self.camera = None  # Set the camera to None
            self.video_label.clear()  # Clear the video label
            self.ret = False

    def update_frame(self):
        if self.is_video_finished:
            self.stop_webcam()
            self.ret = False
            return self.landmark_dataframe
        else:   
            keypoints = None
            if self.camera is not None:
                self.ret, frame = self.camera.read()
            else:
                self.ret = False
                
        if self.ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_holistic.process(frame)
            
            
            keypoints = self.extract_keypoints(results,POINT_LANDMARKS)
            # inputs_model = np.vstack((keypoint, inputs_model))
            # if inputs_model.shape[0] >= 124:
            #     inputs_model =  inputs_model[:124]
            # predicted_labels = predict(inputs_model, sess, label_map, 0.7)
            # print(predicted_labels)
            
            
            
            self.res.append(keypoints)
            landmarks_series = pd.Series(keypoints)
            self.landmark_dataframe = self.landmark_dataframe._append(landmarks_series, ignore_index=True)
            
            nres = len(self.res)
            sequence_arr = np.array(self.res)
            n_frame = sequence_arr.shape[0]
            self.landmark_dataframe = self.landmark_dataframe._append({"sequence_id": "sửa sau","frame": n_frame}, ignore_index=True)
            ranges = [(42, 80)]
            slices_arr = np.concatenate([sequence_arr[n_frame-self.num_frame_space:n_frame, start:end] for start, end in ranges], axis=1)
            print('nres:', slices_arr)
            
            print('nres:', nres)
            if nres > self.num_frame_space and (slices_arr == 0).all():
                self.res = []                                                                          
                print('reset')

            if nres==self.threshold:
                self.threshold += 1
                subarray=sequence_arr[nres-124:nres,:]

                subarray = preprocessLayer(subarray)
                subarray = interpolate_or_pad(preprocess(subarray))
                subarray = subarray.reshape(subarray.shape[0],-1)
                self.predicted = predict(subarray,sess)
                print(self.predicted)
            
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)
            self.predicted_label = QLabel(str(self.predicted))
            self.layout.addWidget(self.predicted_label, 1, 2)  # Add the label to position (1,2)

            self.setLayout(self.layout)
            
        elif self.ret == False:
            self.res = np.array(self.res)
            self.res_1 = preprocessLayer(self.res)
            self.res_1 = interpolate_or_pad(preprocess(self.res_1))
            self.res_1 = self.res_1.reshape(self.res_1.shape[0],-1)
            self.predicted = predict(self.res_1, sess)
            print(f'final predict: {self.predicted}')
            self.is_video_finished = True
            
            self.predicted_label = QLabel(str(self.predicted))
            self.layout.addWidget(self.predicted_label, 1, 2)  # Add the label to position (1,2)

            self.setLayout(self.layout)
            
            convert = ConvertFileToParquet(self.res, None ,None)
            self.landmark_dataframe = convert.convert_to_dataframe()
            
            return self.landmark_dataframe

    def extract_keypoints(self, results,POINT_LANDMARKS):
        face_x  = np.array([res.x for res in results.face_landmarks.landmark],dtype=np.float32) if results.face_landmarks else np.zeros(468)
        face_y  = np.array([res.y for res in results.face_landmarks.landmark],dtype=np.float32) if results.face_landmarks else np.zeros(468)
        face_z  = np.array([res.z for res in results.face_landmarks.landmark],dtype=np.float32) if results.face_landmarks else np.zeros(468)
        
        lh_x  = np.array([res.x for res in results.left_hand_landmarks.landmark],dtype=np.float32) if results.left_hand_landmarks else np.zeros(21)
        lh_y  = np.array([res.y for res in results.left_hand_landmarks.landmark],dtype=np.float32) if results.left_hand_landmarks else np.zeros(21)
        lh_z  = np.array([res.z for res in results.left_hand_landmarks.landmark],dtype=np.float32) if results.left_hand_landmarks else np.zeros(21)
        
        pose_x  = np.array([res.x for res in results.pose_landmarks.landmark],dtype=np.float32) if results.pose_landmarks else np.zeros(33)
        pose_y  = np.array([res.y for res in results.pose_landmarks.landmark],dtype=np.float32) if results.pose_landmarks else np.zeros(33)
        pose_z  = np.array([res.z for res in results.pose_landmarks.landmark],dtype=np.float32) if results.pose_landmarks else np.zeros(33)
        
        
        rh_x  = np.array([res.x for res in results.right_hand_landmarks.landmark],dtype=np.float32) if results.right_hand_landmarks else np.zeros(21)
        rh_y  = np.array([res.y for res in results.right_hand_landmarks.landmark],dtype=np.float32) if results.right_hand_landmarks else np.zeros(21)
        rh_z  = np.array([res.z for res in results.right_hand_landmarks.landmark],dtype=np.float32) if results.right_hand_landmarks else np.zeros(21)

        x_cor = np.concatenate([face_x, lh_x, pose_x, rh_x])
        y_cor = np.concatenate([face_y, lh_y, pose_y, rh_y])
        z_cor = np.concatenate([face_z, lh_z, pose_z, rh_z])
        POINT_LANDMARKS_array = np.array(POINT_LANDMARKS)
        result = np.concatenate((x_cor[POINT_LANDMARKS_array], y_cor[POINT_LANDMARKS_array], z_cor[POINT_LANDMARKS_array]))
        return   result

    def closeEvent(self, event):
        if self.camera is not None:
            self.camera.release()
        event.accept()

    def get_frame_number(self):
        if self.camera is None:
            return 0
        else:
            return self.camera.get(cv2.CAP_PROP_POS_FRAMES)

    def export_csv(self):
        file_dialog = QFileDialog()
        csv_path, _ = file_dialog.getSaveFileName(self, "Export CSV File")
        if csv_path:
            self.landmark_dataframe.to_csv(csv_path, index=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())  # Apply the dark theme
    window = TrackingApp()
    window.show()
    sys.exit(app.exec())