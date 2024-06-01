import cv2
import mediapipe as mp
import numpy as np
import json
import onnxruntime
from data.post_process import Preprocessing, interpolate_or_pad
import sys
from dataset.extract_landmark import POINT_LANDMARKS
import pandas as pd
import json
sys.path.insert(0,'F:\6.Spring_24\VIetNamese_sign_language')
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
def extract_keypoints(results,POINT_LANDMARKS):
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
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 
preprocess = Preprocessing()
# Mở camera
cap = cv2.VideoCapture('./boiroi_tx_2.mp4')
inputs_model = np.zeros((124, 390))

sequence=[]
n_frame = 0
num_frame_space = 15
predicted = ''
while cap.isOpened():
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc từ camera.")
        break
    
    # Chuyển đổi frame sang màu xám để xử lý nhanh hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Dùng Mediapipe để xử lý landmarks trên khuôn mặt và bàn tay
    results = holistic.process(image=frame)
    keypoints = extract_keypoints(results,POINT_LANDMARKS)
    sequence.append(keypoints)
    sequence_arr = np.array(sequence)
    if len(sequence) > num_frame_space and np.all(sequence_arr[n_frame-num_frame_space:n_frame,40:80] == 0):
        sequence_arr = interpolate_or_pad(preprocess(sequence_arr))
        sequence_arr = sequence_arr.reshape(sequence_arr.shape[0],-1)
        sequence=[]
        n_frame = 0 
        predicted = predict(sequence_arr,sess)
        print('reset')
    elif len(sequence) == 124   :
        sequence=[]
        n_frame = 0 
        predicted = predict(sequence_arr,sess)
        print(predicted) 
    n_frame += 1  
    print(predicted)    
    cv2.putText(frame, ' '.join(predicted), (3,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # inputs_model = np.vstack((keypoint, inputs_model))
    # if inputs_model.shape[0] >= 124:
    #     inputs_model =  inputs_model[:124]
    # predicted_labels = predict(inputs_model, sess, label_map, 0.7)
    # print(predicted_labels)
    # Hiển thị frame
    cv2.imshow('MediaPipe Holistic', frame)
    
    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng các tài nguyên
cap.release()
cv2.destroyAllWindows()
