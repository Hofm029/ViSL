import os
import numpy as np
import torch
import cv2
import mediapipe as mp
import sys  
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from dataset.preprocess import PreprocessLayer
import json
from sklearn.model_selection import train_test_split

sys.path.insert(0, './')
path_processed = './data/processed_data/'

if not os.path.exists(path_processed):
    os.mkdir(path_processed)
    
DATA_PATH = "./dataset/landmarks/" 
path = "./dataset/videos/"
actions = os.listdir(path)
# For webcam input:
# def __main__():
def extract_landmark_from_video():
    fieldnames = []
    for axis in ['x', 'y', 'z']:
        for i in range(468):
            fieldnames.append(f'{axis}_face_{i}')
        for i in range(21):
            fieldnames.append(f'{axis}_left_{i}')
        for i in range(33):
            fieldnames.append(f'{axis}_pose_{i}')
        for i in range(21):
            fieldnames.append(f'{axis}_right_{i}')
    df = pd.DataFrame(columns=fieldnames)
    
    
    mp_holistic = mp.solutions.holistic
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    action = actions[0]
    for action in tqdm(actions):
        action_path = os.path.join(path, action)
        if not os.path.exists(action_path):
            os.mkdir(action_path)

        video_list = os.listdir(os.path.join(path, action))
        for video_name in video_list:
            df_new = df.copy()
            video_path = os.path.join(action_path, video_name)
            cap = cv2.VideoCapture(video_path)
            sequence =  []
            with mp_holistic.Holistic() as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_height, frame_width, _ = frame.shape
                    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
                    frame.flags.writeable = False
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints) 
                    sequence_arr = np.array(sequence)
                    # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
                    # Thoát khi bấm phím 'q' (ASCII value của 'q' là 113)
                    if cv2.waitKey(5) & 0xFF == 113:
                        break
                # Giải phóng tài nguyên
                folder_path = DATA_PATH +  action
                if not os.path.exists(DATA_PATH +  action):
                    os.makedirs(folder_path)
                file_path = os.path.join(folder_path, video_name.split('.mp4')[0])
                np.save(file_path, sequence_arr)
                print('\n' + video_name.split('.mp4')[0])
                cap.release()
    print("\n\t\t\t---------------------END to read all Video---------------------\t\t\t\n")
    cv2.destroyAllWindows()

label_map = {label:num for num,label in enumerate(actions)}

def extract_keypoints(results):
    if results.face_landmarks:
        face_x = np.array([res.x for res in results.face_landmarks.landmark]).flatten()   
        face_y = np.array([res.y for res in results.face_landmarks.landmark]).flatten()
        face_z = np.array([res.z for res in results.face_landmarks.landmark]).flatten()
    else: 
        face_x = np.full(468, np.nan)
        face_y = np.full(468, np.nan)
        face_z = np.full(468, np.nan)

    if results.left_hand_landmarks:
        lh_x = np.array([res.x for res in results.left_hand_landmarks.landmark]).flatten()   
        lh_y = np.array([res.y for res in results.left_hand_landmarks.landmark]).flatten()
        lh_z = np.array([res.z for res in results.left_hand_landmarks.landmark]).flatten()
    else: 
        lh_x = np.full(21, np.nan)
        lh_y = np.full(21, np.nan)
        lh_z = np.full(21, np.nan)

    if results.pose_landmarks:
        pose_x = np.array([res.x for res in results.pose_landmarks.landmark]).flatten()   
        pose_y = np.array([res.y for res in results.pose_landmarks.landmark]).flatten()
        pose_z = np.array([res.z for res in results.pose_landmarks.landmark]).flatten()
    else: 
        pose_x = np.full(33, np.nan)
        pose_y = np.full(33, np.nan)
        pose_z = np.full(33, np.nan)
        
    if results.right_hand_landmarks:
        rh_x = np.array([res.x for res in results.right_hand_landmarks.landmark]).flatten()   
        rh_y = np.array([res.y for res in results.right_hand_landmarks.landmark]).flatten()
        rh_z = np.array([res.z for res in results.right_hand_landmarks.landmark]).flatten()
    else: 
        rh_x = np.full(21, np.nan)
        rh_y = np.full(21, np.nan)
        rh_z = np.full(21, np.nan)
    x_cor = np.concatenate([face_x, lh_x, pose_x, rh_x])
    y_cor = np.concatenate([face_y, lh_y, pose_y, rh_y])
    z_cor = np.concatenate([face_z, lh_z, pose_z, rh_z])
    return   np.concatenate([x_cor, y_cor, z_cor])


def load_data(actions):
    x,labels = [],[]
    for action in actions:
        action_path = os.path.join(DATA_PATH+action)
        file_name = os.listdir(action_path)
        for file in file_name:
            print(file)
            res = np.load(os.path.join(DATA_PATH,action + '/' + "{}".format(file)))
            labels.append(label_map[action])
            x.append(res)
    return x, labels

class Convert_file_to_parquet():
    def __init__(self, folder_path, save_path=None):
        self.folder_path = folder_path
        self.save_path = save_path

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
                    for i in range(0, 21):
                        column_name_list.append(coordinate+name+str(i))
        return column_name_list

    @staticmethod
    def convert_to_dataframe(data, file_name):
        data_list = []
        frame = []
        for i in range(data.shape[0]):
            frame.append(i)
            data_list.append(data[i])
        data_df = pd.DataFrame(data=data_list, columns=Convert_file_to_parquet.column_name())
        data_df.insert(0, 'frame', frame)
        data_df.index = [file_name]*len(data_df)
        data_df.index.name = 'sequence_id'
        return data_df

    def convert_to_parquet(self):
        dataframe_list = []
        for root, dirs, files in os.walk(self.folder_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                data = np.load(file_path)
                filename = filename[:len(filename)-4]
                dataframe = Convert_file_to_parquet.convert_to_dataframe(data, filename)
                dataframe_list.append(dataframe)
        dataframe = pd.concat(dataframe_list)
        if self.save_path != None:
            dataframe.to_parquet(self.save_path)
        else:
            dataframe.to_parquet('./dataset/data_csv.parquet')

def train_csv(landmarks_folder,out_folder,parquet_name):
    train_landmark_files = './dataset/train_landmark_files'
    if not os.path.exists(train_landmark_files):
        os.mkdir(train_landmark_files)
    # Tạo một DataFrame rỗng với 3 cột
    df = pd.DataFrame(columns=['path', 'file_id','sequence_id', 'phrase'])
    # Duyệt qua tất cả các thư mục con trong thư mục gốc
    for folder_name in os.listdir(landmarks_folder):
        # Đường dẫn đầy đủ đến thư mục con
        folder_path = os.path.join(landmarks_folder + '/' + folder_name)
        # Kiểm tra xem đây có phải là một thư mục không
        if os.path.isdir(folder_path):
            # Duyệt qua tất cả các tệp .npy trong thư mục con
            for filename in glob.glob(folder_path + '/*.npy'):
                # Tên tệp không có phần mở rộng
                sequence_id = os.path.splitext(os.path.basename(filename))[0]
                # Cập nhật DataFrame
                new_row = pd.DataFrame({'path': [train_landmark_files+'/'+parquet_name],'file_id':parquet_name[:-8], 'sequence_id': [sequence_id], 'phrase': [folder_name]})
                df = pd.concat([df, new_row], ignore_index=True) 
    df.to_csv(out_folder + '/df.csv',index=False)
if __name__ == '__main__':
    # 1. Read video and extract landmark to numpy file
    # extract_landmark_from_video()
    
    ##2. Convert numpy file to parquet file
    # print("\t\t-----CONVERT TO PARQUET FILE-----\t\t")
    # folder_path = './dataset/landmarks'
    # save_path = './dataset/train_landmark_files/VSL9.parquet'
    # convert = Convert_file_to_parquet(folder_path, save_path)
    # dataframe = convert.convert_to_parquet()
    
    ##Extra to CSV Train file
    print("\t\t\t-----Extract to train.csv-----\t\t\t")
    landmarks_folder = './dataset/landmarks'
    out_folder = './dataset'
    parquet_name = 'VSL9.parquet'
    train_csv(landmarks_folder,out_folder,parquet_name)

    data_fr = pd.read_csv('dataset/df.csv')

    label_mapping = {phrase: i for i, phrase in enumerate(data_fr['phrase'].unique())}
    with open('dataset/sign_to_prediction_index_map.json', 'w', encoding='utf-8') as json_file:
        json.dump(label_mapping, json_file,ensure_ascii=False)
    with open('dataset/sign_to_prediction_index_map.json', 'r', encoding='utf-8') as json_file:
        label_mapping = json.load(json_file)
        
    # Tạo cột label mới trong data_fr dựa trên cột phrase và mapping từ file JSON
    data_fr['label'] = data_fr['phrase'].map(label_mapping)
    train_df, test_df = train_test_split(data_fr, test_size=0.15, random_state=1)

    # In thông tin về số lượng mẫu trong tập train và test
    print("Number of class in train:", len(train_df['phrase'].unique().tolist()))
    print("Number of class in test:", len(test_df['phrase'].unique().tolist()))

    # Lưu train_df thành tệp CSV
    train_df.to_csv('dataset/train.csv', index=False)
    # Lưu test_df thành tệp CSV
    test_df.to_csv('dataset/test.csv', index=False)
    
    #  # 3. Load data.npy and labels.npy to preprocess data
    # # # 3. Load data.npy and labels.npy to preprocess data
    # print("\t\t\t-----PreprocessLayer-----\t\t\t")
    # preprocess = PreprocessLayer()
    # x_raws, labels_np = load_data(actions)
    # processed_data  = [preprocess(x_raw).numpy() for x_raw in x_raws]
    
    # # 4. Convert and save file as tensor type , 2 file{tensor_train, tensor labels}  
    # print("\t\t\t-----TENSOR CONVERT-----\t\t\t")
    
    # tensor_train = torch.tensor(processed_data, dtype=torch.float32)
    # tensor_labels = torch.tensor(labels_np, dtype=torch.long)
    
    # # Lưu tensor_data
    # torch.save(tensor_train, path_processed+'tensor_train_9.pt')
    # # Lưu tensor_labels
    # torch.save(tensor_labels, path_processed+'tensor_labels_9.pt')
    
    
    print('\n\t\t\t----------=============\t\tFinish\t\t ----------=============\t\t\t')
    