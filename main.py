import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import mediapipe as mp
from test_code.style_mediapipe import *
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
data_1 = np.load('./test_code/start_up.npy')
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_face_mesh = mp.solutions.face_mesh #Hai_them
face_mesh = mp_face_mesh.FaceMesh() #Hai_them
def draw_plot_mapping(data):
    left_hand_data = np.zeros((124,42))
    right_hand_data = np.zeros((124,42))
    lips_data = np.zeros((124,80))
    # Tạo figure và axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 12))
    plt.subplots_adjust(hspace=0.3)
    # Hàm khởi tạo, không cần thực hiện bất kỳ thay đổi gì ở đây
    def init():
        im1 = ax1.imshow(np.array(left_hand_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title('Left Hand')
        ax1.set_yticks([])  
        
        im2 = ax2.imshow(np.array(right_hand_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_title('Right Hand')
        ax2.set_yticks([]) 
        
        im3 = ax3.imshow(np.array(lips_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax3.set_title('Lips')
        ax3.set_yticks([]) 


        return [im1, im2, im3]
    # Hàm cập nhật animation
    def update(frame):
        # Cập nhật chỉ dòng frame
        left_hand_data[frame,:] = data[frame,:42]
        right_hand_data[frame,:] = data[frame,42:84]
        lips_data[frame,:] = data[frame,84:]
        
        img1 = ax1.imshow(left_hand_data.T, cmap='viridis')
        img2 = ax2.imshow(right_hand_data.T, cmap='viridis')
        img3 = ax3.imshow(lips_data.T, cmap='viridis')

        return [img1, img2, img3]

    # Tạo animation
    animation = FuncAnimation(fig, update, frames=range(200), init_func=init, blit=True)

    plt.show()
n_frames = 0
def draw_plot_mapping_cam():
    global n_frames
    left_hand_data = np.zeros((300,42))
    right_hand_data = np.zeros((300,42))
    lips_data = np.zeros((300,80))
    cap = cv2.VideoCapture(0)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 12))
    plt.subplots_adjust(hspace=0.3)
    # Hàm khởi tạo, không cần thực hiện bất kỳ thay đổi gì ở đây
    def init():
        im1 = ax1.imshow(np.array(left_hand_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title('Left Hand')
        ax1.set_yticks([]) 
        
        im2 = ax2.imshow(np.array(right_hand_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_title('Right Hand')
        ax2.set_yticks([]) 
        
    
        im3 = ax3.imshow(np.array(lips_data).T, aspect='auto', cmap='viridis', origin='lower')
        ax3.set_title('Lips')
        ax3.set_yticks([]) 
        
        return [im1, im2, im3]
    
    # Hàm cập nhật animation
    def update(frame):
        # Cập nhật chỉ dòng frame
        global n_frames
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        keypoints = extract_keypoints(results)
        left_hand_data[n_frames, :] = keypoints[:42]
        right_hand_data[n_frames, :] = keypoints[42:84]
        lips_data[n_frames, :] = keypoints[84:]
        # Cập nhật hình ảnh trên axes
        img1 = ax1.imshow(left_hand_data.T, cmap='viridis')
        img2 = ax2.imshow(right_hand_data.T, cmap='viridis')
        img3 = ax3.imshow(lips_data.T, cmap='viridis')
        n_frames +=1
        return [img1, img2, img3]

    # Tạo animation
    animation = FuncAnimation(fig, update, frames=range(200), init_func=init, blit=True)

    plt.show(block=False)
    # Chạy vòng lặp để hiển thị video từ camera
    while cap.isOpened():
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        frame = cv2.resize(frame, (frame.shape[1]*2+100, frame.shape[0]*2+100))
        results_face_mesh = face_mesh.process(frame_rgb)
        draw_styled_landmarks(frame, results, results_face_mesh)
        cv2.imshow('Video', frame)
        # Kiểm tra phím nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
def map_new_to_old_style(sequence):
    types = []
    landmark_indexes = []
    for column in list(sequence.columns)[1:544]:
        parts = column.split("_")
        if len(parts) == 4:
            types.append(parts[1] + "_" + parts[2])
        else:
            types.append(parts[1])

        landmark_indexes.append(int(parts[-1]))

    data = {
        "frame": [],
        "type": [],
        "landmark_index": [],
        "x": [],
        "y": [],
        "z": []
    }

    for index, row in sequence.iterrows():
        data["frame"] += [int(row.frame)]*543
        data["type"] += types
        data["landmark_index"] += landmark_indexes

        for _type, landmark_index in zip(types, landmark_indexes):
            data["x"].append(row[f"x_{_type}_{landmark_index}"])
            data["y"].append(row[f"y_{_type}_{landmark_index}"])
            data["z"].append(row[f"z_{_type}_{landmark_index}"])

    return pd.DataFrame.from_dict(data)

# assign desired colors to landmarks
def assign_color(row):
    if row == 'face':
        return 'red'
    elif 'hand' in row:
        return 'dodgerblue'
    else:
        return 'green'

# specifies the plotting order
def assign_order(row):
    if row.type == 'face':
        return row.landmark_index + 101
    elif row.type == 'pose':
        return row.landmark_index + 30
    elif row.type == 'left_hand':
        return row.landmark_index + 80
    else:
        return row.landmark_index

def visualise2d_landmarks(parquet_df, title=""):
    connections = [
        [0, 1, 2, 3, 4,],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],


        [38, 36, 35, 34, 30, 31, 32, 33, 37],
        [40, 39],
        [52, 46, 50, 48, 46, 44, 42, 41, 43, 45, 47, 49, 45, 51],
        [42, 54, 56, 58, 60, 62, 58],
        [41, 53, 55, 57, 59, 61, 57],
        [54, 53],


        [80, 81, 82, 83, 84, ],
        [80, 85, 86, 87, 88],
        [80, 89, 90, 91, 92],
        [80, 93, 94, 95, 96],
        [80, 97, 98, 99, 100], ]

    parquet_df = map_new_to_old_style(parquet_df)
    frames = sorted(set(parquet_df.frame))
    first_frame = min(frames)
    parquet_df['color'] = parquet_df.type.apply(lambda row: assign_color(row))
    parquet_df['plot_order'] = parquet_df.apply(lambda row: assign_order(row), axis=1)
    first_frame_df = parquet_df[parquet_df.frame == first_frame].copy()
    first_frame_df = first_frame_df.sort_values(["plot_order"]).set_index('plot_order')


    frames_l = []
    for frame in frames:
        filtered_df = parquet_df[parquet_df.frame == frame].copy()
        filtered_df = filtered_df.sort_values(["plot_order"]).set_index("plot_order")
        traces = [go.Scatter(
            x=filtered_df['x'],
            y=filtered_df['y'],
            mode='markers',
            marker=dict(
                color=filtered_df.color,
                size=9))]

        for i, seg in enumerate(connections):
            trace = go.Scatter(
                    x=filtered_df.loc[seg]['x'],
                    y=filtered_df.loc[seg]['y'],
                    mode='lines',
            )
            traces.append(trace)
        frame_data = go.Frame(data=traces, traces = [i for i in range(17)])
        frames_l.append(frame_data)

    traces = [go.Scatter(
        x=first_frame_df['x'],
        y=first_frame_df['y'],
        mode='markers',
        marker=dict(
            color=first_frame_df.color,
            size=9
        )
    )]
    for i, seg in enumerate(connections):
        trace = go.Scatter(
            x=first_frame_df.loc[seg]['x'],
            y=first_frame_df.loc[seg]['y'],
            mode='lines',
            line=dict(
                color='black',
                width=2
            )
        )
        traces.append(trace)
    fig = go.Figure(
        data=traces,
        frames=frames_l
    )


    fig.update_layout(
        width=500,
        height=800,
        scene={
            'aspectmode': 'data',
        },
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100,
                                                    "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0}}],
                        "label": "&#9654;",
                        "method": "animate",
                    },

                ],
                "direction": "left",
                "pad": {"r": 100, "t": 100},
                "font": {"size":30},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
    )
    camera = dict(
        up=dict(x=0, y=-1, z=0),
        eye=dict(x=0, y=0, z=2.5)
    )
    fig.update_layout(title_text=title, title_x=0.5)
    fig.update_layout(scene_camera=camera, showlegend=False)
    fig.update_layout(xaxis = dict(visible=False),
            yaxis = dict(visible=False),
    )
    fig.update_yaxes(autorange="reversed")

    fig.show()


# def get_phrase(df, file_id, sequence_id):
#     return df[
#         np.logical_and(
#             df.file_id == file_id,
#             df.sequence_id == sequence_id
#         )
#     ].phrase.iloc[0]
    
if __name__ == '__main__':
    #OPTIONS
    
    
    # #1.DRAW WITH 164 landmark:   
    # draw_plot_mapping(np.load('./test_code/start_up.npy'))
    
    # #3.DRAW WITH Camera:
    # draw_plot_mapping_cam()
    
    
    # #2.Visualise all landmark
    # action = pd.read_parquet('./dataset/data.parquet').iloc[:98,:]
    # visualise2d_landmarks(action, f"Phrase: {'Anh Hai'}")