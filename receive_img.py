import cv2
import time
import os

def video_to_images(video_path, output_dir, frame_rate=1):
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    # 获取视频的总帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")

    # 计数器
    frame_idx = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # 每隔指定的帧率保存一帧
        if frame_idx % int(fps / frame_rate) == 0:
            # cv2.imshow('Video Frame', frame)
            img_name = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(img_name, frame)
            # cv2.imshow('a',frame)
            # time.sleep(1)
            print(f"Saved {img_name}")
            saved_frame_count += 1
            #time.sleep(1.5)

        frame_idx += 1

    # 释放视频捕获对象
    cap.release()
    print("Video processing completed.")

# 示例用法
video_path = "G:\科研\\sit.mp4"
output_dir = "E:\python_projects\img_receive"
frame_rate = 1  # 例如，保存每秒1帧


video_to_images(video_path, output_dir, frame_rate)