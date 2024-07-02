import cv2 as cv
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .models.models import EyeTrackingForEveryone
class MPIFaceDataset(Dataset):

    def __init__(self, data_dir="./data/MPIIFaceGaze"):

        self.failed_list = []

        self.data_dir = data_dir
        self.base_options = python.BaseOptions("./blaze_face_short_range.tflite")
        self.options = vision.FaceDetectorOptions(base_options=self.base_options)
        self.detector = vision.FaceDetector.create_from_options(self.options)

        self.BaseOptions = mp.tasks.BaseOptions

        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.lm_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path="./face_landmarker.task"),
            running_mode=self.VisionRunningMode.IMAGE)

        self.fl_detector = self.FaceLandmarker.create_from_options(self.lm_options)
        # self.face_mesh = mp.solutions.face_mesh.FaceMesh()

        self.bin_mask_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((25, 25)),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.eye_face_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((244, 244)),
        ])
        self.file_list = self.load_file_names()

    def get_eyes_and_face(self, image_path):
        cv_img = cv.imread(image_path)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_img)

        res = self.fl_detector.detect(img)
        # try:
        #     landmarks = [i.landmark for i in res[0].multi_face_landmarks]
        # except TypeError:
        #     #print(f"failed {image_path}")
        #     self.failed_list.append(image_path)
        #     # TODO this is added to allow training before fixing, needs removing later
        #     return self.get_eyes_and_face(self.file_list[randint(0, len(self.file_list)-1)][0])
        try:
            right_eye_x2 = int(res.face_landmarks[0][33].x * cv_img.shape[1])
            right_eye_x1 = int(res.face_landmarks[0][133].x * cv_img.shape[1])
            right_eye_y1 = int(res.face_landmarks[0][33].y * cv_img.shape[0])
            right_eye_y2 = int(res.face_landmarks[0][133].y * cv_img.shape[0])
            right_eye_width = right_eye_x1 - right_eye_x2
            right_eye = cv_img[right_eye_y1 - right_eye_width // 2:right_eye_y1 + right_eye_width // 2,
                        right_eye_x2: right_eye_x1]

            left_eye_x2 = int(res.face_landmarks[0][362].x * cv_img.shape[1])
            left_eye_x1 = int(res.face_landmarks[0][263].x * cv_img.shape[1])
            left_eye_y1 = int(res.face_landmarks[0][263].y * cv_img.shape[0])
            left_eye_y2 = int(res.face_landmarks[0][362].y * cv_img.shape[0])
            left_eye_width = left_eye_x1 - left_eye_x2
        except IndexError:
            return False

        left_eye = cv_img[left_eye_y1 - left_eye_width // 2:left_eye_y1 + left_eye_width // 2, left_eye_x2: left_eye_x1]

        detection_result = self.detector.detect(mp.Image.create_from_file(image_path))
        try:
            face_top = detection_result.detections[0].bounding_box.origin_y
            face_bottom = detection_result.detections[0].bounding_box.origin_y + detection_result.detections[
                0].bounding_box.height
            face_left = detection_result.detections[0].bounding_box.origin_x
            face_right = detection_result.detections[0].bounding_box.origin_x + detection_result.detections[
                0].bounding_box.width
        except IndexError:
            # TODO this is added to allow training before fixing, needs removing later
            return False
            # return self.get_eyes_and_face(self.file_list[randint(0, len(self.file_list)-1)][0])

        face = cv_img[face_top:face_bottom, face_left:face_right]

        binary_mask = np.zeros(cv_img.shape[:2])
        binary_mask[face_top:face_bottom, face_left:face_right] = 1
        return left_eye, right_eye, face, binary_mask

    def load_file_names(self):
        file_list = []
        for i in range(15):
            if len(str(i)) == 1:
                i = f"0{i}"
            person_dir = f"{self.data_dir}/p{i}"
            with open(f"{person_dir}/p{i}.txt", 'r') as f:
                lines = f.readlines()
            for l in lines:
                split_line = l.split(" ")
                # 0 - file name
                # 1 - gaze x
                # 2 - gaze y
                if self.get_eyes_and_face(f"{person_dir}/{split_line[0]}") is not False:
                    file_list.append([f"{person_dir}/{split_line[0]}", split_line[1], split_line[2]])

        return file_list

    def __getitem__(self, idx):

        file_name, gaze_x, gaze_y = self.file_list[idx]

        left_eye, right_eye, face, binary_mask = self.get_eyes_and_face(file_name)
        # self.bin_mask_transform(Image.fromarray(binary_mask))
        try:
            left_eye = self.eye_face_transform(left_eye)
            right_eye = self.eye_face_transform(right_eye)
            face = self.eye_face_transform(face)
        except RuntimeError:
            return self.__getitem__(0)

        return left_eye, right_eye, face, self.bin_mask_transform(
            Image.fromarray(binary_mask * 255)), torch.FloatTensor(([int(gaze_x), int(gaze_y)]))

    def __len__(self):

        return len(self.file_list)

train_data = MPIFaceDataset()
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

device = torch.device("cuda")
model = EyeTrackingForEveryone()
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

running_loss = 0
epoch_loss = []
epochs = 5
for e in range(0, epochs):
    running_loss = 0
    for i, data in enumerate(train_loader):
        left_eye, right_eye, face, face_mask, y = data
        optimizer.zero_grad()

        output = model(left_eye.float().to(device), right_eye.float().to(device), face.float().to(device),
                       face_mask.float().to(device))
        loss = loss_fn(output, y.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 25 == 0:
            print(f"Epoch {e} Step {i} loss {loss.item()}")

    print(f"Epoch loss {e + 1} {running_loss / (i + 1)}")
    torch.save(model.state_dict(), f"./increased_learning_rate_01{str(e + 20)}")

