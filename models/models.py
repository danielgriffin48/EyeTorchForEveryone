import cv2 as cv
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# https://github.com/CSAILVision/GazeCapture/blob/master/models/itracker_train_val.prototxt
class EyeNet(nn.Module):
    def __init__(self):
        super(EyeNet, self).__init__()
        self.net = nn.Sequential(
        #CONV-E1
        nn.Conv2d(3, 96, 11, stride=4),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
        #CONV-E2
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
        #CONV-E3
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        #CONV-E4
        nn.Conv2d(384, 64, kernel_size=1, padding=0),
        nn.ReLU()
        )
    def forward(self, x):
        return self.net(x).flatten(start_dim=1)


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.net = nn.Sequential(
            # CONV-F1
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            # CONV-F2
            nn.MaxPool2d(2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            # CONV-F3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            # CONV-F4
            nn.MaxPool2d(2),
            nn.Conv2d(384, 64, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            # FC-F1
            # TODO why is this only 3?
            nn.Linear(3, 128),
            nn.ReLU(),
            # FC-F2
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        # x = self.fc(x)

        return x.flatten(start_dim=1)


class FaceGrid(nn.Module):
    def __init__(self):
        super(FaceGrid, self).__init__()
        self.fc = nn.Sequential(
            #FG-F1
            nn.Linear(625, 256),
            nn.ReLU(),
            #FG-F2
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(x.flatten(start_dim=1))

class FinalClassifier(nn.Module):
    def __init__(self):
        super(FinalClassifier, self).__init__()
        self.fc = nn.Sequential(
        #FC1
        nn.Linear(832, 128),
        nn.ReLU(),
        #FG-F2
        nn.Linear(128, 2),
        nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(x)


class EyeTrackingForEveryone(nn.Module):
    def __init__(self):
        super(EyeTrackingForEveryone, self).__init__()
        self.eye = EyeNet()
        self.eye_fc = nn.Sequential(
            nn.Linear(25088, 128)
        )
        self.facenet = FaceNet()
        self.fgrid = FaceGrid()
        self.fc = FinalClassifier()

    def forward(self, left_eye, right_eye, face, face_mask):
        left_out = torchvision.transforms.v2.functional.horizontal_flip(self.eye(left_eye))
        right_out = self.eye(right_eye)
        both_eyes_out = self.eye_fc(torch.concat([left_out, right_out], dim=1))

        face_out = self.facenet(face)

        face_mask_out = self.fgrid(face_mask)

        y = self.fc(torch.concat([both_eyes_out, face_out, face_mask_out], dim=1))

        return y

