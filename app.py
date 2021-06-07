# Flask Modules
from flask import Flask, flash, render_template, request, redirect, url_for, abort

# PyTorch Modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Face-net Modules
from facenet_pytorch import MTCNN

# Other Modules
import os
import glob
import numpy as np
import cv2
import ssl
from PIL import Image
from datetime import datetime
from scipy.spatial import distance

# Set-up
ssl._create_default_https_context = ssl._create_unverified_context

# Face Mask Detection Models 
def ResNet34():
    model = models.resnet34(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.avgpool = nn.AdaptiveAvgPool2d(1)

    model.fc = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid())

    return model

def ResNet18():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.avgpool = nn.AdaptiveAvgPool2d(1)

    model.fc = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid())

    return model

def VGG19():
    model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[6].in_features
    model.classifier[5] = nn.Linear(num_ftrs, 1)
    model.classifier[6] = nn.Sigmoid()


    return model

# Load a model from .pth file.
device = torch.device("cpu")
model = ResNet34().to(device)
model.load_state_dict(
    torch.load("./model/resnet34_model_v0.pth", map_location=lambda storage, loc: storage)
)
model.eval()

model1 = VGG19().to(device)
# Switch the file path depending on what model you will use.
model1.load_state_dict(
    torch.load("./model/vgg19_model_v0.pth", map_location=lambda storage, loc: storage)
)
model1.eval()

# Face Detection Models
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device=device)

nose_detect_model = cv2.CascadeClassifier('./model/nose.xml')
mouth_detect_model = cv2.CascadeClassifier('./model/mouth.xml')

# Utils
def draw_bbox(bounding_boxes, image):
    for i in range(len(bounding_boxes)):
        x1, y1, x2, y2 = bounding_boxes[i]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    
    return image

def plot_landmarks(landmarks, image):
    for i in range(len(landmarks)):
        for p in range(landmarks[i].shape[0]):
            cv2.circle(image, 
                      (int(landmarks[i][p, 0]), int(landmarks[i][p, 1])), 1, (0, 0, 255), -1, cv2.LINE_AA)
    return image


# Constants
MIN_DISTANCE = 200
MASK_LABEL = {0: "Mask", 1:"No Mask", 2: 'Mask below Nose', 3: 'Mask under Chin'}
MASK_ON_LABEL = {0: (0, 255, 0), 1: (255, 0, 0), 2: (255, 165, 0), 3: (255, 165, 0)}
DIST_LABEL = {0: "", 1: "Too close"}

# Config web app
app = Flask(__name__)
app.config["SECRET_KEY"] = "Sq[UFa!2bk$UG9Zx"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def main_page():
    if request.method == "GET":
        return render_template("index.html")

@app.route("/vgg", methods=["GET", "POST"])
def upload_file_vgg():
    if request.method == "GET":
        return render_template("vgg.html")
    if request.method == "POST":
        # Error handling
        if "file" not in request.files:
            flash("Submitted an empty file")
            return render_template("vgg.html")

        f = request.files["file"]
        if f.filename == '':
            flash("Submitted an empty file")
            return render_template("vgg.html")
        
        if not allowed_file(f.filename):
            flash("File type must be png, jpg, or jpeg")
            return render_template("vgg.html")

        # Init a folder
        for p in glob.glob("./static/*.png"):
            if os.path.isfile(p):
                os.remove(p)

        # Save an uploaded image
        filename = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + "_vgg19"
        filepath = filename + ".png"
        f.save(filepath)

        # Read the image
        input_img = cv2.imread(filepath)
        converted_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        faces, conf = mtcnn.detect(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

        numMask = 0
        numNonMask = 0
        numNonDist = 0
        numNonMaskNose = 0

        if faces is None:
            flash("No faces were detected")
            return render_template("vgg.html", filepath=filepath)

        if len(faces) >= 1:
            label = [0 for i in range(len(faces))]
            for i in range(len(faces) - 1):
                for j in range(i + 1, len(faces)):
                    dist = distance.euclidean(faces[i][:2],faces[j][:2])
                    if dist < MIN_DISTANCE:
                        label[i] = 1
                        label[j] = 1

            new_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
            for i in range(len(faces)):
                (x, y, w, h) = faces[i]
                crop = new_img[int(y) : int(h), int(x) : int(w)]
                crop = cv2.resize(crop,(224, 224))
                crop = torch.tensor(crop/255)
                crop = crop.permute(2, 0, 1)
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                crop = normalize(crop)
                crop = crop.float().to(device)
                crop = crop.unsqueeze(0)
                pred = model1(crop)
                mask_result = 1
                if pred > 0.8:
                    mask_result = 0

                    # Check if nose is covered correctly
                    noses = nose_detect_model.detectMultiScale(new_img[int(y) : int(h), int(x) : int(w)], scaleFactor=1.1, minNeighbors=4)
                    if len(noses) >= 1:
                        mask_result = 2
                
                cv2.putText(new_img, MASK_LABEL[mask_result], (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MASK_ON_LABEL[mask_result], 2)
                d = label[i]
                if mask_result == 0:
                    d = 0
                cv2.rectangle(new_img, (int(x), int(y)), (int(w), int(h)), MASK_ON_LABEL[mask_result], 1)
                cv2.putText(new_img, DIST_LABEL[d], (int(x), int(h) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MASK_ON_LABEL[mask_result], 2)

                if mask_result == 0:
                    numMask += 1
                else:
                    numNonMask += 1

                if mask_result == 2:
                    numNonMaskNose += 1

                if d == 1:
                    numNonDist += 1

            processedImg = Image.fromarray(new_img)
            processedImg.save(filename + "_processed.png")
        else:
            flash("No face has been detected")
            return render_template("vgg.html", filepath=filepath)

        # Clean-up
        if os.path.exists(filepath):
            os.remove(filepath)
        return render_template("vgg.html", filepath=filename + "_processed.png", result=len(faces), mask=numMask, nonmask=numNonMask, nondist=numNonDist, nonmasknose=numNonMaskNose)

@app.route("/resnet", methods=["GET", "POST"])
def upload_file_resnet():
    if request.method == "GET":
        return render_template("resnet.html")
    if request.method == "POST":
        # Error handling
        if "file" not in request.files:
            flash("Submitted an empty file")
            return render_template("resnet.html")

        f = request.files["file"]
        if f.filename == '':
            flash("Submitted an empty file")
            return render_template("resnet.html")
        
        if not allowed_file(f.filename):
            flash("File type must be png, jpg, or jpeg")
            return render_template("resnet..html")

        # Init a folder
        for p in glob.glob("./static/*.png"):
            if os.path.isfile(p):
                os.remove(p)

        # Save an uploaded image
        filename = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + "_resnet34"
        filepath = filename + ".png"
        f.save(filepath)

        # Read the image
        input_img = cv2.imread(filepath)
        converted_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        faces, conf = mtcnn.detect(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

        numMask = 0
        numNonMask = 0
        numNonDist = 0
        numNonMaskNose = 0

        if faces is None:
            flash("No faces were detected")
            return render_template("resnet.html", filepath=filepath)

        if len(faces) >= 1:
            label = [0 for i in range(len(faces))]
            for i in range(len(faces) - 1):
                for j in range(i + 1, len(faces)):
                    dist = distance.euclidean(faces[i][:2],faces[j][:2])
                    if dist < MIN_DISTANCE:
                        label[i] = 1
                        label[j] = 1

            new_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
            for i in range(len(faces)):
                (x, y, w, h) = faces[i]
                crop = new_img[int(y) : int(h), int(x) : int(w)]
                crop = cv2.resize(crop,(224, 224))
                crop = torch.tensor(crop/255)
                crop = crop.permute(2, 0, 1)
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                crop = normalize(crop)
                crop = crop.float().to(device)
                crop = crop.unsqueeze(0)
                pred = model(crop)
                mask_result = 1
                if pred > 0.8:
                    mask_result = 0

                    # Check if nose is covered correctly
                    noses = nose_detect_model.detectMultiScale(new_img[int(y) : int(h), int(x) : int(w)], scaleFactor=1.1, minNeighbors=4)
                    if len(noses) >= 1:
                        mask_result = 2
                
                cv2.putText(new_img, MASK_LABEL[mask_result], (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MASK_ON_LABEL[mask_result], 2)
                d = label[i]
                if mask_result == 0:
                    d = 0
                cv2.rectangle(new_img, (int(x), int(y)), (int(w), int(h)), MASK_ON_LABEL[mask_result], 1)
                cv2.putText(new_img, DIST_LABEL[d], (int(x), int(h) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MASK_ON_LABEL[mask_result], 2)

                if mask_result == 0:
                    numMask += 1
                else:
                    numNonMask += 1

                if mask_result == 2:
                    numNonMaskNose += 1

                if d == 1:
                    numNonDist += 1

            processedImg = Image.fromarray(new_img)
            processedImg.save(filename + "_processed.png")
        else:
            flash("No face has been detected")
            return render_template("resnet.html", filepath=filepath)

        # Clean-up
        if os.path.exists(filepath):
            os.remove(filepath)
        return render_template("resnet.html", filepath=filename + "_processed.png", result=len(faces), mask=numMask, nonmask=numNonMask, nondist=numNonDist, nonmasknose=numNonMaskNose)

if __name__ == "__main__":
    app.run()