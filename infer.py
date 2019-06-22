import numpy as np
from PIL import Image
import json
import imutils

import torch
import torchvision.transforms as transforms

from matplotlib.pyplot import imread

from utils.transform import ToTensor

CONFIG_PATH = "./config.json"


def resize_gray(image):
    # convert to grayscale
    height, width, _ = image.shape

    # Create a black image
    x = height if height > width else width
    y = height if height > width else width
    square = np.zeros((x, y, 3), np.uint8)
    #
    # This does the job
    #
    square[int((y - height) / 2):int(y - (y - height) / 2), int((x - width) / 2):int(x - (x - width) / 2)] = image
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)

    # crop and resize
    # rect_img = gray[int(height/2-width/2) : int(height/2+width/2), 0 : int(width)]
    resized = cv2.resize(gray, (28, 28))

    return resized


def _infer(path_to_classifier, image, angle):

    transform = transforms.Compose([ToTensor()])

    rotated = imutils.rotate(image, angle)

    image = resize_gray(rotated)

    classifier = torch.load(path_to_classifier).eval()

    image = image.reshape(28, 28)

    label = np.array(9)

    sample = {"image": image, "label": label, "image_path": path_to_input_image}

    data = transform(sample)

    test_x, test_y, path = data["image"], data["label"], data["image_path"]

    test_x.unsqueeze_(0)
    pred = classifier.forward(test_x)
    y_hat = np.argmax(pred.data)

    print("pred: ", y_hat)
    print("truth: ", test_y)


if __name__ == '__main__':

    with open(CONFIG_PATH) as cb:
        config = json.loads(cb.read())

        path_to_classifier = config["infer"]["path_to_classifier"]
        path_to_input_image = config["infer"]["path_to_input_image"]

        _infer(path_to_classifier, path_to_input_image)
