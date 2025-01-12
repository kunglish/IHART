import os
import argparse
import csv
import numpy as np
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity
import cv2

class ramboPredictModel(object):
    def __init__(self, model_path, X_train_mean_path):
        self.model = load_model(model_path)
        self.model.compile(optimizer="adam", loss="mse")
        self.X_mean = np.load(X_train_mean_path)
        self.mean_angle = np.array([-0.004179079])
        self.img0 = None
        self.state = deque(maxlen=2)

    def predict(self, img_path0, img_path1, img_path2):
        self.img0 = load_img(img_path0, target_size=(192, 256))
        self.img0 = self.img0.convert('L')
        self.img0 = img_to_array(self.img0)
        img1 = load_img(img_path1, target_size=(192, 256))
        img1 = img1.convert('L')
        img1 = img_to_array(img1)

        img = img1 - self.img0
        img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
        img = np.array(img, dtype=np.uint8)
        self.state.append(img)
        self.img0 = img1

        img1 = load_img(img_path2, target_size=(192, 256))
        img1 = img1.convert('L')
        img1 = img_to_array(img1)
        img = img1 - self.img0
        img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
        img = np.array(img, dtype=np.uint8)  # to replicate initial model
        self.state.append(img)
        self.img0 = img1

        X = np.concatenate(tuple(self.state), axis=-1)
        X = X[:, :, ::-1]
        X = np.expand_dims(X, axis=0)
        X = X.astype('float32')
        X -= self.X_mean
        X /= 255.0
        return self.model.predict(X)[0]


def rambo_predict_final(file_path, model):
    # model = ramboPredictModel("./model/final_model.hdf5", "./model/X_train_mean.npy")
    file_name = os.path.basename(file_path)

    seed_path1 = "Dataset/hmb3"
    seed_path2 = "result"

   #  print(file_name)
    if "_" in file_name:
        dir_name = os.path.basename(os.path.dirname(file_path))
        seed_path = os.path.join(seed_path2, dir_name)
    else:
        seed_path = seed_path1


    filelist = sorted(os.listdir(seed_path))

    file_index = filelist.index(file_name)
    if file_index > 1:
        yhat = model.predict(os.path.join(seed_path, filelist[file_index - 2]),
                             os.path.join(seed_path, filelist[file_index - 1]),
                             os.path.join(seed_path, file_name))
        return yhat[0]
    else:
        return "-0.004179079"



if __name__ == "__main__":
    model = ramboPredictModel("./model/final_model.hdf5", "./model/X_train_mean.npy")
