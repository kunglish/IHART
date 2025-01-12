# -*- coding: utf-8 -*-
import os
import argparse
from collections import deque
import csv
import cv2
import numpy as np
#import rospy
from keras import backend as K
from keras.models import model_from_json


# keras 1.2.2 tf:1.2.0
class chauffeurPredictModel(object):
    def __init__(self, cnn_json_path, cnn_weights_path, lstm_json_path, lstm_weights_path):

        self.cnn = self.load_from_json(cnn_json_path, cnn_weights_path)
        self.encoder = self.load_encoder(cnn_json_path, cnn_weights_path)
        self.lstm = self.load_from_json(lstm_json_path, lstm_weights_path)

        self.scale = 16.
        self.timesteps = 100


        self.timestepped_x = np.empty((1, self.timesteps, 8960))

    def load_encoder(self, cnn_json_path, cnn_weights_path):
        model = self.load_from_json(cnn_json_path, cnn_weights_path)
        model.load_weights(cnn_weights_path)

        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

        return model

    def load_from_json(self, json_path, weights_path):
        model = model_from_json(open(json_path, 'r').read())
        model.load_weights(weights_path)
        return model


    def make_stateful_predictor(self, img1, img2, img3):
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        img3 = cv2.imread(img3)
        steps = deque()

        for img in [img1, img2, img3]:
            # preprocess image to be YUV 320x120 and equalize Y histogram
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img = ((img - (255.0 / 2)) / 255.0)

            # apply feature extractor
            img = self.encoder.predict_on_batch(img.reshape((1, 120, 320, 3)))

            steps.append(img)

        timestepped_x = np.empty((1, self.timesteps, img.shape[1]))
        for i, img in enumerate(steps):
            timestepped_x[0, i] = img

        return self.lstm.predict_on_batch(timestepped_x)[0, 0] / self.scale



def chauffeur_predict_final(file_path, model):

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
        yhat = model.make_stateful_predictor(os.path.join(seed_path, filelist[file_index - 2]),
                             os.path.join(seed_path, filelist[file_index - 1]),
                             os.path.join(seed_path, file_name))
        return yhat
    else:
        return "-0.004179079"



if __name__ == '__main__':
    cnn_json_path = "./model/cnn.json"
    cnn_weights_path = "./model/cnn.weights"
    lstm_json_path = "./model/lstm.json"
    lstm_weights_path = "./model/lstm.weights"

    K.set_learning_phase(0)
    model = chauffeurPredictModel(cnn_json_path, cnn_weights_path, lstm_json_path, lstm_weights_path)

