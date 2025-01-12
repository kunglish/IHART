import os.path
import openpyxl
import time
import sys
import csv
import torch
from model.chauffeur_predict import chauffeurPredictModel
from model.rambo_predict import ramboPredictModel
from artdl_testing import  art_hash_distance
from keras import backend as K

def load_dataset():
    path = os.getcwd()
    originalDir = "dataset/driving_images"
    noiseDir = "result"

    originalDataset = []
    noiseDataset = []

    originalPath = os.path.join(path, originalDir)

    for filename in os.listdir(originalPath):

        if filename.endswith(".jpg"):

            file_path = os.path.join(originalPath, filename)
            file_path = file_path.replace("\\", "/")

            originalDataset.append(file_path)

    # print(originalDataset)


    noisePath = os.path.join(path, noiseDir)

    for root, dirs, files in os.walk(noisePath):

        for file in files:

            if file.endswith(".jpg"):

                file_path = os.path.join(root, file)
                file_path = file_path.replace("\\", "/")

                noiseDataset.append(file_path)

    # print(noiseDataset)

    # load label
    labels = {}
    csvPath = os.path.join(originalPath, "hmb3_steering.csv")
    csvPath = csvPath.replace("\\", "/")
    with open(csvPath, 'r') as csvfile:
        label = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
    label = label[1:]
    for i in label:
        labels[i[0]+".jpg"] = i[1]
    # print(labels)

    return originalDataset, noiseDataset, labels

def hash_artdl(num, model_type, compare_type, hash_type='P', k1 = 10, k2 = 5):
    originalDataset, noiseDataset, labels = load_dataset()

    # 加载模型
    if model_type == 'chauffeur':
        cnn_json_path = "./model/cnn.json"
        cnn_weights_path = "./model/cnn.weights"
        lstm_json_path = "./model/lstm.json"
        lstm_weights_path = "./model/lstm.weights"
        K.set_learning_phase(0)
        model = chauffeurPredictModel(cnn_json_path, cnn_weights_path, lstm_json_path, lstm_weights_path)

    elif model_type == 'rambo':
        model = ramboPredictModel("./model/final_model.hdf5", "./model/X_train_mean.npy")


    workbook = openpyxl.Workbook()

    worksheet = workbook.active

    worksheet['A1'] = "No"
    worksheet['B1'] = "Type"
    worksheet['C1'] = "F-measures"
    worksheet['D1'] = "F-time"

    for i in range(num):
        Type = compare_type + hash_type + "Hash"
        list = [i, Type]
        start = time.time()
        F_meatures = art_hash_distance(model_type, originalDataset, noiseDataset, labels, model, hash_type, compare_type)
        list.append(F_meatures)
        list.append(time.time() - start)
        print(list)
        worksheet.append(list)

    filename = compare_type + hash_type + "Hash_" + str(k1) + "_" + str(k2) + "_driving_" + str(model_type) +"_output.xlsx"
    filepath = './artTesting/' + str(model_type) + '/hash/'
    filepath = os.path.join(filepath, filename)

    workbook.save(filepath)


