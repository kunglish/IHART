import os.path
import openpyxl
import time
import sys
import torch
from model.resnet18 import ResNet18
from model.resnet20 import ResNet20
from artdl_testing import  art_hash_distance

def load_dataset():
    path = os.getcwd()
    # print(path)
    originalDir = "dataset/cifar10_images"
    noiseDir = "result"

    originalDataset = []
    noiseDataset = []

    originalPath = os.path.join(path, originalDir)

    for filename in os.listdir(originalPath):

        if filename.endswith(".jpg") or filename.endswith(".png"):

            file_path = os.path.join(originalPath, filename)
            file_path = file_path.replace("\\", "/")

            originalDataset.append(file_path)

    # print(originalDataset)


    noisePath = os.path.join(path, noiseDir)

    for root, dirs, files in os.walk(noisePath):

        for file in files:

            if file.endswith(".jpg") or filename.endswith(".png"):

                file_path = os.path.join(root, file)
                file_path = file_path.replace("\\", "/")

                noiseDataset.append(file_path)

    # print(noiseDataset)


    return originalDataset, noiseDataset

def hash_artdl(num, model_type, compare_type, subdataset="None", hash_type='P', k1 = 10, k2 = 5):
    originalDataset, noiseDataset = load_dataset()
    print(subdataset)

    # get subdataset
    if subdataset != 'None':
        subfilename = str(subdataset)+'.jpg'
        originalDataset = [file for file in originalDataset if file.endswith(subfilename)]
        noiseDataset = [file for file in noiseDataset if file.endswith(subfilename)]

    # print(originalDataset)
    # print(noiseDataset)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'resnet18':
        model = ResNet18().to(device)
        model.load_state_dict(torch.load("./model/resnet18_cifar10_model.pth"))
        model.eval()  #

    elif model_type == 'resnet20':
        model = ResNet20().to(device)
        model.load_state_dict(torch.load("./model/resnet20_cifar10_model.pth"))
        model.eval()

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
        F_meatures = art_hash_distance(originalDataset, noiseDataset, model, hash_type, compare_type)
        list.append(F_meatures)
        list.append(time.time() - start)
        print(list)
        worksheet.append(list)

    filename = compare_type + hash_type + "Hash_" + str(k1) + "_" + str(k2) + "_cifar10_" + str(model_type) +"_output.xlsx"
    filepath = './artTesting/' + str(model_type) +'/hash/'
    if subdataset != 'None':
        filepath = os.path.join(filepath, str(subdataset))
    filepath = os.path.join(filepath, filename)

    workbook.save(filepath)



