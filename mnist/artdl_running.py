import os.path
import openpyxl
import time
import sys
import torch
import torchvision.models as models
from model.lenet1 import LeNet1
from model.lenet5 import LeNet5
from artdl_testing import  art_feature_distance


def load_dataset():
    path = os.getcwd()
    # print(path)
    originalDir = "dataset/mnist_images"
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

def artdl(num, model_type, compare_type, subdataset="None", hash_type="None", k1 = 10, k2 = 5):
    originalDataset, noiseDataset = load_dataset()

    # get subdataset
    if subdataset != 'None':
        subfilename = str(subdataset)+'.jpg'
        originalDataset = [file for file in originalDataset if file.endswith(subfilename)]
        noiseDataset = [file for file in noiseDataset if file.endswith(subfilename)]


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'lenet1':
        model = LeNet1().to(device)
        model.load_state_dict(torch.load("./model/lenet1_mnist_model.pth"))
        model.eval()

    elif model_type == 'lenet5':
        model = LeNet5().to(device)
        model.load_state_dict(torch.load("./model/lenet5_mnist_model.pth"))
        model.eval()


    vggnet = models.vgg16(pretrained=True)


    workbook = openpyxl.Workbook()

    worksheet = workbook.active

    worksheet['A1'] = "No"
    worksheet['B1'] = "Type"
    worksheet['C1'] = "F-measures"
    worksheet['D1'] = "F-time"

    for i in range(num):
        Type = compare_type + "ART"
        list = [i, Type]
        start = time.time()
        F_meatures = art_feature_distance(originalDataset, noiseDataset, vggnet, model, compare_type)
        list.append(F_meatures)
        list.append(time.time() - start)
        print(list)
        worksheet.append(list)

    filename = compare_type +"ART_" + str(k1) + "_" + str(k2) + "_mnist_" + str(model_type) +"_output.xlsx"
    filepath = './artTesting/' + str(model_type)
    if subdataset != 'None':
        filepath = os.path.join(filepath, str(subdataset))
    filepath = os.path.join(filepath, filename)

    workbook.save(filepath)






