#coding=utf-8
import random
import numpy as np
import cv2
import os


def imageTranslation(img, params):
    if not isinstance(params, list):
        params = [params, params]
    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst


def imageShear(img, params):
    rows, cols, ch = img.shape
    factor = params * (-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def imageRotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def imageContrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))

    return new_img


def imageBrightness(img, params):
    beta = params
    new_img = cv2.add(img, beta)

    return new_img


def imageBlur(img, params):
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    if params == 9:
        blur = cv2.blur(img, (6, 6))
    if params == 10:
        blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def imageMotionBlur(image, degree, angle=45):
    image = np.array(image)

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def loadData(dataset_path):
    print(dataset_path, type(dataset_path))
    seed_inputs = os.path.join(dataset_path, "dataset/driving_images")

    filelist = []
    for file in sorted(os.listdir(seed_inputs)):
        if file.endswith(".jpg"):
            filelist.append(file)


    return filelist

    print("Data load success!!!")


def addNoise(filelist):

    for i in range(len(filelist)):
        img_path = os.path.join(dir_path, filelist[i])
        # print("Image path:" ,img_path)
        img = cv2.imread(img_path)

        path = os.getcwd()
        result_path = os.path.join(path, 'result')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print("Folder created")
        else:
            print("Folder already exists")

        transformations = [imageTranslation, imageShear, imageRotation,
                           imageContrast, imageBrightness, imageBlur, imageMotionBlur]
        params = []
        params.append(list(range(-50, 51, 10)))
        params.append(list(map(lambda x: x * 0.1, list(range(-5, 5)))))
        params.append(list(range(-30, 30)))
        params.append(list(map(lambda x: x * 0.1, list(range(1, 20)))))
        params.append(list(range(-21, 21)))
        params.append(list(range(1, 11)))
        params.append(list(range(1, 13)))

        for j in range(len(transformations)):
            transformation = transformations[j]

            file = str(transformation).split(" ", 1)[1].split(" ", 1)[0]

            output_path = os.path.join(result_path, file)

            if not os.path.exists(output_path):
                os.makedirs(output_path)
                print("Folder created")
            else:
                print("Folder already exists")

            for param in params[j]:
                param = round(param, 1)
                new_img = transformation(img, param)
                save_file = os.path.join(output_path, file) + "_" + str(param) + "_" + filelist[i]
                print("file:", save_file)
                cv2.imwrite(save_file, new_img)

def maim():
    path = os.getcwd()
    filelist = loadData(path)
    addNoise(filelist)


if __name__ == "__main__":
    maim()
