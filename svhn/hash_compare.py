import numpy as np
import cv2


def get_gray_image(image):
    image = cv2.imread(image)

    # Convert images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def aHash(img,shape=(10,10)):
    img = cv2.imread(img)

    img = cv2.resize(img, shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(shape[0]):
        for j in range(shape[1]):
            s = s + gray[i, j]

    avg = s / 100

    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dHash(img,shape=(10,10)):
    img = cv2.imread(img)

    img = cv2.resize(img, (shape[0]+1, shape[1]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str



def pHash(img,shape=(10,10)):
    img = cv2.imread(img)
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:10, 0:10]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)

    return hash


def str_to_list(str):
    list = [int(char) for char in str]
    return list


def compare_hash(hash1, hash2, compare_type, shape=(10,10)):

    if isinstance(hash1, str):
        hash1 = str_to_list(hash1)
    if isinstance(hash2, str):
        hash2 = str_to_list(hash1)

    if len(hash1)!=len(hash2):
        return -1

    hash1 = np.array(hash1)
    hash2 = np.array(hash2)

    if compare_type == 'None':
        n = np.sum(hash1 == hash2)
        similarity = 1 - (n / (shape[0] * shape[1]))
        return similarity

    if compare_type == 'ED':
        euclidean_dist = np.linalg.norm(hash1 - hash2)
        return euclidean_dist
    if compare_type == 'MD':
        manhattan_dist = np.sum(np.abs(hash1 - hash2))
        return manhattan_dist
    if compare_type == 'CBD':
        chebyshev_dist = np.max(np.abs(hash1 - hash2))
        return chebyshev_dist
    if compare_type == 'HD':
        hamming_dist = np.sum(hash1 != hash2)
        return hamming_dist

    if compare_type == 'CD':
        if np.linalg.norm(hash1) == 0 or np.linalg.norm(hash2) == 0:
            cosine_dist = 1
        else:
            cosine_similarity = np.dot(hash1, hash2) / (np.linalg.norm(hash1) * np.linalg.norm(hash2))
            cosine_dist = 1 - cosine_similarity
        return cosine_dist

    if compare_type == 'PD':
        if np.std(hash1) == 0 or np.std(hash2) == 0:
            pearson_dist = 1
        else:
            pearson_corr= np.corrcoef(hash1, hash2)[0,1]
            pearson_dist = 1 - pearson_corr
        return pearson_dist