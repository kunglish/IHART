import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch


def get_vggnet_conv3_2_features(image_path, vggnet):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error {image_path}: {e}")
        return None

    image_tensor = transform(image).unsqueeze(0)

    features = None

    def hook(module, input, output):
        nonlocal features
        features = output.detach().cpu().numpy()

    vggnet.features[12].register_forward_hook(hook)

    with torch.no_grad():
        _ = vggnet(image_tensor)


    features = (features - features.mean()) / features.std()
    features = features.reshape(1, -1)

    return features

def compare_feature(matrix1, matrix2, compare_type='ED'):
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrix dimensions are inconsistent!")

    if compare_type == 'ED':
        euclidean_dist = np.linalg.norm(matrix1 - matrix2)
        return euclidean_dist
    if compare_type == 'PD':
        pearson_corr = np.corrcoef(matrix1.ravel(), matrix2.ravel())[0, 1]
        pearson_dist = 1 - np.abs(pearson_corr)
        return pearson_dist
    if compare_type == 'CD':
        matrix1_norm = np.linalg.norm(matrix1, axis=1)
        matrix2_norm = np.linalg.norm(matrix2, axis=1)
        dot_product = np.sum(matrix1 * matrix2, axis=1)
        cosine_dist = 1 - (dot_product / (matrix1_norm * matrix2_norm))
        return cosine_dist
    if compare_type == 'CBD':
        chebyshev_dist = np.max(np.abs(matrix1 - matrix2), axis=1)
        return chebyshev_dist
    if compare_type == 'MD':
        manhattan_dist = np.sum(np.abs(matrix1 - matrix2), axis=1)
        return manhattan_dist
    if compare_type == 'HD':
        hamming_dist = np.sum(matrix1 != matrix2, axis=1)
        return hamming_dist




def MSE(img1,img2):
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    diff_squared = cv2.absdiff(img1, img2) ** 2
    mse = np.mean(diff_squared)
    return mse


