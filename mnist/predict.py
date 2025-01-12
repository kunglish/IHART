import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
from PIL import Image
import numpy as np
from model.lenet1 import LeNet1
from model.lenet5 import LeNet5



def predict_image(model, image_path):

    classes = [str(i) for i in range(10)]


    data_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        img = Image.open(image_path).convert('L')
        img = data_transform(img)
        img = Variable(torch.unsqueeze(img, dim=0)).to(device)
    except Exception as e:
        print(f"Error: {e}")
        return None

    with torch.no_grad():
        pred = model(img)
        predicted = classes[torch.argmax(pred[0])]

    # print(f'Predicted: "{predicted}"')
    return predicted


if __name__ == "__main__":


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LeNet1().to(device)
    model.load_state_dict(torch.load("./model/lenet1_mnist_model.pth"))
    model.eval()

    # image_path = 'D:/CarDetection/Car-Recognition-master/mnist/result/imageRotation/imageRotation_-10_0_7.jpg'
    # predict = predict_image(model, image_path)
    # print(f'Predicted: "{predict}"')