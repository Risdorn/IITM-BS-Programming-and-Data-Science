from adaptiveLearningRates import main
import matplotlib.pyplot as plt
import numpy as np
from layers import artificialNeuronLayer, convolutionLayer, maxPoolingLayer

class CNN():
    def __init__(self, output_size: int):
        self.conv1 = convolutionLayer(3, depth=3, out_depth=10, stride=1)
        self.pool1 = maxPoolingLayer(2, 2)
        self.conv2 = convolutionLayer(3, depth=10, out_depth=20, stride=1)
        self.pool2 = maxPoolingLayer(2, 2)
        self.fc1 = artificialNeuronLayer(46*65*20, 128)
        self.fc2 = artificialNeuronLayer(128, output_size)
    
    def forward(self, input: float) -> float:
        x = self.conv1.forward(input)
        #print(x.shape)
        x = np.maximum(0, x)
        x = self.pool1.forward(x)
        #print(x.shape)
        x = self.conv2.forward(x)
        #print(x.shape)
        x = np.maximum(0, x)
        x = self.pool2.forward(x)
        #print(x.shape)
        x = x.reshape((1, -1))
        #print(x.shape)
        x = self.fc1.forward(x)
        #print(x.shape)
        x = np.maximum(0, x)
        x = self.fc2.forward(x)
        #x = np.exp(x) # Softmax
        #x = x / np.sum(x) # Softmax
        return x
    
    def backward(self, output_grad: float) -> float:
        #output_grad = output_grad * np.sum(output_grad) # Softmax
        #print(output_grad.shape)
        output_grad = self.fc2.backward(output_grad)
        #print(np.max(output_grad), np.min(output_grad))
        #print(output_grad.shape)
        output_grad = np.maximum(0, output_grad)
        output_grad = self.fc1.backward(output_grad)
        #print(output_grad.shape)
        #print(np.max(output_grad), np.min(output_grad))
        output_grad = output_grad.reshape(46, 65, 20)
        #print(output_grad.shape)
        output_grad = self.pool2.backward(output_grad)
        #print(np.max(output_grad), np.min(output_grad))
        #print(output_grad.shape)
        output_grad = np.maximum(0, output_grad)
        output_grad = self.conv2.backward(output_grad)
        #print(np.max(output_grad), np.min(output_grad))
        #print(output_grad.shape)
        output_grad = self.pool1.backward(output_grad)
        #print(output_grad.shape)
        output_grad = np.maximum(0, output_grad)
        output_grad = self.conv1.backward(output_grad)
        #print(output_grad.shape)
        return output_grad
    
    def update(self, lr: float):
        self.conv1.update(lr)
        self.conv2.update(lr)
        self.fc1.update(lr)
        self.fc2.update(lr)

img = plt.imread('Degree\Deep Learning\dog.jpg')
img = img / 255
img = 2 * img - 1 # Normalizing the image
y = [[1, 0]]
#print(img.shape)
concnetwork = CNN(2)
yhat = concnetwork.forward(img)
print("Current Prediction is: ", yhat)

for i in range(50):
    yhat = concnetwork.forward(img)
    if i % 10 == 0: print("Current Prediction is: ", yhat)
    output_grad = yhat - y
    concnetwork.backward(output_grad)
    concnetwork.update(1e-2)

yhat = concnetwork.forward(img)
print("Current Prediction is: ", yhat)