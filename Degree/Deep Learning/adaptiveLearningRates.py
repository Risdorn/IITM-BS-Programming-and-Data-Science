from layers import artificialNeuronLayer
import numpy as np
import matplotlib.pyplot as plt

class adaptiveLearningRate():
    def __init__(self, input_size: int, hidden_size: list, output_size: int, updateType: str = 'default'):
        assert updateType in ['adaGrad', 'rmsProp', 'adaDelta', 'adam', 'maxProp', 'adaMax', 'Nadam', 'default']
        assert len(hidden_size) > 0
        self.layers = []
        self.vt = []
        self.mt = []
        for i in range(len(hidden_size)):
            if i == 0:
                self.layers.append(artificialNeuronLayer(input_size, hidden_size[i]))
                #print("Layer 0 weight and bias size: ", self.layers[0].weights.shape, self.layers[0].bias.shape)
                self.vt.append(np.zeros((input_size+1, hidden_size[i]))) # We add 1 to the input size to account for the bias
                self.mt.append(np.zeros((input_size+1, hidden_size[i])))
            else:
                self.layers.append(artificialNeuronLayer(hidden_size[i - 1], hidden_size[i]))
                #print("Layer ", i, " weight and bias size: ", self.layers[i].weights.shape, self.layers[i].bias.shape)
                self.vt.append(np.zeros((hidden_size[i - 1]+1, hidden_size[i])))
                self.mt.append(np.zeros((hidden_size[i - 1]+1, hidden_size[i])))
        self.outputLayer = artificialNeuronLayer(hidden_size[-1], output_size)
        #print("Output Layer weight and bias size: ", self.outputLayer.weights.shape, self.outputLayer.bias.shape)
        self.vt.append(np.zeros((hidden_size[-1]+1, output_size)))
        self.mt.append(np.zeros((hidden_size[-1]+1, output_size)))
        self.updateType = updateType
        self.k = 0 # Corresponds to the vt that is being used in the update method
        self.t = 0 # corresponds to t in all the methods, time step
        self.beta1 = 0.1 # corresponds to beta, and beta1 in all the methods
        self.beta2 = 0.001 # corresponds to beta2 in all the methods
        self.eta = 1e-3 # corresponds to lr in all the methods
    
    def forward(self, input: float) -> float:
        x = input
        for layer in self.layers:
            x = layer(x)
            x = np.maximum(0, x) # ReLU activation function
        x = self.outputLayer(x)
        x = x # no activation function for the output layer
        return x
    
    def backward(self, output_grad: float) -> float:
        output_grad = self.outputLayer.backward(output_grad)
        for layer in reversed(self.layers):
            output_grad *= (output_grad > 0) # ReLU derivative
            output_grad = layer.backward(output_grad)
        return output_grad
    
    def update(self):
        method = getattr(self, self.updateType)
        self.t += 1
        for layer in self.layers:
            weights = np.concatenate((layer.weights, layer.bias), axis = 0)
            dw = method(weights)
            layer.weights -= dw[:-1]
            layer.bias -= dw[-1]
            self.k += 1
        weights = np.concatenate((self.outputLayer.weights, self.outputLayer.bias), axis = 0)
        dw = method(weights)
        self.outputLayer.weights -= dw[:-1]
        self.outputLayer.bias -= dw[-1]
        self.k = 0
    
    def adaGrad(self, grad):
        self.vt[self.k] += grad**2
        return self.eta / np.sqrt(self.vt[self.k] + 1e-8) * grad
    
    def rmsProp(self, grad):
        self.vt[self.k] = self.beta1 * self.vt[self.k] + (1 - self.beta1) * grad**2 # beta1 is beta
        return (self.eta / np.sqrt(self.vt[self.k] + 1e-8)) * grad
    
    def adaDelta(self, grad):
        self.vt[self.k] = self.beta1 * self.vt[self.k] + (1 - self.beta1) * grad**2 # beta1 is beta
        dw = np.sqrt(self.mt[self.k] + 1e-8) / np.sqrt(self.vt[self.k] + 1e-8) * grad # mt is ut
        self.mt[self.k] = self.beta1 * self.mt[self.k] + (1 - self.beta1) * dw**2
        return dw
    
    def adam(self, grad):
        self.mt[self.k] = self.beta1 * self.mt[self.k] + (1 - self.beta1) * grad
        m_hat = self.mt[self.k] / (1 - self.beta1**self.t)
        self.vt[self.k] = self.beta2 * self.vt[self.k] + (1 - self.beta2) * grad**2
        v_hat = self.vt[self.k] / (1 - self.beta2**self.t)
        return (self.eta / (np.sqrt(v_hat) + 1e-8)) * m_hat
    
    def maxProp(self, grad):
        self.vt[self.k] = np.maximum(self.beta1 * self.vt[self.k], abs(grad)) # beta1 is beta
        return (self.eta / (self.vt[self.k] + 1e-8)) * grad
        
    def adaMax(self, grad):
        self.mt[self.k] = self.beta1 * self.mt[self.k] + (1 - self.beta1) * grad
        m_hat = self.mt[self.k] / (1 - self.beta1**self.t)
        self.vt[self.k] = np.maximum(self.beta2 * self.vt[self.k], abs(grad))
        return (self.eta / (self.vt[self.k] + 1e-8)) * m_hat
    
    def Nadam(self, grad):
        self.mt[self.k] = self.beta1 * self.mt[self.k] + (1 - self.beta1) * grad
        m_hat = self.mt[self.k] / (1 - self.beta1**self.t)
        self.vt[self.k] = self.beta2 * self.vt[self.k] + (1 - self.beta2) * grad**2
        v_hat = self.vt[self.k] / (1 - self.beta2**self.t)
        dw = (self.eta / np.sqrt(v_hat) + 1e-8) * (m_hat + ((1 - self.beta1) * grad / (1 - self.beta1**self.t)))
        return dw
    
    def default(self, grad):
        return self.eta * grad


def main():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([[1], [1], [1], [0]])
    nn = adaptiveLearningRate(2, [4], 1, 'default')
    nn.eta = 1e-4
    Yhat = nn.forward(X)
    print("Current Prediction: ", Yhat)
    printAt = 100
    for i in range(2000):
        Yhat = nn.forward(X)
        loss = np.sum((Y - Yhat)**2)
        #if i % printAt == 0: print("Loss: ", loss, "Iteration: ", i, "Prediction: ", Yhat)
        grad = 2 * (Yhat - Y)
        nn.backward(grad)
        nn.update()
    
    types = ['adaGrad', 'rmsProp', 'adaDelta', 'adam', 'maxProp', 'adaMax', 'Nadam', 'default']
    plt.figure(figsize = (20, 10))
    for i in range(len(types)):
        nn = adaptiveLearningRate(2, [4], 1, types[i])
        Yhat = nn.forward(X)
        loss = []
        for j in range(1000):
            Yhat = nn.forward(X)
            loss.append(np.sum((Y - Yhat)**2))
            grad = 2 * (Yhat - Y)
            nn.backward(grad)
            nn.update()
        print("Final prediction for ", types[i], ": ", Yhat)
        print("Final loss for ", types[i], ": ", loss[-1])
        plt.plot(loss, label = types[i])
    plt.legend()
    plt.show()
