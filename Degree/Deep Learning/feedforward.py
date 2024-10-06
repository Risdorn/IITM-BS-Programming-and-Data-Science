import numpy as np

class feedforward():
    """
    This class implements a feedforward neural network.
    The network is defined by a list of layers, each layer being a list of neurons.
    The network is trained using backpropagation.
    Functions:
    - activation(x): returns the output of the activation function for a given input x.
    - forwardPass(x): returns the output of the network for a given input x.
    - backprop(x, y): returns the gradients of the loss function with respect to the weights and bias.
    - loss(x, y): returns the loss function for a given input x and output y.
    - updateParameters(dL_dW, dL_db, eta): updates the weights and bias using the gradients and a learning rate eta.
    """
    def __init__(self, layers=[], weights=None, bias=None):
        self.layers = layers
        if weights is None:
            self.weights = [np.random.rand(layers[i+1], layers[i]) for i in range(len(layers)-1)]
        else:
            assert len(weights) == len(layers)-1, "Number of weight matrices must be equal to the number of layers minus one."
            self.weights = weights
        if bias is None:
            self.bias = [np.random.rand(layers[i+1], 1) for i in range(len(layers)-1)]
        else:
            assert len(bias) == len(layers)-1, "Number of bias vectors must be equal to the number of layers minus one."
            self.bias = bias
    
    def activation(self, x):
        return 1 / (1 + np.exp(-x))
    
    def outputActivation(self, x):
        return np.exp(x) / np.exp(x).sum()
    
    def forwardPass(self, x):
        a = [ ]
        h = [ ]
        for i in range(len(self.layers)-2):
            a.append(np.dot(self.weights[i], x) + self.bias[i])
            h.append(self.activation(a[-1]))
            x = h[-1]
        a.append(np.dot(self.weights[-1], x) + self.bias[-1])
        h.append(self.outputActivation(a[-1]))
        return a, h
    
    def backprop(self, x, y):
        _, h = self.forwardPass(x)
        h.insert(0, x)
        dL_da = [ ]
        dL_dh = [ ]
        dL_dW = [ ]
        dL_db = [ ]
        dL_da.append(h[-1] - y)
        dL_dW.append(np.dot(dL_da[-1], h[-2].T))
        dL_db.append(dL_da[-1])
        for i in range(len(self.layers)-2, 0, -1):
            dL_dh.append(np.dot(self.weights[i].T, dL_da[-1]))
            dL_da.append(np.multiply(dL_dh[-1], np.multiply(h[i], (1 - h[i]))))
            dL_dW.append(np.dot(dL_da[-1], h[i-1].T))
            dL_db.append(dL_da[-1])
        return dL_dW[::-1], dL_db[::-1]
    
    def loss(self, x, y):
        # Cross-entropy loss
        _, h = self.forwardPass(x)
        yHat = h[-1]
        return -np.sum(y * np.log(yHat))
    
    def updateParameters(self, dL_dW, dL_db, eta):
        for i in range(len(self.layers)-1):
            self.weights[i] -= eta * dL_dW[i]
            self.bias[i] -= eta * dL_db[i]

def main():
    files = np.load('Degree/Deep Learning/parameters.npz')
    weights = [files.get('W1'), files.get('W2'), files.get('W3')]
    biases = [files.get('b1'), files.get('b2'), files.get('b3')]
    model = feedforward(layers=[3,3,3,3], weights=weights, bias=biases)
    x = np.array([1,0,1])
    x.shape = (3,1)
    y = np.array([0,0,1])
    y.shape = (3,1)
    a, h = model.forwardPass(x)
    print("a[0], h[0] sum is", sum(a[0]), sum(h[0]))
    print("a[1], h[1] sum is", sum(a[1]), sum(h[1]))
    print("a[2], h[2] sum is", sum(a[2]), sum(h[2]))
    print("Loss is", model.loss(x, y))
    dL_dW, dL_db = model.backprop(x, y)
    model.updateParameters(dL_dW, dL_db, 1)
    print("New loss is", model.loss(x, y))
    
    # Test for PA 2
    model = feedforward(layers=[6,12,10,4])
    x = np.random.rand(6, 1)
    y = np.array([0,0,1,0])
    y.shape = (4,1)
    print("Initial loss is", model.loss(x, y))
    dL_dW, dL_db = model.backprop(x, y)
    model.updateParameters(dL_dW, dL_db, 1)
    print("New loss is", model.loss(x, y))
    _, h = model.forwardPass(x)
    print("Output is", h[-1])
    
if __name__ == "__main__":
    main()