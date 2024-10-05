import numpy as np

class mpNeuron():
    """
    This class implements the McCulloch-Pitts Neuron model.
    The model is defined by a threshold and a list of inhibitor indices.
    The model returns 1 if the sum of the input is greater than or equal to the threshold and none of the inhibitor indices are 1.
    Functions:
    - model(x): returns the output of the model for a given input x.
    - predict(X): returns the output of the model for a list of inputs X.
    """
    def __init__(self, threshold=0, inhibitor=[]):
        self.threshold = threshold
        self.inhibitor = inhibitor
        
    def model(self, x):
        for i in self.inhibitor:
            if x[i] == 1:
                return 0
        return int(sum(x) >= self.threshold)
    
    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.model(x))
        return Y

class perceptron():
    """
    This class implements the Perceptron model.
    Along with the learning algorithm.
    """
    def __init__(self, weights=[], bias=0):
        self.weights = weights
        self.bias = bias
    
    def model(self, x):
        return int(sum([x[i]*self.weights[i] for i in range(len(x))]) + self.bias >= 0)
    
    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.model(x))
        return Y
    
    def fit(self, X, Y):
        Y_hat = self.predict(X)
        while Y_hat != Y:
            for j in range(len(Y)):
                if Y[j] == Y_hat[j]: continue
                for i in range(len(self.weights)):
                    self.weights[i] += (Y[j] - Y_hat[j]) * X[j][i]
                    self.bias += (Y[j] - Y_hat[j])
            Y_hat = self.predict(X)

class networkPerceptron():
    """
    This class implements a network of perceptrons.
    Input layer is a list of perceptrons, each perceptron representing a different input. The weights and bias are constant.
    Output layer is a single perceptron.
    """
    def __init__(self, inputs = 2, weights=[], bias=0):
        self.inputs = inputs
        self.generatePermutations(inputs)
        self.makeInputLayer(inputs)
        self.outputPerceptron = perceptron(weights=weights, bias=bias)
    
    def generatePermutations(self, inputs):
        self.permutations = []
        for i in range(2**inputs):
            self.permutations.append([int(x) for x in bin(i)[2:].zfill(inputs)])
        for i in range(2**inputs):
            self.permutations[i] = [self.permutations[i][j] * 2 - 1 for j in range(inputs)]
    
    def makeInputLayer(self, inputs):
        self.inputLayer = []
        for i in range(2**inputs):
            self.inputLayer.append(perceptron(weights=self.permutations[i], bias=-inputs))
    
    def inputLayerOutput(self, X):
        result = [self.inputLayer[i].model(X) for i in range(2**self.inputs)]
        result = [x * 2 - 1 for x in result]
        return result
    
    def model(self, x):
        result = self.outputPerceptron.model(self.inputLayerOutput(x))
        result = 1 if result == 1 else -1
        return result
    
    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.model(x))
        return Y

class sigmoidNeuron():
    """
    This class implements a sigmoid neuron model.
    The model is defined by a list of weights and a bias.
    Uses gradient descent to fit the model.
    """
    def __init__(self, inputs=2, weights = None, bias = None):
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(inputs)
            self.weights = self.weights * 2 - 1
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.rand()
            self.bias = self.bias * 2 - 1
    
    def model(self, x):
        result = sum([x[i]*self.weights[i] for i in range(len(x))]) + self.bias
        result = 1 / (1 + np.exp(-result))
        return result
    
    def sigmoidDerivative(self, x, y):
        constant = (self.model(x) - y) * self.model(x) * (1 - self.model(x))
        weightDerivative = [constant * x[i] for i in range(len(x))]
        biasDerivative = constant
        return weightDerivative, biasDerivative
    
    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.model(x))
        return Y
    
    def loss(self, X, Y):
        loss = 0
        for i in range(len(Y)):
            loss += (self.model(X[i]) - Y[i])**2
        return loss / len(Y)
    
    def fit(self, X, Y, epochs=100, lr=0.1, printAt=10):
        for i in range(epochs):
            print(self.weights, self.bias)
            loss = self.loss(X, Y)
            if i%printAt == 0: print("Epoch:", i, "Loss:", loss)
            print(self.predict(X))
            for j in range(len(Y)):
                weightD, biasD = self.sigmoidDerivative(X[j], Y[j])
                self.weights -= np.multiply(lr, weightD)
                self.bias -= np.multiply(lr, biasD)
        print(self.weights, self.bias)


def mpNeuronTest():
    """
    MP neuron model test.
    AND gate = threshold 3, no inhibitor
    OR gate = threshold 1, no inhibitor
    NOT gate = threshold 0, inhibitor 0
    NOR gate = threshold 0, inhibitor 0, 1, 2
    """
    X = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    notX = [[0], [1]]
    neuron = mpNeuron(threshold=0, inhibitor=[0])
    Y = neuron.predict(notX)
    for i in range(len(notX)):
        print(notX[i], Y[i])

def perceptronTest():
    """
    Perceptron model test.
    AND, OR, NOR, and NOT gates fit as expected.
    XOR gate fit does not converge as expected.
    """
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    #notX = [[0], [1]]
    classical = perceptron(weights=[1, 1], bias=0)
    classical.fit(X, [1, 0, 0, 1])
    print("Weights:", classical.weights, "Bias:", classical.bias)
    Y = classical.predict(X)
    for i in range(len(X)):
        print(X[i], Y[i])

def networkTest():
    """
    Network perceptron model test.
    Fit algorithm does not work, something needs to be changed.
    But, it is still capable of implementing any boolean function.
    Below is the XOR gate.
    """
    network = networkPerceptron(inputs=2, weights=[-1, 0, 0, -1], bias=-1)
    X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    Y = network.predict(X)
    for i in range(2**network.inputs):
        print("Input weights:", network.inputLayer[i].weights, "Input bias:", network.inputLayer[i].bias)
    print("Output weights:", network.outputPerceptron.weights, "Output bias:", network.outputPerceptron.bias)
    for i in range(len(X)):
        print(X[i], Y[i])

def sigmoidTest():
    """
    Sigmoid neuron model test.
    Linearly separable functions seem to work very well.
    But it gives about 50% probability for all inputs in case of non linearly separable functions.
    This is very interesting.
    """
    '''
    Uncomment this for testing
    X = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    Y = [1, 1, 1, 0, 1, 1, 0, 1]
    sigmoid = sigmoidNeuron(inputs=3)
    sigmoid.fit(X, Y, epochs=1000, lr = 0.05, printAt=100)
    for i in range(len(X)):
        print(X[i], sigmoid.model(X[i]))
    '''
    # This is assignment 2 test
    X = [[-1], [0.2]]
    Y = [0.5, 0.97]
    weights = [[[2], [1.9], [1.19], [0.39]], [[2], [2.008], [2.19], [2.39]], [[2], [2.008], [2.19], [2.39]], [[2], [-2.008], [-2.19], [-2.39]]]
    bias = [[2, 2.04,2.20,2.31], [2, 2.04,2.20,2.31], [2, 1.9,1.19,0.39], [2, 1.9,1.19,0.39]]
    for i in range(4):
        print("Run", i+1)
        for j in range(4):
            sigmoid = sigmoidNeuron(inputs=1, weights=weights[i][j], bias=bias[i][j])
            loss = sigmoid.loss(X, Y)
            print("Loss:", j, loss)

def main():
    #mpNeuronTest()
    #perceptronTest()
    #networkTest()
    sigmoidTest()
    

if __name__ == '__main__':
    main()