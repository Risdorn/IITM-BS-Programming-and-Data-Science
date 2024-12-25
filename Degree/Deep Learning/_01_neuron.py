import numpy as np

class mpNeuron():
    """
    A class that implements the McCulloch Pitts Neuron model.
    
    Attributes
    ----------
    threshold : int
        The threshold for the model.
    inhibitor : list
        The list of inhibitor indices.
    
    Methods
    -------
    model(x):
        Returns the output of the model for the given input.
    predict(X):
        Returns the output of the model for the given list of inputs.
    """
    def __init__(self, threshold=0, inhibitor=[]):
        """
        Constructs all the necessary attributes for the MP Neuron model.

        Parameters
        ----------
        threshold : int, optional
            The threshold for the model. The default is 0.
        inhibitor : list, optional
            The list of inhibitor indices. The default is [].
        
        Returns
        -------
        None
        """
        self.threshold = threshold
        self.inhibitor = inhibitor
        
    def model(self, x):
        """
        Returns the output of the model for the given input.

        Parameters
        ----------
        x : list
            The input vector.
        
        Returns
        -------
        int
        """
        for i in self.inhibitor:
            if x[i] == 1:
                return 0
        return int(sum(x) >= self.threshold)
    
    def predict(self, X):
        """
        Returns the output of the model for the given list of inputs.

        Parameters
        ----------
        X : list
            The list of input vectors.

        Returns
        -------
        list
        """
        Y = []
        for x in X:
            Y.append(self.model(x))
        return Y

class perceptron():
    """
    A class that implements the Perceptron model.
    
    Attributes
    ----------
    weights : list
        The list of weights for the model.
    bias : int
        The bias for the model.
    
    Methods
    -------
    model(x):
        Returns the output of the model for the given input.
    predict(X):
        Returns the output of the model for the given list of inputs.
    fit(X, Y):
        Fits the model to the given input and output.
    """
    def __init__(self, weights=[], bias=0):
        """
        Constructs all the necessary attributes for the Perceptron model.

        Parameters
        ----------
        weights : list, optional
            The list of weights for the model. The default is [].
        bias : int, optional
            The bias for the model. The default is 0.
        
        Returns
        -------
        None
        """
        self.weights = weights
        self.bias = bias
    
    def model(self, x):
        """
        Returns the output of the model for the given input.

        Parameters
        ----------
        x : list
            The input vector.

        Returns
        -------
        int
        """
        return int(sum([x[i]*self.weights[i] for i in range(len(x))]) + self.bias >= 0)
    
    def predict(self, X):
        """
        Returns the output of the model for the given list of inputs.

        Parameters
        ----------
        X : list
            The list of input vectors.

        Returns
        -------
        list
        """
        Y = []
        for x in X:
            Y.append(self.model(x))
        return Y
    
    def fit(self, X, Y, max_epochs=100):
        """
        Fits the model to the given input and output, using the perceptron learning algorithm.

        Parameters
        ----------
        X : list
            The list of input vectors.
        Y : list
            The list of output values.
        max_epochs : int, optional
            The maximum number of epochs to run. The default is 100.
        
        Returns
        -------
        None
        """
        Y_hat = self.predict(X) # Initial prediction
        for _ in range(max_epochs):
            if Y == Y_hat: break # If the prediction is correct, break
            for j in range(len(Y)):
                if Y[j] == Y_hat[j]: continue # We only need to update the weights if the prediction is wrong
                # Update the weights and bias
                for i in range(len(self.weights)):
                    self.weights[i] += (Y[j] - Y_hat[j]) * X[j][i]
                self.bias += (Y[j] - Y_hat[j])
            Y_hat = self.predict(X) # Update the prediction

class networkPerceptron():
    """
    A class that implements a network of perceptrons.
    
    Hidden layer perceptrons are fixed and are not trainable.
    
    Output layer perceptron is trainable.
    
    This class is dependent on the perceptron class, defined above.
    
    Attributes
    ----------
    inputs : int
        The number of inputs to the network.
    permutations : list
        The list of input permutations. corresponds to the input layer weights.
    inputLayer : list
        The list of input layer perceptrons.
    outputPerceptron : perceptron
        The output layer perceptron.
    
    Methods
    -------
    generatePermutations(inputs):
        Generates the list of input permutations, corresponding to the input layer weights.
    makeInputLayer(inputs):
        Makes the input layer perceptrons.
    inputLayerOutput(X):
        Returns the output of the input layer for the given input.
    model(x):
        Returns the output of the model for the given input.
    predict(X):
        Returns the output of the model for the given list of inputs.
    fit(X, Y):
        Fits the model to the given input and output, using the perceptron learning algorithm.
    """
    def __init__(self, inputs = 2, weights=[], bias=0):
        """
        Constructs all the necessary attributes for the Network Perceptron model.
        
        If weights and bias are not provided, they are randomly initialized.

        Parameters
        ----------
        inputs : int, optional
            The number of inputs to the network. The default is 2.
        weights : list, optional
            The list of weights for the output layer perceptron. The default is [].
        bias : int, optional
            The bias for the output layer perceptron. The default is 0.
        
        Returns
        -------
        None
        """
        self.inputs = inputs
        self.generatePermutations(inputs)
        self.makeInputLayer(inputs)
        if weights == []: weights = np.random.rand(2**inputs)
        if bias == 0: bias = np.random.rand()
        self.outputPerceptron = perceptron(weights=weights, bias=bias) # This is the only trainable perceptron
    
    def generatePermutations(self, inputs):
        """
        Generates the list of input permutations, corresponding to the input layer weights.
        
        Should only be called by the constructor.

        Parameters
        ----------
        inputs : int
            The number of inputs to the network.
            
        Returns
        -------
        None
        """
        self.permutations = []
        for i in range(2**inputs):
            # bin(i) returns a string of the binary representation of i, format is 0bxxxx
            # We take the substring from the 2nd character to the end and fill it with 0s to make it of length inputs
            self.permutations.append([int(x) for x in bin(i)[2:].zfill(inputs)])
        for i in range(2**inputs):
            self.permutations[i] = [self.permutations[i][j] * 2 - 1 for j in range(inputs)] # Convert to range -1 to 1
    
    def makeInputLayer(self, inputs):
        """
        Makes the input layer perceptrons, will call the perceptron class.
        
        Should only be called by the constructor.

        Parameters
        ----------
        inputs : int
            The number of inputs to the network.
            
        Returns
        -------
        None
        """
        self.inputLayer = []
        for i in range(2**inputs):
            self.inputLayer.append(perceptron(weights=self.permutations[i], bias=-inputs))
    
    def inputLayerOutput(self, X):
        """
        Returns the output of the input layer for the given input.

        Parameters
        ----------
        X : list
            The input vector.
            
        Returns
        -------
        list
        """
        result = [self.inputLayer[i].model(X) for i in range(2**self.inputs)]
        result = [x * 2 - 1 for x in result]
        return result
    
    def model(self, x):
        """
        Returns the output of the model for the given input.

        Parameters
        ----------
        x : list
            The input vector.

        Returns
        -------
        int
        """
        # Pass the input through the input layer and then through the output perceptron
        result = self.outputPerceptron.model(self.inputLayerOutput(x))
        result = 1 if result == 1 else -1 # Because perceptron output 1 or 0
        return result
    
    def predict(self, X):
        """
        Returns the output of the model for the given list of inputs.

        Parameters
        ----------
        X : list
            The list of input vectors.

        Returns
        -------
        list
        """
        Y = []
        for x in X:
            Y.append(self.model(x))
        return Y
    
    def fit(self, X, Y, max_epochs=100):
        """
        Fits the model to the given input and output, using the perceptron learning algorithm.

        Parameters
        ----------
        X : list
            The list of input vectors.
        Y : list
            The list of output values.
        max_epochs : int, optional
            The maximum number of epochs to run. The default is 100.
            
        Returns
        -------
        None
        """
        Y_hat = self.predict(X) # Initial prediction
        for _ in range(max_epochs):
            if Y == Y_hat: break # If the prediction is correct, break
            for j in range(len(Y)):
                if Y[j] == Y_hat[j]: continue # We only need to update the weights if the prediction is wrong
                # Update the weights and bias
                for i in range(len(self.outputPerceptron.weights)):
                    self.outputPerceptron.weights[i] += (Y[j] - Y_hat[j]) * self.inputLayerOutput(X[j])[i]
                self.outputPerceptron.bias += (Y[j] - Y_hat[j])
            Y_hat = self.predict(X) # Update the prediction

class sigmoidNeuron():
    """
    A class that implements the Sigmoid Neuron model.
    
    Attributes
    ----------
    inputs : int
        The number of inputs to the model.
    weights : list
        The list of weights for the model.
    bias : int
        The bias for the model.
        
    Methods
    -------
    sigmoidDerivative(x, y):
        Returns the derivative of the model for the given input and output.
    model(x):
        Returns the output of the model for the given input.
    predict(X):
        Returns the output of the model for the given list of inputs.
    loss(X, Y):
        Returns the loss of the model for the given list of inputs and outputs.
    fit(X, Y):
        Fits the model to the given input and output, using the sigmoid learning algorithm.
    """
    def __init__(self, inputs=2, weights = None, bias = None):
        """
        Constructs all the necessary attributes for the Sigmoid Neuron model.
        
        If weights and bias are not provided, they are randomly initialized.

        Parameters
        ----------
        inputs : int, optional
            The number of inputs to the model. The default is 2.
        weights : list, optional
            The list of weights for the model. The default is None.
        bias : int, optional
            The bias for the model. The default is None.
            
        Returns
        -------
        None
        """
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(inputs)
            self.weights = self.weights * 2 - 1 # Random weights between -1 and 1
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.rand()
            self.bias = self.bias * 2 - 1 # Random bias between -1 and 1
    
    def sigmoidDerivative(self, x, y):
        """
        Returns the derivative of the model for the given input and output.

        Parameters
        ----------
        x : list
            The input vector.
        y : int
            The output value.

        Returns
        -------
        list, int
        """
        constant = (self.model(x) - y) * self.model(x) * (1 - self.model(x)) # Derivative of sigmoid, (y_hat - y) * sigmoid * (1 - sigmoid)
        weightDerivative = [constant * x[i] for i in range(len(x))]
        biasDerivative = constant
        return weightDerivative, biasDerivative
    
    def model(self, x):
        """
        Returns the output of the model for the given input.

        Parameters
        ----------
        x : list
            The input vector.

        Returns
        -------
        int
        """
        result = sum([x[i]*self.weights[i] for i in range(len(x))]) + self.bias
        result = 1 / (1 + np.exp(-result))
        return result
    
    def predict(self, X):
        """
        Returns the output of the model for the given list of inputs.

        Parameters
        ----------
        X : list
            The list of input vectors.

        Returns
        -------
        list
        """
        Y = []
        for x in X:
            Y.append(self.model(x))
        return Y
    
    def loss(self, X, Y):
        """
        Returns the loss of the model for the given list of inputs and outputs.

        Parameters
        ----------
        X : list
            The list of input vectors.
        Y : list
            The list of output values

        Returns
        -------
        float
        """
        loss = 0
        for i in range(len(Y)):
            loss += (self.model(X[i]) - Y[i])**2
        return loss / len(Y)
    
    def fit(self, X, Y, epochs=100, lr=0.1, printAt=10):
        """
        Fits the model to the given input and output, using the gradient descent algorithm.

        Parameters
        ----------
        X : list
            The list of input vectors.
        Y : list
            The list of output values.
        epochs : int, optional
            The number of epochs to run. The default is 100.
        lr : float, optional
            The learning rate. The default is 0.1.
        printAt : int, optional
            The number of epochs after which to print the loss. The default is 10.
        
        Returns
        -------
        None
        """
        for i in range(epochs):
            loss = self.loss(X, Y) # Calculate the loss
            if i%printAt == 0: print("Epoch:", i, "Loss:", loss)
            # Update the weights and bias
            for j in range(len(Y)):
                weightD, biasD = self.sigmoidDerivative(X[j], Y[j])
                self.weights -= np.multiply(lr, weightD)
                self.bias -= np.multiply(lr, biasD)

# The following functions are for testing the models, feel free to play around with them

def mpNeuronTest():
    """
    MP neuron model test.
    AND gate = threshold 3, no inhibitor
    OR gate = threshold 1, no inhibitor
    NOT gate = threshold 0, inhibitor 0
    NOR gate = threshold 0, inhibitor 0, 1, 2
    """
    X = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    # The following commented code is for NOT gate as it only has 1 input
    #X = [[0], [1]]
    #neuron = mpNeuron(threshold=0, inhibitor=[0])
    neuron = mpNeuron(threshold=3) # AND gate
    Y = neuron.predict(X)
    for i in range(len(X)):
        print(X[i], Y[i])

def perceptronTest():
    """
    Perceptron model test.
    AND, OR, NOR, and NOT gates fit as expected.
    XOR gate fit does not converge as expected.
    """
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    #X = [[0], [1]] # Uncomment this for NOT gate, and also need to change the output list
    classical = perceptron(weights=[1, 1], bias=0)
    classical.fit(X, [0, 0, 0, 1]) # Change the output list to get different gates
    print("Weights:", classical.weights, "Bias:", classical.bias)
    Y = classical.predict(X)
    for i in range(len(X)):
        print(X[i], Y[i])

def networkTest():
    """
    Network perceptron model test.
    Capable of implementing any boolean function.
    """
    network = networkPerceptron(inputs=2)
    X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    Y = [1, -1, -1, 1] # Change the output list to get different gates
    network.fit(X, Y)
    # Visualize the weights and bias, after training
    for i in range(2**network.inputs):
        print("Input weights:", network.inputLayer[i].weights, "Input bias:", network.inputLayer[i].bias)
    print("Output weights:", network.outputPerceptron.weights, "Output bias:", network.outputPerceptron.bias)
    Yhat = network.predict(X) # Predict the output
    for i in range(len(X)):
        print(X[i], Yhat[i])

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

if __name__ == "__main__":
    main()