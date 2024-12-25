from _06_layers import artificialNeuronLayer
import numpy as np
import matplotlib.pyplot as plt

class adaptiveLearningRate():
    """
    Implement a neural network with adaptive learning rates using different methods.
    
    Attributes
    ----------
    layers : list
        List of artificialNeuronLayer objects representing the layers of the neural network.
    vt : list
        List of numpy.ndarray representing the vt for each layer.
    mt : list
        List of numpy.ndarray representing the mt for each layer.
    outputLayer : artificialNeuronLayer
        The output layer of the neural network.
    updateType : str
        The type of update to use.
    k : int
        The index of the vt and mt that is being used in the update method.
    t : int
        The time step.
    beta1 : float
        The beta1 parameter, also used as beta in some methods.
    beta2 : float
        The beta2 parameter.
    eta : float
        The learning rate.
    
    Methods
    -------
    forward(input)
        Forward pass through the neural network.
    backward(output_grad)
        Backward pass through the neural network.
    update()
        Update the weights and bias of the neural network.
    adaGrad(grad)
        AdaGrad update method.
    rmsProp(grad)
        RMSProp update method.
    adaDelta(grad)
        AdaDelta update method.
    adam(grad)
        Adam update method.
    maxProp(grad)
        MaxProp update method.
    adaMax(grad)
        AdaMax update method.
    Nadam(grad)
        Nadam update method.
    default(grad)
        Default update method, equivalent to vanilla gradient descent.
    """
    def __init__(self, input_size: int, hidden_size: list, output_size: int, updateType: str = 'default'):
        """
        Constructor for the adaptiveLearningRate class.

        Parameters
        ----------
        input_size : int
            The number of input neurons.
        hidden_size : list
            List of integers representing the number of neurons in each hidden layer.
        output_size : int
            The number of output neurons.
        updateType : str, optional
            The type of update to use. The default is 'default'. Can take values 'adaGrad', 'rmsProp', 'adaDelta', 'adam', 'maxProp', 'adaMax', 'Nadam', 'default'.
            
        Returns
        -------
        None
        """
        assert updateType in ['adaGrad', 'rmsProp', 'adaDelta', 'adam', 'maxProp', 'adaMax', 'Nadam', 'default']
        assert len(hidden_size) > 0
        self.layers = []
        self.vt = []
        self.mt = []
        for i in range(len(hidden_size)):
            if i == 0:
                self.layers.append(artificialNeuronLayer(input_size, hidden_size[i], 0))
                #print("Layer 0 weight and bias size: ", self.layers[0].weights.shape, self.layers[0].bias.shape)
                self.vt.append(np.zeros((input_size+1, hidden_size[i]))) # We add 1 to the input size to account for the bias
                self.mt.append(np.zeros((input_size+1, hidden_size[i])))
            else:
                self.layers.append(artificialNeuronLayer(hidden_size[i - 1], hidden_size[i], 0))
                #print("Layer ", i, " weight and bias size: ", self.layers[i].weights.shape, self.layers[i].bias.shape)
                self.vt.append(np.zeros((hidden_size[i - 1]+1, hidden_size[i])))
                self.mt.append(np.zeros((hidden_size[i - 1]+1, hidden_size[i])))
        self.outputLayer = artificialNeuronLayer(hidden_size[-1], output_size, 0)
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
        """
        Forward pass through the neural network.

        Parameters
        ----------
        input : float
            The input to the neural network.

        Returns
        -------
        float
        """
        x = input
        for layer in self.layers:
            x = layer(x)
            x = np.maximum(0, x) # ReLU activation function
        x = self.outputLayer(x)
        x = x # no activation function for the output layer
        return x
    
    def backward(self, output_grad: float) -> float:
        """
        Backward pass through the neural network.

        Parameters
        ----------
        output_grad : float
            The gradient of the output.

        Returns
        -------
        float
        """
        output_grad = self.outputLayer.backward(output_grad)
        for layer in reversed(self.layers):
            output_grad *= (output_grad > 0) # ReLU derivative
            output_grad = layer.backward(output_grad)
        return output_grad
    
    def update(self):
        """
        Update the weights and bias of the neural network.
        
        Returns
        -------
        None
        """
        method = getattr(self, self.updateType) # Get the method to use
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
        # vt = vt + grad**2
        self.vt[self.k] += grad**2
        # dw = eta / sqrt(vt + 1e-8) * grad
        return self.eta / np.sqrt(self.vt[self.k] + 1e-8) * grad
    
    def rmsProp(self, grad):
        # vt = beta1 * vt + (1 - beta1) * grad**2
        self.vt[self.k] = self.beta1 * self.vt[self.k] + (1 - self.beta1) * grad**2
        # dw = eta / sqrt(vt + 1e-8) * grad
        return (self.eta / np.sqrt(self.vt[self.k] + 1e-8)) * grad
    
    def adaDelta(self, grad):
        # vt = beta1 * vt + (1 - beta1) * grad**2
        self.vt[self.k] = self.beta1 * self.vt[self.k] + (1 - self.beta1) * grad**2
        # dw = sqrt(mt + 1e-8) / sqrt(vt + 1e-8) * grad
        dw = np.sqrt(self.mt[self.k] + 1e-8) / np.sqrt(self.vt[self.k] + 1e-8) * grad
        # mt = beta1 * mt + (1 - beta1) * dw**2
        self.mt[self.k] = self.beta1 * self.mt[self.k] + (1 - self.beta1) * dw**2
        return dw
    
    def adam(self, grad):
        # mt = beta1 * mt + (1 - beta1) * grad
        self.mt[self.k] = self.beta1 * self.mt[self.k] + (1 - self.beta1) * grad
        # m_hat = mt / (1 - beta1**t)
        m_hat = self.mt[self.k] / (1 - self.beta1**self.t)
        # vt = beta2 * vt + (1 - beta2) * grad**2
        self.vt[self.k] = self.beta2 * self.vt[self.k] + (1 - self.beta2) * grad**2
        # v_hat = vt / (1 - beta2**t)
        v_hat = self.vt[self.k] / (1 - self.beta2**self.t)
        # dw = eta / (sqrt(v_hat) + 1e-8) * m_hat
        return (self.eta / (np.sqrt(v_hat) + 1e-8)) * m_hat
    
    def maxProp(self, grad):
        # vt = max(beta1 * vt, |grad|)
        self.vt[self.k] = np.maximum(self.beta1 * self.vt[self.k], abs(grad))
        # dw = eta / (vt + 1e-8) * grad
        return (self.eta / (self.vt[self.k] + 1e-8)) * grad
        
    def adaMax(self, grad):
        # mt = beta1 * mt + (1 - beta1) * grad
        self.mt[self.k] = self.beta1 * self.mt[self.k] + (1 - self.beta1) * grad
        # m_hat = mt / (1 - beta1**t)
        m_hat = self.mt[self.k] / (1 - self.beta1**self.t)
        # vt = max(beta2 * vt, |grad|)
        self.vt[self.k] = np.maximum(self.beta2 * self.vt[self.k], abs(grad))
        # dw = eta / (vt + 1e-8) * m_hat
        return (self.eta / (self.vt[self.k] + 1e-8)) * m_hat
    
    def Nadam(self, grad):
        # mt = beta1 * mt + (1 - beta1) * grad
        self.mt[self.k] = self.beta1 * self.mt[self.k] + (1 - self.beta1) * grad
        # m_hat = mt / (1 - beta1**t)
        m_hat = self.mt[self.k] / (1 - self.beta1**self.t)
        # vt = beta2 * vt + (1 - beta2) * grad**2
        self.vt[self.k] = self.beta2 * self.vt[self.k] + (1 - self.beta2) * grad**2
        # v_hat = vt / (1 - beta2**t)
        v_hat = self.vt[self.k] / (1 - self.beta2**self.t)
        # dw = (eta / sqrt(v_hat) + 1e-8) * (m_hat + ((1 - beta1) * grad / (1 - beta1**t)))
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
