import numpy as np
import matplotlib.pyplot as plt

def gradientDescent(X, y, batch, gradient, loss, eta = 0.1, kind = "vanilla", beta = 0.1, epochs = 100, printAt = 10):
    """
    Implement the gradient descent algorithm with different kinds of gradient descent algorithms
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data
    y : numpy.ndarray
        Output data
    batch : int
        Batch size
    gradient : function
        Gradient function
    loss : function
        Loss function
    eta : float, optional
        Step Size. The default is 0.1.
    kind : str, optional
        Can take values "momentum", "nesterov", or "vanilla". The default is "vanilla".
    beta : float, optional
        Momentum/Nesterov parameter. The default is 0.1.
    epochs : int, optional
        Number of Epochs. The default is 100.
    printAt : int, optional
        Print loss at every printAt epochs. The default is 10.
    
    Returns
    -------
    numpy.ndarray, numpy.ndarray
    """
    # Check if the kind is valid
    if kind not in ["momentum", "nesterov", "vanilla"]: raise ValueError("Invalid kind")
    # Check if X and y have the same number of rows
    if X.shape[0] != y.shape[0]: raise ValueError("X and y must have the same number of rows")
    # Check if the batch size is valid and an integer
    if type(batch) != int: raise ValueError("Batch size must be an integer")
    if batch <= 0: raise ValueError("Batch size must be greater than 0")
    # Check if the learning rate is valid
    if eta <= 0: raise ValueError("Learning rate must be greater than 0")
    # Check if the momentum parameter is valid
    if beta < 0 or beta > 1: raise ValueError("Momentum parameter must be between 0 and 1")
    # Check if the number of epochs is valid
    if epochs <= 0: raise ValueError("Number of epochs must be greater than 0")
    # Check if the printAt parameter is valid
    if printAt <= 0: raise ValueError("printAt parameter must be greater than 0")
    
    # Setup plt for plotting, we will plot the loss after each batch
    plt.figure(figsize=(10, 6))
    points = []
    # Initialize the weights and biases
    weights = np.random.rand(X.shape[1], y.shape[1])
    bias = np.random.rand(1, y.shape[1])
    # Initialize the velocity
    u = np.zeros((weights.shape[0]+1, weights.shape[1])) # 1 more row for bias
    # Loop through the epochs
    for i in range(epochs):
        # Print the loss
        if i % printAt == 0:
            print("Loss at epoch", i, "is", loss(X, y, weights, bias))
        # Initialize the gradients
        dL_dW = np.zeros(weights.shape)
        dL_db = np.zeros(bias.shape)
        # Loop through the batches
        for j in range(0, X.shape[0], batch):
            # Compute the gradient
            dL_dW, dL_db = gradient(X[j:j+batch], y[j:j+batch], weights, bias)
            # Update the weights and biases after each batch
            if kind == "momentum":
                u = beta * u +  np.concatenate((dL_dW, dL_db), axis = 0)
                weights -= eta * u[:-1,:]
                bias -= eta * u[-1,:]
            elif kind == "nesterov":
                u = beta * u + np.concatenate((gradient(X[j:j+batch], y[j:j+batch], weights - beta * u[:-1, :], bias - beta * u[-1, :])), axis = 0)
                weights -= eta * u[:-1,:]
                bias -= eta * u[-1,:]
            else: 
                weights -= eta * dL_dW
                bias -= eta * dL_db
            # Print the loss after each batch
            print("Loss after batch", j // batch, "is", loss(X[j:j+batch], y[j:j+batch], weights, bias))
            points.append(loss(X[j:j+batch], y[j:j+batch], weights, bias))
    plt.plot(points, color = "red")
    plt.scatter(range(len(points)), points, color = "blue")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title(f"{kind} Gradient Descent with batch size {batch}")
    plt.show()
    return weights, bias

def loss(X, y, weights, bias):
    """
    Loss function
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data
    y : numpy.ndarray
        Output data
    weights : numpy.ndarray
        Weights
    bias : numpy.ndarray
        Bias
        
    Returns
    -------
    float
    """
    return np.sum((y - np.dot(X, weights) - bias)**2) / X.shape[0]

def gradient(X, y, weights, bias):
    """
    Gradient function
    
    Parameters
    ----------
    X : numpy.ndarray
        Input data
    y : numpy.ndarray
        Output data
    weights : numpy.ndarray
        Weights
    bias : numpy.ndarray
        Bias
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray
    """
    dL_dW = -2 * np.dot(X.T, y - np.dot(X, weights) - bias) / X.shape[0]
    dL_db = -2 * np.sum(y - np.dot(X, weights) - bias) / X.shape[0]
    dL_db = np.array([dL_db]).reshape(1,1)
    return dL_dW, dL_db

def main():
    # Nesterov is very sensitive to the learning rate and the momentum parameter
    # Nesterov will explode if the learning rate/momentum is too high but will converge faster
    X = np.random.rand(100, 3)
    y = np.random.rand(100, 1)
    weights, bias = gradientDescent(X, y, 5, gradient, loss, eta = 0.01, kind = "nesterov", beta = 0.01, epochs = 100, printAt = 10)
    #print("Weights are", weights)
    #print("Bias is", bias)
    print("Loss is", loss(X, y, weights, bias))
    fval = 1 / (1 + np.exp(-(1.78*0.5) - 0))
    grad = (fval - 1)*fval*(1-fval)*0.5
    print("Gradient is", grad)


if __name__ == "__main__":
    main()