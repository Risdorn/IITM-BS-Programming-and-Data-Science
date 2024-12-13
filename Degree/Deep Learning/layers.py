from typing import Protocol
import numpy as np

class defaultLayer(Protocol):
    def forward(self, input: float) -> float:
        pass

    def backward(self, output_grad: float) -> float:
        pass

    def update(self, lr: float):
        pass

    def __call__(self, input: float) -> float:
        return self.forward(input)

class artificialNeuronLayer(defaultLayer):
    def __init__(self, input_size: int, output_size: int):
        np.random.seed(0)
        self.weights = np.random.randn(input_size, output_size)* np.sqrt(2. / input_size)
        self.bias = np.random.randn(1, output_size)* np.sqrt(2. / input_size)
        self.input = None
        self.output = None
        self.dL_dW = None
        self.dL_db = None

    def forward(self, input: float) -> float:
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_grad: float) -> float:
        self.dL_dW = np.dot(self.input.T, output_grad)
        self.dL_db = np.sum(output_grad, axis = 0)
        return np.dot(output_grad, self.weights.T)

    def update(self, lr: float):
        self.weights -= lr * self.dL_dW
        self.bias -= lr * self.dL_db
        self.dL_dW = None
        self.dL_db = None

class convolutionLayer(defaultLayer):
    def __init__(self, kernel_size: int, depth: int, out_depth: int, padding: int = 0, stride: int = 1):
        np.random.seed(0)
        self.weights = np.random.randn(kernel_size, kernel_size, depth, out_depth)* np.sqrt(2. / depth)
        self.bias = np.random.randn(1, out_depth) * np.sqrt(2. / depth)
        self.padding = padding
        self.stride = stride
        self.input = None
        self.output = None
        self.dL_dW = np.zeros(self.weights.shape)
        self.dL_db = np.zeros(self.bias.shape)
    
    def forward(self, input: float) -> float:
        self.input = input
        input = np.pad(input, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode = 'constant')
        self.output = np.zeros((int((input.shape[0] - self.weights.shape[0] + 2 * self.padding) / self.stride) + 1, 
                                int((input.shape[1] - self.weights.shape[1] + 2 * self.padding) / self.stride) + 1, 
                                self.weights.shape[3]))
        #print(self.output.shape)
        for i in range(0,self.output.shape[0], self.stride):
            for j in range(0,self.output.shape[1], self.stride):
                for k in range(self.weights.shape[3]):
                    self.output[i, j, k] = np.sum(input[i:i+self.weights.shape[0], j:j+self.weights.shape[1], :] * self.weights[:,:,:,k]) + self.bias[0,k]
        return self.output
    
    def backward(self, output_grad: float) -> float:
        input_grad = np.zeros(self.input.shape)
        input = np.pad(self.input, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode = 'constant')
        
        # For input_grad, output_grad is convoluted with weights
        flipped_weights = np.flip(self.weights, (0, 1)) # Flipping the weights for convolution
        #print(flipped_weights.shape)
        for i in range(0,output_grad.shape[0], self.stride):
            for j in range(0,output_grad.shape[1], self.stride):
                val = output_grad[i, j, :]
                for k in range(flipped_weights.shape[2]):
                    #print(val.shape, self.weights[:,:,k,:].shape)
                    input_grad[i:i+flipped_weights.shape[0], j:j+flipped_weights.shape[1], k] += np.sum(val * flipped_weights[:,:,k,:], axis=-1)
        
        # For dL_dW and dL_db, input is convoluted with output_grad
        for i in range(0,self.output.shape[0], self.stride):
            for j in range(0,self.output.shape[1], self.stride):
                for k1 in range(self.weights.shape[2]):
                    for k2 in range(self.weights.shape[3]):
                        self.dL_dW[:,:,k1,k2] += input[i:i+self.weights.shape[0], j:j+self.weights.shape[1], k1] * output_grad[i, j, k2]
                    self.dL_db[0,k1] += np.sum(output_grad[i, j, k1], axis = 0)
        return input_grad
    
    def update(self, lr: float):
        self.weights -= lr * self.dL_dW
        self.bias -= lr * self.dL_db
        self.dL_dW = np.zeros(self.weights.shape)
        self.dL_db = np.zeros(self.bias.shape)

class maxPoolingLayer(defaultLayer):
    def __init__(self, kernel_size: int, stride: int):
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.output = None
        self.max_indices = None
    
    def forward(self, input: float) -> float:
        self.input = input
        self.output = np.zeros((input.shape[0] // self.stride, input.shape[1] // self.stride, input.shape[2]))
        self.max_indices = np.zeros((input.shape[0] // self.stride, input.shape[1] // self.stride, input.shape[2]), dtype = int)
        
        # For each kernel, find the maximum value and its index
        for i in range(0,self.output.shape[0], self.stride):
            for j in range(0,self.output.shape[1], self.stride):
                for k in range(input.shape[2]):
                    self.output[i, j, k] = np.max(input[i:i+self.kernel_size, j:j+self.kernel_size, k])
                    self.max_indices[i, j, k] = np.argmax(input[i:i+self.kernel_size, j:j+self.kernel_size, k])
        return self.output
    
    def backward(self, output_grad: float) -> float:
        input_grad = np.zeros(self.input.shape)
        for i in range(0,self.output.shape[0], self.stride):
            for j in range(0,self.output.shape[1], self.stride):
                for k in range(self.input.shape[2]):
                    max_index = self.max_indices[i, j, k]
                    
                    max_row = (i * self.stride) + (max_index // self.kernel_size)
                    max_col = (j * self.stride) + (max_index % self.kernel_size)
                    
                    input_grad[max_row, max_col, k] += output_grad[i, j, k]
        return input_grad

class rnnLayer(defaultLayer):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        np.random.seed(0)
        # U, c
        self.input_weights = np.random.randn(hidden_size, input_size)* np.sqrt(2. / input_size)
        self.input_bias = np.random.randn(hidden_size, 1)* np.sqrt(2. / input_size)
        # W
        self.hidden_weights = np.random.randn(hidden_size, hidden_size)* np.sqrt(2. / input_size)
        # V, b
        self.output_weights = np.random.randn(output_size, hidden_size)* np.sqrt(2. / input_size)
        self.output_bias = np.random.randn(output_size, 1)* np.sqrt(2. / input_size)
        self.inputs = []
        self.outputs = []
        self.hidden = [np.zeros((hidden_size, 1))]
        self.dL_dV = np.zeros(self.output_weights.shape)
        self.dL_db = np.zeros(self.output_bias.shape)
        self.dL_dW = np.zeros(self.hidden_weights.shape)
        self.dL_dU = np.zeros(self.input_weights.shape)
        self.dL_dc = np.zeros(self.input_bias.shape)
        self.dL_dh_next = np.zeros((hidden_size, 1))
    
    def forward_step(self, input):
        input = input.reshape(-1, 1)
        #print(input.shape, self.input_weights.shape)
        #print(np.dot(self.input_weights, input).shape, self.input_bias.shape)
        h = np.dot(self.input_weights, input) + self.input_bias # Ux + c
        #print(h.shape)
        h += np.dot(self.hidden_weights, self.hidden[-1]) # Ws
        #print(h.shape)
        self.hidden.append(np.tanh(h)) # s = tanh(Ux + Ws + c)
        #print("hidden state s: ", np.sum(self.hidden[-1]))
        out = np.dot(self.output_weights, self.hidden[-1]) + self.output_bias # Vs + b
        #print(out.shape)
        softmax = np.exp(out) / np.sum(np.exp(out)) # softmax(Vs + b)
        self.outputs.append(softmax)
        result = np.zeros(softmax.shape)
        #print(result)
        result[np.argmax(softmax)] = 1
        return result
    
    def forward(self, input, time_steps):
        # This is in auto-regressive mode
        self.inputs = []
        self.outputs = []
        #self.hidden = [np.zeros(self.hidden[0].shape)]
        self.inputs.append(input)
        #self.inputs = inputs
        for time in range(time_steps):
            input = self.forward_step(input) # Output from the forward step is input to the next step
            self.inputs.append(input)
        self.inputs.pop(-1) # We remove the last input as it is not needed
        return self.outputs # We return softmax outputs
    
    def forward_constant(self, inputs, time_steps):
        self.inputs = inputs
        self.outputs = []
        #self.hidden = [np.zeros(self.hidden[0].shape)]
        for time in range(time_steps):
            self.forward_step(self.inputs[time])
        return self.outputs
    
    def backward_step(self, output_grad, time):
        # Gradient for the softmax layer
        dL_do = output_grad[time]
        
        # Gradient for output weights and bias, V and b
        self.dL_dV += np.dot(dL_do, self.hidden[time+1].T)
        self.dL_db += dL_do
        
        # Gradient w.r.t hidden state, s
        dL_ds = np.dot(self.output_weights.T, dL_do) + self.dL_dh_next
        
        # Gradient through tanh
        dL_dh = (1 - self.hidden[time+1] ** 2) * dL_ds
        
        # Gradient for hidden weights, W
        self.dL_dW += np.dot(dL_dh, self.hidden[time].T)
        
        # Gradient for input weights and bias, U and c
        self.dL_dU += np.dot(dL_dh, self.inputs[time].T)
        self.dL_dc += dL_dh
        
        # Gradient for the previous hidden state
        self.dL_dh_next = np.dot(self.hidden_weights.T, dL_dh)
        
        # Gradient for the input
        return np.dot(self.input_weights.T, dL_dh)
    
    def backward(self, output_grad):
        time_steps = len(self.inputs)
        #print(len(self.hidden)) # this should be equal to time_steps + 1
        input_gradients = []
        for time in reversed(range(time_steps)):
            input_gradient = self.backward_step(output_grad, time)
            input_gradients.append(input_gradient)
        return input_gradients # We return the gradients for the input
    
    def update(self, lr: float):
        # Update the weights and biases
        self.input_weights -= lr * self.dL_dU
        self.input_bias -= lr * self.dL_dc
        self.hidden_weights -= lr * self.dL_dW
        self.output_weights -= lr * self.dL_dV
        self.output_bias -= lr * self.dL_db
        # Reset the gradients
        self.dL_dV = np.zeros(self.output_weights.shape)
        self.dL_db = np.zeros(self.output_bias.shape)
        self.dL_dW = np.zeros(self.hidden_weights.shape)
        self.dL_dU = np.zeros(self.input_weights.shape)
        self.dL_dc = np.zeros(self.input_bias.shape)
        self.dL_dh_next = np.zeros(self.hidden[0].shape)
        self.dL_dh = None
        self.inputs = []
        self.outputs = []
        self.hidden = [np.zeros(self.hidden[0].shape)]

class rnnEncoder(defaultLayer):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        np.random.seed(0)
        # U, c
        self.input_weights = np.random.randn(hidden_size, input_size)* np.sqrt(2. / input_size)
        self.input_bias = np.random.randn(hidden_size, 1)* np.sqrt(2. / input_size)
        # W
        self.hidden_weights = np.random.randn(hidden_size, hidden_size)* np.sqrt(2. / input_size)
        self.inputs = []
        self.hidden = [np.zeros((hidden_size, 1))]
        self.dL_dW = np.zeros(self.hidden_weights.shape)
        self.dL_dU = np.zeros(self.input_weights.shape)
        self.dL_dc = np.zeros(self.input_bias.shape)
        self.dL_dh_next = np.zeros((hidden_size, 1))
    
    def forward_step(self, input):
        input = input.reshape(-1, 1)
        #print(input.shape, self.input_weights.shape)
        #print(np.dot(self.input_weights, input).shape, self.input_bias.shape)
        h = np.dot(self.input_weights, input) + self.input_bias # Ux + c
        #print(h.shape)
        h += np.dot(self.hidden_weights, self.hidden[-1]) # Ws
        #print(h.shape)
        self.hidden.append(np.tanh(h)) # s = tanh(Ux + Ws + c)
        #print("hidden state s: ", np.sum(self.hidden[-1]))
        return self.hidden[-1]
    
    def forward_constant(self, inputs, time_steps):
        self.inputs = inputs
        self.hidden = [np.zeros(self.hidden[0].shape)]
        for time in range(time_steps):
            self.forward_step(self.inputs[time])
        return self.hidden
    
    def backward_step(self, time):
        # Only hidden state gradient is thrown back from the decoder
        # Gradient w.r.t hidden state, s
        dL_ds = self.dL_dh_next
        
        # Gradient through tanh
        dL_dh = (1 - self.hidden[time+1] ** 2) * dL_ds
        
        # Gradient for hidden weights, W
        self.dL_dW += np.dot(dL_dh, self.hidden[time].T)
        
        # Gradient for input weights and bias, U and c
        self.dL_dU += np.dot(dL_dh, self.inputs[time].T)
        self.dL_dc += dL_dh
        
        # Gradient for the previous hidden state
        self.dL_dh_next = np.dot(self.hidden_weights.T, dL_dh)
        
        # Gradient for the input
        return np.dot(self.input_weights.T, dL_dh)
    
    def backward(self):
        time_steps = len(self.inputs)
        #print(len(self.hidden)) # this should be equal to time_steps + 1
        input_gradients = []
        for time in reversed(range(time_steps)):
            input_gradient = self.backward_step(time)
            input_gradients.append(input_gradient)
        return input_gradients # We return the gradients for the input
    
    def update(self, lr: float):
        # Update the weights and biases
        self.input_weights -= lr * self.dL_dU
        self.input_bias -= lr * self.dL_dc
        self.hidden_weights -= lr * self.dL_dW
        # Reset the gradients
        self.dL_dW = np.zeros(self.hidden_weights.shape)
        self.dL_dU = np.zeros(self.input_weights.shape)
        self.dL_dc = np.zeros(self.input_bias.shape)
        self.dL_dh_next = np.zeros(self.hidden[0].shape)
        self.inputs = []
        self.outputs = []
        self.hidden = [np.zeros(self.hidden[0].shape)]