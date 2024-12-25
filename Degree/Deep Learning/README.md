# README for Deep Learning Folder

---

This folder contains code, resources, and notes related to the **Deep Learning** course. Below is a detailed explanation of the contents:

## Python Scripts

1. **`_01_neuron.py`**
   - Implements classes for different types of neurons:
     - **mpNeuron**: Models a McCulloch-Pitts neuron.
     - **perceptron**: Implements the perceptron model.
     - **networkPerceptron**: Handles a network of perceptrons.
     - **sigmoidNeuron**: Models a sigmoid activation-based neuron.

2. **`_02_feedforward.py`**
   - Contains a class for building and testing **feedforward networks**.

3. **`_03_gradientDescents.py`**
   - Provides an implementation of **gradient descent methods**:
     - Vanilla Gradient Descent
     - Momentum Gradient Descent
     - Nesterov Accelerated Gradient Descent

4. **`_04_adaptiveLearningRates.py`**
   - Implements a neural network with various **adaptive learning rate update functions**:
     - `adaGrad`, `rmsProp`, `adaDelta`, `adam`, `maxProp`, `adaMax`, `Nadam`, and a default update method.

5. **`_05_wordRepresentation.py`**
   - Demonstrates methods to represent words in a corpus:
     - One-hot Encoding
     - Positive Pointwise Mutual Information (PPMI)
     - Singular Value Decomposition (SVD)
     - Bag of Words (BoW) Model

6. **`_06_layers.py`**
   - Implements different types of neural network layers:
     - Artificial Neuron Layer
     - Convolutional Layer
     - Max Pooling Layer
     - Recurrent Neural Network (RNN) Layer
     - RNN Encoder Layer

7. **`_11_convolutionTest.py`**
   - A simple test script for **Convolutional Neural Networks (CNNs)**.

8. **`_12_recurrentNetworksTest.py`**
   - A test script for **Recurrent Neural Networks (RNNs)**.

9. **`_13_encoderDecoderTest.py`**
   - Tests an RNN-based **encoder-decoder architecture**.

---

## Jupyter Notebook

- **`Deep_Learning_Workshop.ipynb`**
  - Contains code and explanations from a **PyTorch workshop** provided as part of bonus assignments.

---

## Resources and Supporting Files

1. **`Deep_Learning.pdf`**
   - The complete notes for the course, including theory, concepts, and examples.

2. **`dog.jpg`**
   - An image used as input for the CNN test.

3. **`parameters_rnn.npz`**
   - Contains pre-trained parameters for the RNN test.

4. **`parameters_w11.npz`**
   - Stores pre-trained parameters for the encoder-decoder test.

5. **`parameters.npz`**
   - Includes pre-trained parameters for the feedforward network test.

---

## Usage Notes

- The scripts in this folder demonstrate various foundational and advanced deep learning concepts.
- Pre-trained parameter files (`*.npz`) are used for specific tests and examples to save time and ensure accurate results.
- **`dog.jpg`** serves as a sample input image for CNN tests.
- For additional explanations and implementation details, refer to **`Deep_Learning.pdf`** and the workshop notebook.

---

Feel free to explore the scripts, modify the code, or use them as a reference for building your deep learning projects!
