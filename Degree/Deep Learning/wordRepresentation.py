import numpy as np

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we live and work.",
    "Machine learning algorithms can improve with more data and experience.",
    "The sun sets behind the mountains, casting a golden glow on the landscape.",
    "Reinforcement learning is a type of machine learning where agents learn by interacting with their environment.",
    "The trees in the forest swayed gently in the cool breeze.",
    "A strong foundation in mathematics is essential for understanding machine learning models.",
    "The library was quiet, with only the sound of turning pages filling the air.",
    "Data scientists use statistical techniques to analyze and interpret complex data.",
    "The cat curled up on the windowsill, basking in the warm sunlight.",
    "Quantum computing has the potential to revolutionize industries like cryptography and drug discovery.",
    "The river flowed calmly through the valley, reflecting the clear blue sky above.",
    "Natural language processing helps computers understand and generate human language.",
    "The city skyline was illuminated by a thousand lights as night fell.",
    "In the world of finance, understanding market trends is crucial for making informed decisions."
]

# One hot encoding
words = []
for sentence in corpus:
    temp = sentence.lower().split()
    for i, word in enumerate(temp):
        if word[-1] == ',' or word[-1] == '.':
            temp[i] = word[:-1] # Remove the comma or period at the end of the word

    words.extend(temp)

words = list(set(words))
print("Total number of unique words in the corpus: ", len(words))

word_to_index = {word: i for i, word in enumerate(words)}
one_hot_vectors = np.zeros((len(words), len(words)))
for word, i in word_to_index.items():
    one_hot_vectors[i, i] = 1

print("One hot vector for the word 'machine': ", one_hot_vectors[word_to_index['machine']])


# PPMI
word_freq = {}
for sentence in corpus:
    temp = sentence.lower().split()
    for word in temp:
        if word[-1] == ',' or word[-1] == '.':
            word = word[:-1] # Remove the comma or period at the end of the word

        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

word_to_index = {word: i for i, word in enumerate(word_freq.keys())}
index_to_word = {i: word for i, word in enumerate(word_freq.keys())}

coocurrence_matrix = np.zeros((len(word_freq), len(word_freq)))
window_size = 2
for sentence in corpus:
    temp = sentence.lower().split()
    for i, word in enumerate(temp):
        if word[-1] == ',' or word[-1] == '.':
            temp[i] = word[:-1] # Remove the comma or period at the end of the word
    for i, word in enumerate(temp):
        for j in range(max(i - window_size, 0), min(i + window_size, len(temp))):
            if i != j:
                coocurrence_matrix[word_to_index[word], word_to_index[temp[j]]] += 1 # This will make count(w,c)

coocurrence_matrix *= len(word_freq) # Multiply by the size of the corpus
# Divide by count(c) * count(w)
for i in range(len(word_freq)):
    for j in range(len(word_freq)):
        if coocurrence_matrix[i, j] != 0:
            coocurrence_matrix[i, j] = np.log(coocurrence_matrix[i, j] / (word_freq[index_to_word[i]] * word_freq[index_to_word[j]]))
            coocurrence_matrix[i, j] = max(0, coocurrence_matrix[i, j]) # PPMI

print("PPMI value for the word 'machine' and 'learning': ", coocurrence_matrix[word_to_index['machine'], word_to_index['learning']])
print("Word representation for the word 'machine': ", coocurrence_matrix[word_to_index['machine']])
# Non zero entries in word representation of machine correspond to
print("Words that have non zero PPMI value with the word 'machine': ", [index_to_word[i] for i in range(len(word_freq)) if coocurrence_matrix[word_to_index['machine'], i] != 0])
# learning, type, of, for, understanding, from corpus this makes sense.

# SVD to get rank k approximation
print("Shape of coocurrence matrix: ", coocurrence_matrix.shape)
U, S, V = np.linalg.svd(coocurrence_matrix)
k = 4
U_k = U[:, :k]
S_k = np.diag(S[:k])
V_k = V[:k, :]
print("Size of U_k: ", U_k.shape, "Size of S_k: ", S_k.shape, "Size of V_k: ", V_k.shape)
word_representation = np.dot(U_k, S_k)
print("Word representation for the word 'machine': ", word_representation[word_to_index['machine']])
context_representation = V_k
print("Context representation for the word 'machine': ", context_representation[:, word_to_index['machine']])

# Bag of words model
# Predict nth word given n-1 words

# Training only for 1 sentence, so redefining words
print("Training just for one sentence, ", corpus[2])
sentence = corpus[2].lower().split()
for i, word in enumerate(sentence):
    if word[-1] == ',' or word[-1] == '.':
        sentence[i] = word[:-1] # Remove the comma or period at the end of the word
words = list(set(sentence))
word_to_index = {word: i for i, word in enumerate(words)}
one_hot_vectors = np.zeros((len(words), len(words)))
for word, i in word_to_index.items():
    one_hot_vectors[i, i] = 1

print("Bag of Words with vocab size ", len(words))
input_size = len(words) # One hot encoding of words
output_size = len(words) # One hot encoding of words
hidden_size = 10
print("Input size: ", input_size, "Output size: ", output_size, "Hidden size: ", hidden_size)
learning_rate = 1e-2
epochs = 500
k = 2 # Number of previous words to consider
input_weights = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
input_weights_gradient = np.zeros((hidden_size, input_size))
output_weights = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
output_weights_gradient = np.zeros((output_size, hidden_size))

# Number of parameters = input_size * hidden_size + hidden_size * output_size
print("Number of parameters: ", input_size * hidden_size + hidden_size * output_size)

# For one hot encoding, the index where the word is 1 is the index of the weight
def forward_propagation(input_word):
    #print("Input word: ", input_word.shape, input_word) 
    indices = []
    for i in range(len(input_word)):
        indices.append(np.where(input_word[i] == 1)[0])
    #print("Indices: ", indices)
    hidden_layer = input_weights[:,indices].sum(axis=1)
    #print("Hidden layer: ", hidden_layer.shape)
    hidden_layer = np.tanh(hidden_layer) # Activation function
    output_layer = np.dot(output_weights, hidden_layer)
    #print("Output layer: ", output_layer.shape)
    output_layer = np.exp(output_layer) / np.sum(np.exp(output_layer)) # Softmax
    return hidden_layer, output_layer

def backword_propagation(hidden_layer, output_layer, target, input_word):
    input_word = input_word.reshape(-1, 1)
    target = target.reshape(-1, 1)
    error = output_layer - target
    output_weights_gradient = np.outer(error, hidden_layer)
    hidden_error = np.dot(output_weights.T, error)
    hidden_error = hidden_error * (1 - np.square(hidden_layer))
    indices = []
    for i in range(len(input_word)):
        indices.append(np.where(input_word[i] == 1)[0])
    input_weights_gradient = np.zeros((hidden_size, input_size))
    for i in range(len(indices)):
        input_weights_gradient[:,indices[i]] += hidden_error
    return output_weights_gradient, input_weights_gradient

def update_weights(output_weights_gradient, input_weights_gradient):
    global output_weights, input_weights
    output_weights -= learning_rate * output_weights_gradient
    input_weights -= learning_rate * input_weights_gradient

# Training
hidden_layer, output_layer = forward_propagation(one_hot_vectors[[word_to_index[sentence[0]], word_to_index[sentence[1]]]])
print("Initial prediction for the word 'machine learning': ", words[np.argmax(output_layer)])
for epoch in range(epochs):
    temp = corpus[2].lower().split()
    for i, word in enumerate(temp):
        if word[-1] == ',' or word[-1] == '.':
            temp[i] = word[:-1] # Remove the comma or period at the end of the word
    for i in range(k, len(temp)):
        indices = []
        for j in range(i - k, i):
            indices.append(word_to_index[temp[j]])
        input_word = one_hot_vectors[indices]
        target = one_hot_vectors[word_to_index[temp[i]]]
        hidden_layer, output_layer = forward_propagation(input_word)
        new_output_weights_gradient, new_input_weights_gradient = backword_propagation(hidden_layer, output_layer, target, input_word)
        #print("Output Gradient shape: ", output_weights_gradient.shape, "Input Gradient shape: ", input_weights_gradient.shape)
        output_weights_gradient += new_output_weights_gradient
        input_weights_gradient += new_input_weights_gradient
    update_weights(output_weights_gradient, input_weights_gradient)
    output_weights_gradient = np.zeros((output_size, hidden_size))
    input_weights_gradient = np.zeros((hidden_size, input_size))
    if epoch % (epochs//10) == 0:
        print("Epoch: ", epoch, "Loss: ", -np.log(output_layer[word_to_index[temp[i]]]))

hidden_layer, output_layer = forward_propagation(one_hot_vectors[[word_to_index['machine'], word_to_index['learning']]])
print("Next word prediction after 'machine learning': ", words[np.argmax(output_layer)])
print("Loss: ", -np.log(output_layer[word_to_index['algorithms']]))