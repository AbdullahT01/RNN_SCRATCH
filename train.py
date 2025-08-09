import numpy as np 
import random 
from data import train_data, test_data

# ------------------ STEP 1: Simple Preprocessing ------------------

all_sentences = list(train_data.keys()) + list(test_data.keys())

vocab_set = set()
for sentence in all_sentences:
    words = sentence.split()
    for w in words:
        vocab_set.add(w)

vocab = sorted(vocab_set)

word_to_index = {}
for i, w in enumerate(vocab):
    word_to_index[w] = i


index_to_word = {}
for w, i in word_to_index.items():
    index_to_word[i] = w

# Example labels
output_labels = ['lights_on', 'lights_off', 'play_music', 'stop_music', 'set_alarm', 'check_weather']

# Create mappings
label_to_index = {}
for i, w in enumerate(output_labels):
    label_to_index[w] = i


index_to_label = {}
for w, i in word_to_index.items():
    index_to_label[i] = w

vocab_size = len(word_to_index)
print(f'There are {vocab_size} unique words in our data set.')


#Now we will convert the sentences into a list of one-hot vectors
def sentence_to_onehot(sentence):
    vectors = [] 
    for word in sentence.split():
        v = np.zeros((vocab_size, 1))
        v[word_to_index[word]] = 1
        vectors.append(v)
    return vectors

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

# Cross entropy loss
def cross_entropy_loss(pred, target_index):
    return -np.log(pred[target_index, 0] + 1e-8)

# ------------------ STEP 2: Initialize RNN Parameters ------------------

hidden_size = 64
output_size = 6
learning_rate = 0.01
epochs = 500

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(output_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))

# ------------------ STEP 3: Training ------------------

print("\n--- Training Started ---")

for epoch in range(epochs):
    total_loss = 0
    # shuffling the training data to give variation in training 
    combined = list(train_data.items())
    random.shuffle(combined)

    for sentence, target in combined:
        inputs = sentence_to_onehot(sentence) # creates a list of one hot vectors of our sentence
        target = label_to_index[target]

        # ------------------------------- Forward pass----------------------------------------------

        hs = {} # dictionary to store hidden states for each timestep.
        hs[-1] = np.zeros((hidden_size, 1)) # the hidden state before the first word — initialized to all zeros.

        for t in range(len(inputs)):
            hs[t] = np.tanh(np.dot(Wxh, inputs[t]) + np.dot(Whh, hs[t - 1]) + bh) # using the formula to update hiddent state

        y = np.dot(Why, hs[len(inputs) - 1]) + by
        p = softmax(y)
        loss = cross_entropy_loss(p, target)
        total_loss += loss


        # ------------------- Backpropagation from final output only --------------------------------

        dy = p
        dy[target] -= 1

        dWhy = np.dot(dy, hs[len(inputs) - 1].T)
        dby = dy
        dWxh = np.zeros_like(Wxh)
        dWhh = np.zeros_like(Whh)
        dbh = np.zeros_like(bh)
        dh_next = np.dot(Why.T, dy)

        for t in reversed(range(len(inputs))):
            dh = dh_next
            dh_raw = (1 - hs[t] ** 2) * dh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, inputs[t].T)
            dWhh += np.dot(dh_raw, hs[t - 1].T)
            dh_next = np.dot(Whh.T, dh_raw)

        # Clip gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update weights
        Wxh -= learning_rate * dWxh
        Whh -= learning_rate * dWhh
        Why -= learning_rate * dWhy
        bh -= learning_rate * dbh
        by -= learning_rate * dby

    if epoch % 25 == 0:
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

# ------------------ STEP 4: Final Evaluation ------------------

print("\n--- Final Evaluation on Test Set ---")
correct = 0
for sentence, label in test_data.items():
    inputs = sentence_to_onehot(sentence)
    h = np.zeros((hidden_size, 1))
    for x in inputs:
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = softmax(y)
    pred = np.argmax(p)
    if pred == label_to_index[label]:
        correct += 1
print(f"Accuracy: {correct} / {len(test_data)} ({correct / len(test_data) * 100:.2f}%)")
# ------------------ STEP 5: Sample Predictions ------------------

print("\n--- Sample Predictions ---")
sample_sentences = [
    "turn on lights for seven",
    "please switch off the lights",
    "can you play a song please",
    "pause music",
    "set an alarm for seven",
]

for sentence in sample_sentences:
    inputs = sentence_to_onehot(sentence)
    h = np.zeros((hidden_size, 1))
    
    for x in inputs:
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    
    y = np.dot(Why, h) + by
    p = softmax(y)
    pred_index = np.argmax(p)
    predicted_label = output_labels[pred_index]

    print(f"Input: '{sentence}' ----------------→ Predicted Intent: {predicted_label}")
    print("")


