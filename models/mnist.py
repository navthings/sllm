import math
import random

f = open("mnist/train-images-idx3-ubyte", "rb")
f.read(16)
images = f.read()

lf = open("mnist/train-labels-idx1-ubyte", "rb")
lf.read(8)
labels = lf.read()

def get_image(i):
    start = i * 784
    raw = images[start:start + 784]
    pixels = []
    for i in raw:
        pixels.append(i / 255)
    return pixels

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def neuron(inputs, weights, bias):
    total = 0
    for i in range(len(inputs)):
        total += inputs[i] * weights[i]
    total += bias
    return sigmoid(total)

def make_layer(n_inputs, n_neurons):
    weights = []
    for _ in range(n_neurons):
        w = []
        for _ in range(n_inputs):
            w.append(random.uniform(-0.1, 0.1))
        weights.append(w)
    biases = [0.0] * n_neurons
    return weights, biases

def layer(inputs, weights, biases):
    outputs = []
    for j in range(len(weights)):
        outputs.append(neuron(inputs, weights[j], biases[j]))
    return outputs

hidden_w, hidden_b = make_layer(784, 32)
out_w, out_b = make_layer(32, 10)

img = get_image(0)
h = layer(img, hidden_w, hidden_b)
out = layer(h, out_w, out_b)
print(out)
print("guess:", out.index(max(out)), "answer:", labels[0])