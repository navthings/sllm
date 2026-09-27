import json
import os
import math
import random

# load data
f = open("mnist/train-images-idx3-ubyte", "rb")
f.read(16)
images = f.read()

lf = open("mnist/train-labels-idx1-ubyte", "rb")
lf.read(8)
labels = lf.read()

# testing
tf = open("mnist/t10k-images-idx3-ubyte", "rb")
tf.read(16)
test_images = tf.read()

tlf = open("mnist/t10k-labels-idx1-ubyte", "rb")
tlf.read(8)
test_labels = tlf.read()


# get the pixels for a image
def get_image(i):
    start = i * 784
    raw = images[start:start + 784]
    pixels = []
    for p in raw:
        pixels.append(p / 255)
    return pixels


# network pieces
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


def make_target(digit):
    a = [0] * 10
    a[digit] = 1
    return a


def loss(out, target):
    total = 0
    for k in range(len(out)):
        total += (out[k] - target[k]) ** 2
    return total


# build the network
if os.path.exists("reversed_weights.json"):
    with open("reversed_weights.json") as f:
        hidden_w, hidden_b, out_w, out_b = json.load(f)
    print("loaded saved weights")
else:
    hidden_w, hidden_b = make_layer(784, 32)
    out_w, out_b = make_layer(32, 10)

lr = 0.5


# one training step on picture i
def train_step(i):
    # guess
    img = get_image(i)
    h = layer(img, hidden_w, hidden_b)
    out = layer(h, out_w, out_b)
    target = make_target(labels[i])

    # blame for digit judges
    out_blame = []
    for k in range(10):
        said = out[k]
        wanted = target[k]
        out_blame.append((said - wanted) * said * (1 - said))

    # blame for hidden judges
    hidden_blame = []
    for j in range(32):
        total = 0
        for k in range(10):
            total += out_blame[k] * out_w[k][j]
        said = h[j]
        hidden_blame.append(total * said * (1 - said))

    # nudge digit judges
    for k in range(10):
        for j in range(32):
            out_w[k][j] -= lr * out_blame[k] * h[j]
        out_b[k] -= lr * out_blame[k]

    # nudge hidden judges
    for j in range(32):
        for p in range(784):
            hidden_w[j][p] -= lr * hidden_blame[j] * img[p]
        hidden_b[j] -= lr * hidden_blame[j]

    return loss(out, target)


# draw a test picture and show the network's guess
def show(i):
    start = i * 784
    raw = test_images[start:start + 784]

    # draw it
    for r in range(28):
        line = ""
        for c in range(28):
            if raw[r * 28 + c] > 128:
                line += "##"
            else:
                line += "  "
        print(line)

    # guess
    img = []
    for p in raw:
        img.append(p / 255)
    h = layer(img, hidden_w, hidden_b)
    out = layer(h, out_w, out_b)
    print("guess:", out.index(max(out)), "answer:", test_labels[i])


# training loop
def train(steps):
    for i in range(steps):
        idx = random.randint(0, 59999)
        l = train_step(idx)
        print(i, l)


t = int(input('training steps: '))
train(t)
with open("reversed_weights.json", "w") as f:
    json.dump([hidden_w, hidden_b, out_w, out_b], f)
print("saved weights")

show(random.randint(0, 9999))