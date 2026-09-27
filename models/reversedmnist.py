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


def get_image(i):
    start = i * 784
    raw = images[start:start + 784]
    pixels = []
    for p in raw:
        pixels.append(p / 255)
    return pixels


# network pieces
def sigmoid(x):
    # clamp so math.exp can't overflow if a total gets huge
    x = max(-50, min(50, x))
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


# nudge one layer's weights and biases using its blame
def nudge(weights, biases, blame, inputs):
    for j in range(len(weights)):
        for p in range(len(inputs)):
            weights[j][p] -= lr * blame[j] * inputs[p]
        biases[j] -= lr * blame[j]


# pass blame back through a layer to whatever fed into it
def blame_back(blame, weights, said):
    back = []
    for p in range(len(said)):
        total = 0
        for j in range(len(blame)):
            total += blame[j] * weights[j][p]
        back.append(total * said[p] * (1 - said[p]))
    return back


# build the network
# describer: 784 pixels -> 32 hidden -> 2 style numbers
# drawer: 10 digit + 2 style -> 32 hidden -> 784 pixels
STYLES = 2

if os.path.exists("reversed_weights.json"):
    with open("reversed_weights.json") as f:
        enc_hw, enc_hb, enc_ow, enc_ob, dec_hw, dec_hb, dec_ow, dec_ob = json.load(f)
    print("loaded saved weights")
else:
    enc_hw, enc_hb = make_layer(784, 32)
    enc_ow, enc_ob = make_layer(32, STYLES)
    dec_hw, dec_hb = make_layer(10 + STYLES, 32)
    dec_ow, dec_ob = make_layer(32, 784)

lr = 0.2


def train_step(i):
    img = get_image(i)

    # describer looks at the real picture
    eh = layer(img, enc_hw, enc_hb)
    style = layer(eh, enc_ow, enc_ob)

    # drawer gets the digit plus the style
    inp = make_target(labels[i]) + style
    dh = layer(inp, dec_hw, dec_hb)
    out = layer(dh, dec_ow, dec_ob)

    # blame for pixel judges
    out_blame = []
    for k in range(784):
        said = out[k]
        out_blame.append((said - img[k]) * said * (1 - said))

    # blame flows back: drawer hidden, then style numbers, then describer hidden
    dh_blame = blame_back(out_blame, dec_ow, dh)
    inp_blame = blame_back(dh_blame, dec_hw, inp)
    style_blame = inp_blame[10:]
    eh_blame = blame_back(style_blame, enc_ow, eh)

    # nudge every layer
    nudge(dec_ow, dec_ob, out_blame, dh)
    nudge(dec_hw, dec_hb, dh_blame, inp)
    nudge(enc_ow, enc_ob, style_blame, eh)
    nudge(enc_hw, enc_hb, eh_blame, img)

    return loss(out, img)


# draw a digit with the style numbers you pick
def show(digit, style):
    inp = make_target(digit) + style
    dh = layer(inp, dec_hw, dec_hb)
    out = layer(dh, dec_ow, dec_ob)

    for r in range(28):
        line = ""
        for c in range(28):
            p = out[r * 28 + c]
            if p > 0.6:
                line += "##"
            elif p > 0.3:
                line += ".."
            else:
                line += "  "
        print(line)


# training loop
def train(steps):
    total = 0
    for i in range(steps):
        total += train_step(random.randint(0, len(labels) - 1))
        if i % 100 == 99:
            print(i + 1, total / 100)
            total = 0


t = int(input("training steps: "))
train(t)
with open("reversed_weights.json", "w") as f:
    json.dump([enc_hw, enc_hb, enc_ow, enc_ob, dec_hw, dec_hb, dec_ow, dec_ob], f)
print("saved weights")

while True:
    d = int(input("digit to draw (or -1 to quit): "))
    if d == -1:
        break
    s1 = float(input("style 1 (0 to 1): "))
    s2 = float(input("style 2 (0 to 1): "))
    show(d, [s1, s2])