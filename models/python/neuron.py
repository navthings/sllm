import math

steps = int(input('training steps: '))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def neuron(inputs, weights, bias):
    total = 0
    for i in range(len(inputs)):
        total += inputs[i] * weights[i]
    total += bias
    return sigmoid(total)

data = [
    ([1, 0], 1),
    ([1, 1], 0),
]

def loss(weights, bias):
    total = 0
    for inputs, target in data:
        prediction = neuron(inputs, weights, bias) # get the neuron's prediction for these inputs
        total += (prediction - target) ** 2 # add (prediction - target) squared to total
    return total


def gradients(weights, bias):
    step = 0.001
    base = loss(weights, bias)

    g_sun = (loss([weights[0] + step, weights[1]], bias) - base) / step
    g_rain = (loss([weights[0], weights[1] + step], bias) - base) / step 
    g_bias = (loss(weights, bias + step) - base) / step

    return g_sun, g_rain, g_bias

weights = [0.0, 0.0]
bias = 0.0
lr = 0.5


for i in range(steps):
    g_sun, g_rain, g_bias = gradients(weights, bias)
    
    weights[0] -= lr * g_sun
    weights[1] -= lr * g_rain
    bias -= lr * g_bias
    print(i, loss(weights, bias))
    

print(weights, bias)

print(neuron([1, 0], weights, bias))
print(neuron([1, 1], weights, bias))