import math

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

# [sunny, rainy]
print(loss([0.5, -1], 0.1))
print(loss([2, -4], 0.4))

before = loss([0.5, -1], 0.1)
after = loss([0.5, -1 + 0.001], 0.1)
print(before, after)
print((after - before) / 0.001)