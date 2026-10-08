/* def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def neuron(inputs, weights, bias):
    total = 0
    for i in range(len(inputs)):
        total += inputs[i] * weights[i]
    total += bias
    return sigmoid(total) */
#include <cmath>

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
