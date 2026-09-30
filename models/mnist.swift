import Foundation

// mnist

let trainImagesPath = "mnist/train-images-idx3-ubyte"
let trainLabelsPath = "mnist/train-labels-idx1-ubyte"
let testImagesPath = "mnist/t10k-images-idx3-ubyte"
let testLabelsPath = "mnist/t10k-labels-idx1-ubyte"

func loadFile(_ path: String) -> [UInt8] {
    let url = URL(fileURLWithPath: path)

    do {
        return try Array(Data(contentsOf: url))
    } catch {
        fatalError("Could not load \(path): \(error)")
    }
}

// mnist image files have a 16-byte header.
// mnist label files have an 8-byte header.

let trainImages = loadFile(trainImagesPath)
let trainLabels = loadFile(trainLabelsPath)

let testImages = loadFile(testImagesPath)
let testLabels = loadFile(testLabelsPath)


// mnist image

func getImage(_ index: Int) -> [Float] {
    let start = 16 + index * 784

    var pixels = [Float]()
    pixels.reserveCapacity(784)

    for i in 0..<784 {
        pixels.append(Float(trainImages[start + i]) / 255.0)
    }

    return pixels
}

func getTestImage(_ index: Int) -> [Float] {
    let start = 16 + index * 784

    var pixels = [Float]()
    pixels.reserveCapacity(784)

    for i in 0..<784 {
        pixels.append(Float(testImages[start + i]) / 255.0)
    }

    return pixels
}

func trainLabel(_ index: Int) -> Int {
    return Int(trainLabels[8 + index])
}

func testLabel(_ index: Int) -> Int {
    return Int(testLabels[8 + index])
}


// math :|

func sigmoid(_ x: Float) -> Float {
    return 1.0 / (1.0 + exp(-x))
}


// layer

struct Layer: Codable {
    var weights: [[Float]]
    var biases: [Float]
}

func makeLayer(inputs: Int, neurons: Int) -> Layer {
    var weights = [[Float]]()

    for _ in 0..<neurons {
        var neuronWeights = [Float]()
        neuronWeights.reserveCapacity(inputs)

        for _ in 0..<inputs {
            let value = Float.random(in: -0.1...0.1)
            neuronWeights.append(value)
        }

        weights.append(neuronWeights)
    }

    let biases = Array(repeating: Float(0.0), count: neurons)

    return Layer(
        weights: weights,
        biases: biases
    )
}


// forward pass

func layer(
    _ inputs: [Float],
    _ layer: Layer
) -> [Float] {

    var outputs = [Float]()
    outputs.reserveCapacity(layer.weights.count)

    for j in 0..<layer.weights.count {
        var total = layer.biases[j]

        for i in 0..<inputs.count {
            total += inputs[i] * layer.weights[j][i]
        }

        outputs.append(sigmoid(total))
    }

    return outputs
}


// target

func makeTarget(_ digit: Int) -> [Float] {
    var target = Array(repeating: Float(0.0), count: 10)
    target[digit] = 1.0
    return target
}


// loss

func loss(
    _ output: [Float],
    _ target: [Float]
) -> Float {

    var total: Float = 0.0

    for i in 0..<output.count {
        let difference = output[i] - target[i]
        total += difference * difference
    }

    return total
}


// network

struct Network: Codable {
    var hidden: Layer
    var output: Layer
}

var network: Network

let weightsPath = "weights.json"

if FileManager.default.fileExists(atPath: weightsPath) {

    do {
        let data = try Data(contentsOf: URL(fileURLWithPath: weightsPath))

        network = try JSONDecoder().decode(
            Network.self,
            from: data
        )

        print("loaded saved weights")

    } catch {
        fatalError("Could not load weights: \(error)")
    }

} else {

    network = Network(
        hidden: makeLayer(
            inputs: 784,
            neurons: 32
        ),

        output: makeLayer(
            inputs: 32,
            neurons: 10
        )
    )
}

let learningRate: Float = 0.5


// train step

func trainStep(_ index: Int) -> Float {

    // forward pass

    let image = getImage(index)

    let hidden = layer(
        image,
        network.hidden
    )

    let output = layer(
        hidden,
        network.output
    )

    let target = makeTarget(
        trainLabel(index)
    )


    // output gradients

    var outputBlame = Array(
        repeating: Float(0.0),
        count: 10
    )

    for k in 0..<10 {

        let said = output[k]
        let wanted = target[k]

        outputBlame[k] =
            (said - wanted)
            * said
            * (1.0 - said)
    }


    // hidden gradients

    var hiddenBlame = Array(
        repeating: Float(0.0),
        count: 32
    )

    for j in 0..<32 {

        var total: Float = 0.0

        for k in 0..<10 {
            total +=
                outputBlame[k]
                * network.output.weights[k][j]
        }

        let said = hidden[j]

        hiddenBlame[j] =
            total
            * said
            * (1.0 - said)
    }


    // update weights output

    for k in 0..<10 {

        for j in 0..<32 {

            network.output.weights[k][j] -=
                learningRate
                * outputBlame[k]
                * hidden[j]
        }

        network.output.biases[k] -=
            learningRate
            * outputBlame[k]
    }


    // update hidden weights

    for j in 0..<32 {

        for p in 0..<784 {

            network.hidden.weights[j][p] -=
                learningRate
                * hiddenBlame[j]
                * image[p]
        }

        network.hidden.biases[j] -=
            learningRate
            * hiddenBlame[j]
    }


    return loss(
        output,
        target
    )
}


// test image display

func show(_ index: Int) {

    let start = 16 + index * 784

    print("")
    print("────────────────────────────")
    print("Test image \(index)")
    print("────────────────────────────")

    for row in 0..<28 {

        var line = ""

        for column in 0..<28 {

            let pixel =
                testImages[
                    start + row * 28 + column
                ]

            if pixel > 128 {
                line += "##"
            } else {
                line += "  "
            }
        }

        print(line)
    }


    // guess

    let image = getTestImage(index)

    let hidden = layer(
        image,
        network.hidden
    )

    let output = layer(
        hidden,
        network.output
    )

    var bestDigit = 0

    for i in 1..<10 {
        if output[i] > output[bestDigit] {
            bestDigit = i
        }
    }

    print("")
    print("prediction:", bestDigit)
    print("answer:", testLabel(index))

    print("")
    print("probabilities:")

    for i in 0..<10 {
        let percentage = output[i] * 100.0

        print(
            "\(i): \(String(format: "%.2f", percentage))%"
        )
    }
}


// training

func train(_ steps: Int) {

    for step in 0..<steps {

        let index = Int.random(
            in: 0..<60_000
        )

        let currentLoss = trainStep(index)

        if step % 100 == 0 {
            print(
                "step \(step) | loss \(currentLoss)"
            )
        }
    }
}


// main

print("")
print("mnist neural network in swift because ssc")
print("784 → 32 → 10")
print("")

print("training steps:", terminator: " ")

if let input = readLine(),
   let steps = Int(input) {

    train(steps)

} else {

    print("Invalid number of steps.")
    exit(1)
}


// save weights

do {

    let data = try JSONEncoder().encode(network)

    try data.write(
        to: URL(fileURLWithPath: weightsPath)
    )

    print("")
    print("saved weights")

} catch {

    print("Could not save weights:", error)
}


// test

let randomTestImage = Int.random(
    in: 0..<10_000
)

show(randomTestImage)
