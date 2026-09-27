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
    # loop through raw, add each pixel divided by 255 to pixels
    return pixels

img = get_image(0)
print(len(img), max(img), labels[0])