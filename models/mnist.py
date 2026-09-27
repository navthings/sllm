f = open("mnist/train-images-idx3-ubyte", "rb")

header = f.read(16)
magic = int.from_bytes(header[0:4], "big")
count = int.from_bytes(header[4:8], "big")
rows = int.from_bytes(header[8:12], "big")
cols = int.from_bytes(header[12:16], "big")
print(magic, count, rows, cols)

pixels = f.read(784)

for r in range(28):
    line = ""
    for c in range(28):
        p = pixels[r * 28 + c]
        if p > 128:
            line += "##"
        else:
            line += "  "
    print(line)
    

lf = open("mnist/train-labels-idx1-ubyte", "rb")
lheader = lf.read(8)
label = lf.read(1)
print(label[0])