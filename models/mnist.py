f = open("mnist/train-images-idx3-ubyte", "rb")
header = f.read(16)
magic = int.from_bytes(header[0:4], "big")
print(magic)