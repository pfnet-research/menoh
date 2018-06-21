import numpy as np
import chainer
import chainer.links as L
import argparse
import time


def main():
    parser = argparse.ArgumentParser(
        description='generate 2d proessing operator output')
    parser.add_argument('--with-ideep', action='store_true', help='enable ideep')
    parser.add_argument('--input', type=str, help='input file path')
    args = parser.parse_args()
    with open(args.input, "r") as f:
        dims_num = int(f.readline())
        shape = tuple(int(d) for d in f.readline().strip().split(" "))
        raw_data = [np.float32(d) for d in f.readline().strip().split(" ")]
        x = np.array(raw_data).reshape(shape)

    chainer.config.train = False

    model = L.VGG16Layers()
    if args.with_ideep:
        chainer.config.use_ideep = "auto"
        model.to_intel64()

    start = time.process_time()
    y = model(x)
    end = time.process_time()
    print((end - start) * 1000)


if __name__ == "__main__":
    main()
