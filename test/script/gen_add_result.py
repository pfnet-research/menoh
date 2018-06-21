import numpy as np
import argparse


def load_np_array_from_file(filename):
    with open(filename, "r") as f:
        dims_num = int(f.readline())
        shape = tuple(int(d) for d in f.readline().strip().split(" "))
        raw_data = [np.float32(d) for d in f.readline().strip().split(" ")]
        return np.array(raw_data).reshape(shape)


def main():
    parser = argparse.ArgumentParser(description='generate operator output')
    parser.add_argument('--input_a', type=str, help='input file path')
    parser.add_argument('--input_b', type=str, help='input file path')
    parser.add_argument('--output', type=str, help='output file path')

    args = parser.parse_args()
    print(args)

    a = load_np_array_from_file(args.input_a)
    b = load_np_array_from_file(args.input_a)

    output = a + b

    with open(args.output, "w") as f:
        f.write(str(len(output.shape)))
        f.write("\n")
        f.write(" ".join([str(d) for d in output.shape]))
        f.write("\n")
        f.write(" ".join([str(x) for x in output.flat]))


if __name__ == "__main__":
    main()
