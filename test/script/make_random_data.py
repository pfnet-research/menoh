import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='generate 2d proessing operator output')
    parser.add_argument('--output', type=str, help='output file path')
    parser.add_argument('dims', type=int, nargs="*", help='dims')
    parser.add_argument(
        '--positive', action="store_true", help='values are all positive')
    args = parser.parse_args()

    dims = args.dims

    np.random.seed(0)
    x = np.random.rand(*dims).astype(np.float32) - 0.5
    #x = np.ones(dims).astype(np.float32)
    if args.positive:
        x = np.abs(x)

    with open(args.output, "w") as f:
        f.write(str(len(x.shape)))
        f.write("\n")
        f.write(" ".join([str(d) for d in x.shape]))
        f.write("\n")
        f.write(" ".join([str(d) for d in x.flat]))


if __name__ == "__main__":
    main()
