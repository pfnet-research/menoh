import numpy as np
import chainer
import argparse


def main():
    parser = argparse.ArgumentParser(description='generate operator output')
    parser.add_argument('--input', type=str, help='input file path')
    parser.add_argument('--output', type=str, help='output file path')
    parser.add_argument('--op', type=str, default=None, help='operator')
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            if (arg == "--W" or arg == "--b") or (
                    parsed.op == "fixed_batch_normalization" and
                (arg == "--gamma" or arg == "--beta" or arg == "--mean"
                 or arg == "--var")):
                parser.add_argument(arg, type=str)
            elif (parsed.op == "fixed_batch_normalization" and arg == "--eps"
                  ) or (parsed.op == "leaky_relu" and arg == "--slope") or (
                      parsed.op == "elu" and arg == "--alpha") or (
                          parsed.op == "local_response_normalization" and
                          (arg == "--alpha" or arg == "--beta"
                           or arg == "--bias")):
                parser.add_argument(arg, type=float)
            else:
                parser.add_argument(arg, type=int)
    args = parser.parse_args()
    print(args)

    with open(args.input, "r") as f:
        dims_num = int(f.readline())
        shape = tuple(int(d) for d in f.readline().strip().split(" "))
        raw_data = [np.float32(d) for d in f.readline().strip().split(" ")]
        x = np.array(raw_data).reshape(shape)

    def layer(x, **kwargs):
        chainer.using_config("train", False)
        return eval("chainer.functions.{}(x, **kwargs)".format(args.op))

    op = eval("chainer.functions.{}".format(args.op))
    op_arg_count = op.__code__.co_argcount
    op_args = op.__code__.co_varnames[:op_arg_count]

    op_args_dict = {}
    for k in op_args:
        if k in vars(args).keys():
            if k == "W":
                with open(args.W, "r") as f:
                    dims_num = int(f.readline())
                    shape = [int(v) for v in f.readline().split()]
                    assert len(shape) == dims_num
                    data = [float(v) for v in f.readline().split()]
                    W = np.asarray(data).reshape(shape).astype(np.float32)
                    assert (k == "W")
                    op_args_dict[k] = W
            elif k == "b":
                with open(args.b, "r") as f:
                    dims_num = int(f.readline())
                    shape = [int(v) for v in f.readline().split()]
                    assert len(shape) == dims_num
                    data = [float(v) for v in f.readline().split()]
                    b = np.asarray(data).reshape(shape).astype(np.float32)
                    assert (k == "b")
                    op_args_dict[k] = b
            elif args.op == "fixed_batch_normalization" and (
                    k == "gamma" or k == "beta" or k == "mean" or k == "var"):
                with open(vars(args)[k], "r") as f:
                    dims_num = int(f.readline())
                    shape = [int(v) for v in f.readline().split()]
                    assert len(shape) == dims_num
                    data = [float(v) for v in f.readline().split()]
                    data = np.asarray(data).reshape(shape).astype(np.float32)
                    op_args_dict[k] = data
            else:
                op_args_dict[k] = vars(args)[k]

    with chainer.using_config('train', False):
        output = layer(x, **op_args_dict)
    with open(args.output, "w") as f:
        f.write(str(len(output.shape)))
        f.write("\n")
        f.write(" ".join([str(d) for d in output.shape]))
        f.write("\n")
        f.write(" ".join([str(x) for x in output.array.flat]))


if __name__ == "__main__":
    main()
