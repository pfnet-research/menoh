import errno
import os
import os.path
import sys
import subprocess


def call(cmd):
    """ call command line.
    """

    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    ret = p.wait()
    print('')


try:
    os.mkdir('data')
except OSError as e:
    if e.errno != errno.EEXIST:
        traceback.print_exc()
        sys.exit(1)
    if not os.path.isdir('data'):
        sys.stderr.write('Failed to create data directory. Is there a regular file named `data`?\n')
        sys.exit(1)

pyexe = sys.executable

# Generate input data
call(
    pyexe +
    ' test/script/make_random_data.py 3 4096 --output data/random_input_3_4096.txt'
)  # 4096 = 4*32*32
call(
    pyexe +
    ' test/script/make_random_data.py 3 4 32 32 --output data/random_input_3_4_32_32.txt'
)
call(
    pyexe +
    ' test/script/make_random_data.py 3 4096 --output data/random_positive_input_3_4096.txt --positive'
)
call(
    pyexe +
    ' test/script/make_random_data.py 3 4 32 32 --output data/random_positive_input_3_4_32_32.txt --positive'
)

# Generate weight and bias data
call(
    pyexe +
    ' test/script/make_random_data.py 5 4 1 1 --output data/random_weight_5_4_1_1.txt'
)
call(
    pyexe +
    ' test/script/make_random_data.py 5 4 2 2 --output data/random_weight_5_4_2_2.txt'
)
call(
    pyexe +
    ' test/script/make_random_data.py 5 4 3 3 --output data/random_weight_5_4_3_3.txt'
)
call(
    pyexe +
    ' test/script/make_random_data.py 4 5 1 1 --output data/random_weight_4_5_1_1.txt'
)
call(
    pyexe +
    ' test/script/make_random_data.py 4 5 2 2 --output data/random_weight_4_5_2_2.txt'
)
call(
    pyexe +
    ' test/script/make_random_data.py 4 5 3 3 --output data/random_weight_4_5_3_3.txt'
)
call(pyexe +
     ' test/script/make_random_data.py 5 --output data/random_bias_5.txt')
call(pyexe +
     ' test/script/make_random_data.py 5 --output data/random_bias_4.txt')
call(
    pyexe +
    ' test/script/make_random_data.py 256 4096 --output data/random_weight_256_4096.txt'
)
call(pyexe +
     ' test/script/make_random_data.py 256 --output data/random_bias_256.txt')
call(pyexe +
     ' test/script/make_random_data.py 4 --output data/random_gamma_4.txt')
call(pyexe +
     ' test/script/make_random_data.py 4 --output data/random_beta_4.txt')
call(pyexe +
     ' test/script/make_random_data.py 4 --output data/random_mean_4.txt')
call(pyexe +
     ' test/script/make_random_data.py 4 --output data/random_var_4.txt')

# Operators

## Relu
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4096.txt --output data/relu_1d.txt ' \
    '--op relu')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/relu_2d.txt ' \
    '--op relu')

## LeakyRelu
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4096.txt --output data/leaky_relu_1d.txt ' \
    '--op leaky_relu --slope 0.001')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/leaky_relu_2d.txt ' \
    '--op leaky_relu --slope 0.001')

## Elu
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4096.txt --output data/elu_1d.txt ' \
    '--op elu --alpha 1.1')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/elu_2d.txt ' \
    '--op elu --alpha 1.1')

## Abs
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4096.txt --output data/abs_1d.txt ' \
    '--op absolute')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/abs_2d.txt ' \
    '--op absolute')

## Tanh
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4096.txt --output data/tanh_1d.txt ' \
    '--op tanh')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/tanh_2d.txt ' \
    '--op tanh')

## Sqrt
call(pyexe + ' test/script/gen_op_result.py --input data/random_positive_input_3_4096.txt --output data/sqrt_1d.txt ' \
    '--op sqrt')
call(pyexe + ' test/script/gen_op_result.py --input data/random_positive_input_3_4_32_32.txt --output data/sqrt_2d.txt ' \
    '--op sqrt')

## Softmax
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4096.txt --output data/softmax_1d.txt ' \
    '--op softmax')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/softmax_2d.txt ' \
    '--op softmax')

## FC and Gemm
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4096.txt ' \
    '--output data/linear_1d_w256_4096_b_256.txt ' \
    '--op linear --W data/random_weight_256_4096.txt --b data/random_bias_256.txt')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt ' \
    '--output data/linear_2d_w256_4096_b_256.txt ' \
    '--op linear --W data/random_weight_256_4096.txt --b data/random_bias_256.txt')

## MaxPool2d
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/max_pooling_2d_k2_s2_p0.txt ' \
    '--op max_pooling_2d --ksize 2 --stride 2 --pad 0 --cover_all 0')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/max_pooling_2d_k3_s2_p0.txt ' \
    '--op max_pooling_2d --ksize 3 --stride 2 --pad 0 --cover_all 0')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/max_pooling_2d_k3_s2_p1.txt ' \
    '--op max_pooling_2d --ksize 3 --stride 2 --pad 1 --cover_all 0')

## AveragePool2d
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/average_pooling_2d_k2_s2_p0.txt ' \
    '--op average_pooling_2d --ksize 2 --stride 2 --pad 0')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/average_pooling_2d_k3_s2_p0.txt ' \
    '--op average_pooling_2d --ksize 3 --stride 2 --pad 0')
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/average_pooling_2d_k3_s2_p1.txt ' \
    '--op average_pooling_2d --ksize 3 --stride 2 --pad 1')

## GlobalMaxPool
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/global_max_pooling_2d.txt ' \
    '--op max_pooling_2d --ksize 32 --stride 1 --pad 0 --cover_all 0')

## GlobalAveragePool
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/global_average_pooling_2d.txt ' \
    '--op average_pooling_2d --ksize 32 --stride 1 --pad 0')


## Conv2d
def conv_data(v1, v2, v3):
    ksize = str(v1)
    stride = str(v2)
    pad = str(v3)

    # Without bias
    call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt ' \
        '--output data/convolution_2d_w5_4_'+ksize+'_'+ksize+'_k'+ksize+'_s'+stride+'_p'+pad+'.txt ' \
        '--op convolution_2d --ksize '+ksize+' --stride '+stride+' --pad '+pad+' ' \
        '--W data/random_weight_5_4_'+ksize+'_'+ksize+'.txt')

    # With bias
    call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt ' \
        '--output data/convolution_2d_w5_4_'+ksize+'_'+ksize+'_k'+ksize+'_s'+stride+'_p'+pad+'_with_bias.txt ' \
        '--op convolution_2d --ksize '+ksize+' --stride '+stride+' --pad '+pad+' ' \
        '--W data/random_weight_5_4_'+ksize+'_'+ksize+'.txt ' \
        '--b data/random_bias_5.txt')


conv_data(1, 1, 0)
conv_data(2, 1, 0)
conv_data(2, 1, 1)
conv_data(2, 2, 0)
conv_data(2, 2, 1)
conv_data(3, 1, 1)
conv_data(3, 2, 0)
conv_data(3, 2, 1)

## BatchNorm
call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/batch_normalization.txt ' \
    '--op fixed_batch_normalization --gamma data/random_gamma_4.txt --beta data/random_beta_4.txt ' \
    '--mean data/random_mean_4.txt --var data/random_var_4.txt --eps 1e-5')

## Add
call(
    pyexe +
    ' test/script/gen_add_result.py --input_a data/random_input_3_4096.txt --input_b data/random_input_3_4096.txt --output data/add_1d.txt'
)
call(
    pyexe +
    ' test/script/gen_add_result.py --input_a data/random_input_3_4_32_32.txt --input_b data/random_input_3_4_32_32.txt --output data/add_2d.txt'
)

## Concat
call(
    pyexe +
    ' test/script/gen_concat_result.py --inputs data/random_input_3_4096.txt data/random_input_3_4096.txt --axis 0 --output data/concat_1d_6_4096.txt'
)
call(
    pyexe +
    ' test/script/gen_concat_result.py --inputs data/random_input_3_4096.txt data/random_input_3_4096.txt --axis 1 --output data/concat_1d_3_8192.txt'
)
call(
    pyexe +
    ' test/script/gen_concat_result.py --inputs data/random_input_3_4096.txt data/random_input_3_4096.txt data/random_input_3_4096.txt --axis 0 --output data/concat_1d_9_4096.txt'
)
call(
    pyexe +
    ' test/script/gen_concat_result.py --inputs data/random_input_3_4096.txt data/random_input_3_4096.txt data/random_input_3_4096.txt --axis 1 --output data/concat_1d_3_12288.txt'
)


## Deconv
def deconv_data(v1, v2, v3):
    ksize = str(v1)
    stride = str(v2)
    pad = str(v3)

    # Without bias
    call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt ' \
        '--output data/deconvolution_2d_w4_5_'+ksize+'_'+ksize+'_k'+ksize+'_s'+stride+'_p'+pad+'.txt ' \
        '--op deconvolution_2d --W data/random_weight_4_5_'+ksize+'_'+ksize+'.txt ' \
        '--ksize '+ksize+' --stride '+stride+' --pad '+pad+' --cover_all 0')

    # With bias
    call(pyexe + ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt ' \
        '--output data/deconvolution_2d_w4_5_'+ksize+'_'+ksize+'_k'+ksize+'_s'+stride+'_p'+pad+'_with_bias.txt ' \
        '--op deconvolution_2d --W data/random_weight_4_5_'+ksize+'_'+ksize+'.txt ' \
        '--ksize '+ksize+' --stride '+stride+' --pad '+pad+' --cover_all 0 ' \
        '--b data/random_bias_4.txt')


deconv_data(1, 1, 0)
deconv_data(2, 1, 0)
deconv_data(2, 1, 1)
deconv_data(2, 2, 0)
deconv_data(2, 2, 1)
deconv_data(3, 1, 1)
deconv_data(3, 2, 0)
deconv_data(3, 2, 1)


## LRN
def lrn_data(alpha, beta, bias, size):
    call(
        pyexe +
        ' test/script/gen_op_result.py --input data/random_input_3_4_32_32.txt --output data/lrn_alpha{alpha}_beta{beta}_bias{bias}_size{size}.txt --op local_response_normalization --alpha {alpha} --beta {beta} --n {size} --k {bias}'.
        format(
            alpha=str(alpha), beta=str(beta), bias=str(bias), size=str(size)))


lrn_data(0.0001, 0.75, 1, 1)
lrn_data(0.0001, 0.75, 1, 2)
lrn_data(0.0001, 0.75, 1, 3)
lrn_data(0.0001, 0.75, 1, 4)
lrn_data(0.0001, 0.75, 2, 1)
lrn_data(0.0001, 0.75, 2, 2)
lrn_data(0.0001, 0.75, 2, 3)
lrn_data(0.0001, 0.75, 2, 4)
