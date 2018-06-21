import sys
import subprocess

def call( cmd ):
    """ call command line.
    """

    print( cmd )
    p = subprocess.Popen(cmd, shell=True)
    ret = p.wait()
    print('')

pyexe = 'python'

# Generate input data
call(pyexe + ' test/script/make_random_data.py 1 3 224 224 --output data/random_input_1_3_224_224.txt')
call(pyexe + ' test/script/make_random_data.py 2 3 224 224 --output data/random_input_2_3_224_224.txt')
call(pyexe + ' test/script/make_random_data.py 4 3 224 224 --output data/random_input_4_3_224_224.txt')
call(pyexe + ' test/script/make_random_data.py 8 3 224 224 --output data/random_input_8_3_224_224.txt')
call(pyexe + ' test/script/make_random_data.py 16 3 224 224 --output data/random_input_16_3_224_224.txt')
call(pyexe + ' test/script/make_random_data.py 32 3 224 224 --output data/random_input_32_3_224_224.txt')
call(pyexe + ' test/script/make_random_data.py 64 3 224 224 --output data/random_input_64_3_224_224.txt')
