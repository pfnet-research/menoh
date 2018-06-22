cmake \
    -DENABLE_TEST=ON \
    -DMKLDNN_INCLUDE_DIR="$HOME/mkl-dnn/include" \
    -DMKLDNN_LIBRARY="$HOME/mkl-dnn/lib/libmkldnn.so" \
    ..
