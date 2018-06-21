cmake \
    -DENABLE_TEST=ON \
    -DProtobuf_PROTOC_EXECUTABLE="$HOME/protoc/bin/protoc" \
    -DProtobuf_INCLUDE_DIR="$HOME/protoc/include" \
    -DProtobuf_LIBRARY="$HOME/protoc/lib/libprotobuf.so" \
    -DProtobuf_PROTOC_LIBRARY="$HOME/protoc/lib/libprotoc.so" \
    -DMKLDNN_INCLUDE_DIR="$HOME/mkl-dnn/include" \
    -DMKLDNN_LIBRARY="$HOME/mkl-dnn/lib/libmkldnn.so" \
    ..
