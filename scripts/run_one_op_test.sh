backend=$1
op=$2
./test/menoh_test --gtest_filter=OperatorTest."$backend"_"$op"\*
