#!/bin/bash

# retrieve arguments
while [[ $# != 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --source-dir)
            ARG_SOURCE_DIR="$2"
            shift 2
            ;;
        --python-executable)
            ARG_PYTHON_EXECUTABLE="$2"
            shift 2
            ;;
        -*)
            err Unknown option \"$1\"
            exit
            ;;
        *)
            break
            ;;

    esac
done

# validate the arguments
test -n "${ARG_SOURCE_DIR}" || { echo "ARG_SOURCE_DIR can't be empty" 1>&2; exit 1; }
test -n "${ARG_PYTHON_EXECUTABLE}" || ARG_PYTHON_EXECUTABLE=python

echo -e "\e[33;1mPreparing data/ for Menoh\e[0m"

cd ${ARG_SOURCE_DIR}
[ -d "data" ] || mkdir -p data

${ARG_PYTHON_EXECUTABLE} retrieve_data.py
${ARG_PYTHON_EXECUTABLE} gen_test_data.py
