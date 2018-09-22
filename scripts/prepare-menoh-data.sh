#!/bin/bash -e

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

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

# options that have default value
test -n "${ARG_SOURCE_DIR}" || readonly ARG_SOURCE_DIR="${BASE_DIR}/.."
test -n "${ARG_PYTHON_EXECUTABLE}" || readonly ARG_PYTHON_EXECUTABLE=python

echo -e "\e[33;1mPreparing data for Menoh tests and examples\e[0m"

cd "${ARG_SOURCE_DIR}"
[ -d "data" ] || mkdir -p data

${ARG_PYTHON_EXECUTABLE} scripts/retrieve_data.py
${ARG_PYTHON_EXECUTABLE} scripts/gen_test_data.py
