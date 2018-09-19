#!/bin/bash -ex

# retrieve arguments
while [[ $# != 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --CC)
            ARG_CC="$2"
            shift 2
            ;;
        --CPP)
            ARG_CPP="$2"
            shift 2
            ;;
        --CXX)
            ARG_CXX="$2"
            shift 2
            ;;
        --CXXCPP)
            ARG_CXXCPP="$2"
            shift 2
            ;;
        -*)
            echo Unknown option \"$1\" 1>&2
            exit
            ;;
        *)
            break
            ;;
    esac
done

export CC=${ARG_CC}
[ -n "${ARG_CPP}" ] && export CPP=${ARG_CPP} || export CPP="${ARG_CC} -E"
export CXX=${ARG_CXX}
[ -n "${ARG_CXXCPP}" ] && export CXXCPP=${ARG_CXXCPP} || export CXXCPP="${ARG_CXX} -E"

./configure $@
