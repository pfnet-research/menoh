#!/bin/bash -e

# retrieve arguments
while [[ $# != 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --source-dir)
            readonly ARG_SOURCE_DIR="$2"
            shift 2
            ;;
        --dest-dir)
            readonly ARG_DESTDIR="$2"
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

# validate the arguments
test -n "${ARG_SOURCE_DIR}" || { echo "--source-dir is not specified" 1>&2; exit 1; }

# install (always)
echo -e "\e[33;1mInstalling libboost\e[0m"

cd "${ARG_SOURCE_DIR}"
./b2 install $(cat ${ARG_SOURCE_DIR}/b2_opts.txt)
