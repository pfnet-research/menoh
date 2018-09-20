#!/bin/bash

# retrieve arguments
while [[ $# != 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --build-dir)
            readonly ARG_BUILD_DIR="$2"
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
test -n "${ARG_BUILD_DIR}" || { echo "--build-dir is not specified" 1>&2; exit 1; }

# install (always)
echo -e "\e[33;1mInstalling libprotobuf\e[0m"

# install to ${DESTDIR}/`--prefix` if it is specified
[ -n "${ARG_DESTDIR}" ] && export DESTDIR=${ARG_DESTDIR}

cd ${ARG_BUILD_DIR}
make install
