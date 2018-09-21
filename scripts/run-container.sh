#!/bin/bash

# Run a container for building source code and set it's container ID to ${BUILDENV_CONTAINER_ID}

# retrieve arguments
while [[ $# != 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --image)
            ARG_IMAGE="$2"
            shift 2
            ;;
        --work-dir)
            ARG_WORK_DIR="$2"
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

unset BUILDENV_CONTAINER_ID

docker pull ${ARG_IMAGE} || true

# TODO: the ownership of files and directories mounted on the container
export BUILDENV_CONTAINER_ID=$(docker run -d -it -v ${ARG_WORK_DIR}:${ARG_WORK_DIR} -v /sys/fs/cgroup:/sys/fs/cgroup:ro ${ARG_IMAGE} /bin/bash)

# Note: You shouldn't do `docker run` with `--privileged /sbin/init`.
# See https://bugzilla.redhat.com/show_bug.cgi?id=1046469 for the details.

if [ -z "${BUILDENV_CONTAINER_ID}" ]; then
    echo 'Failed to run a Docker container: '${ARG_IMAGE} 1>&2
    exit 1
fi

# Stop the container when run-build.sh exits
trap '[[ "${BUILDENV_CONTAINER_ID}" ]] && docker stop ${BUILDENV_CONTAINER_ID} && docker rm -v ${BUILDENV_CONTAINER_ID}' 0 1 2 3 15

docker logs ${BUILDENV_CONTAINER_ID}
