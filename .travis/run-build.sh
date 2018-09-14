#!/bin/bash

echo "travis_fold:start:run-build.sh"
echo -e "\e[33;1mRunning .travis/${PLATFORM}/run-build.sh in ${TRAVIS_REPO_SLUG}\e[0m"

# check if variables have values
test -n "${PLATFORM}" || { echo "PLATFORM does not exist"; exit 1; }

export PLATFORM_DIR=${TRAVIS_BUILD_DIR}/.travis/${PLATFORM}

# "$TRAVIS_OS_NAME" == "linux"
if [[ "$PLATFORM" == "linux-x86" ]] || [[ "$PLATFORM" == "linux-x86_64" ]] || [[ "$PLATFORM" =~ android ]]; then
    test -n "${BUILDENV_IMAGE}" || { echo "BUILDENV_IMAGE does not exist"; exit 1; }

    docker pull ${BUILDENV_IMAGE} || true

    # Run a docker container and map Travis's $HOME to the container's $HOME
    # $HOME:$HOME = /home/travis                     : /home/travis
    #               /home/travis/build               : /home/travis/build
    #               /home/travis/build/<user>/<repo> : /home/travis/build/<user>/<repo> (= ${TRAVIS_BUILD_DIR})
    # TODO: the ownership of files and directories mounted on the container
    export DOCKER_CONTAINER_ID=$(docker run -d -it -v $HOME:$HOME -v /sys/fs/cgroup:/sys/fs/cgroup:ro ${BUILDENV_IMAGE} /bin/bash)

    # Note: You shouldn't do `docker run` with `--privileged /sbin/init`.
    # See https://bugzilla.redhat.com/show_bug.cgi?id=1046469 for the details.

    if [ -z "${DOCKER_CONTAINER_ID}" ]; then
        echo 'Failed to run a Docker container: '${BUILDENV_IMAGE}
        exit 1
    fi

    # Stop the container when run-build.sh exits
    trap '[[ "$DOCKER_CONTAINER_ID" ]] && docker stop ${DOCKER_CONTAINER_ID} && docker rm -v ${DOCKER_CONTAINER_ID}' 0 1 2 3 15

    docker logs ${DOCKER_CONTAINER_ID}

    if [ -e "${PLATFORM_DIR}/build.sh" ]; then
        # Dispatch to the platform specific script
        /bin/bash -ex ${PLATFORM_DIR}/build.sh
    else
        echo 'The specified platform not found: '${PLATFORM} 1>&2
        exit 1
    fi
fi

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
    if [ -e "${PLATFORM_DIR}/build.sh" ]; then
        # Dispatch to the platform specific script
        /bin/bash -ex ${PLATFORM_DIR}/build.sh
    else
        echo 'The specified platform not found: '${PLATFORM} 1>&2
        exit 1
    fi
fi

echo "travis_fold:end:run-build.sh"
