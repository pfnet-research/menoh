#!/bin/bash

# check if variables have values
test -n "${PLATFORM}" || { echo "PLATFORM can't be empty" 1>&2; exit 1; }

export PLATFORM_DIR=${TRAVIS_BUILD_DIR}/.travis/${PLATFORM}

if [ "$TRAVIS_OS_NAME" == "linux" ]; then
    # Run some setup procedures (basically it should be done in build.sh)
    true
elif [ "$TRAVIS_OS_NAME" == "osx" ]; then
    # Run some setup procedures (basically it should be done in build.sh)
    true
fi

# Dispatch to platform specific build script
if [ -e "${PLATFORM_DIR}/build.sh" ]; then
    /bin/bash -ex ${PLATFORM_DIR}/build.sh
    result=$?
    exit ${result}
else
    echo 'The specified platform not found: '${PLATFORM} 1>&2
    exit 1
fi
