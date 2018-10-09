# Build scripts for Travis

## Prerequisites
Linux-based platforms requires a container image with the following softwares:

- GCC 5.2 or later
- CMake 3.1 or later
- Python 3.6
- pip
- OpenCV 2.4 or later

Currently it uses [okapies/buildenv](https://hub.docker.com/r/okapies/buildenv/) image for linux-x86_64 platform.

## Architecture
`.travis.yml` -> `run-build.sh` -> `build.sh` -> `install-*.sh` & `build-menoh.sh`

1. `run-build.sh` just calls a platform's `build.sh` for running the actual build workflow
2. `build.sh`
    - For Linux-based platforms, the build workflow is executed *in the container*
        - `build.sh` runs a Docker container at startup before executing the workflow
        - All commands are run via `docker_exec` and `docker_exec_script` functions (defined in `init-build-linux.sh`)
        - You can access `${HOME}` (`/home/travis`) directory from inside the container transparently because it maps `/home/travis` in the Travis environment to container's `/home/travis`
    - For macOS (OSX), the build workflow is executed on the Travis' VM
    - Install the prerequisites
    - Run a build
    - Run a test
3. Release and clean up

## Directory
- `/home/travis` (= `${HOME}`)
    - `/downloads` (cache)
    - `/build`
        - `/protobuf-<ver>` (cache)
        - `/mkl-dnn-<ver>` (cache)
        - `/<user>/<repo>` (= `${TRAVIS_BUILD_DIR}`)
            - `/menoh`
            - `/test`
            - `/cmake`
            - `CMakeLists.txt`
            - ...
            - `/build`
                - `/menoh`
                - `/test`
                - ...

## Cache
```yaml
cache:
  directories:
    - ${HOME}/downloads
    - ${HOME}/build/protobuf-${PROTOBUF_VERSION}
    - ${HOME}/build/mkl-dnn-${MKLDNN_VERSION}
```
