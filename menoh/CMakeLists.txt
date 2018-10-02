# Options
option(LINK_STATIC_LIBGCC "Link static libgcc to libmenoh" OFF)
option(LINK_STATIC_LIBSTDCXX "Link static libstdc++ to libmenoh" OFF)

# Note: The libraries can be static (.a) or shared (.so)
if(NOT DEFINED MKLDNN_LIBRARIES)
    message(FATAL_ERROR "MKLDNN_LIBRARIES is not found")
endif()

file(GLOB_RECURSE SOURCES "." "*.cpp")

# Create a object library for generating shared library
add_library(menoh_objlib OBJECT ${SOURCES})
set_target_properties(menoh_objlib PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(ConfigureMenoh)

# menoh
add_library(menoh SHARED $<TARGET_OBJECTS:menoh_objlib>)
if(NOT APPLE AND NOT MSVC)
    # Remove private symbols (Note: it works in MINGW but not in MSVC)
    set_property(
        TARGET menoh APPEND_STRING PROPERTY
            LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/menoh.map")
endif()
menoh_link_libraries(menoh PRIVATE)

# menoh_test_target: only used in `test` subdirectory
add_library(menoh_test_target SHARED $<TARGET_OBJECTS:menoh_objlib>)
menoh_link_libraries(menoh_test_target PRIVATE)

install(TARGETS menoh
    RUNTIME DESTINATION "bin"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib")

configure_file("version.h.in" "version.h")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h" DESTINATION "include/menoh")
