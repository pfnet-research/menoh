# SCOPE can be an empty string
macro(menoh_link_libraries TARGET_NAME SCOPE)
    # Link libraries to menoh
    if(LINK_STATIC_LIBGCC)
        target_link_libraries(${TARGET_NAME} ${SCOPE} -static-libgcc)
    endif()
    if(LINK_STATIC_LIBSTDCXX)
        target_link_libraries(${TARGET_NAME} ${SCOPE} -static-libstdc++)
    endif()

    target_link_libraries(${TARGET_NAME} ${SCOPE} onnx) # onnx also contains protobuf

    if(ENABLE_MKLDNN)
        if(NOT ${SCOPE})
            # PUBLIC will add transitive dependencies (`mklml_intel` and `iomp5`) to the link interface
            # Note: change it to PRIVATE after building mkldnn itself
            target_link_libraries(${TARGET_NAME} PUBLIC ${MKLDNN_LIBRARIES})
        else()
            target_link_libraries(${TARGET_NAME} ${MKLDNN_LIBRARIES})
        endif()
    endif()

    if(ENABLE_TENSORRT)
	target_link_libraries(${TARGET_NAME} ${SCOPE} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES})
    endif()
endmacro()
