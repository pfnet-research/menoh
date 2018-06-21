install(FILES
    menoh/menoh.h
    menoh/menoh.hpp
    DESTINATION "include/menoh")

set(INSTALL_PKGCONFIG_DIR "${CMAKE_INSTALL_PREFIX}/share/pkgconfig" CACHE PATH
    "Installation directory for pkgconfig (.pc) files")
configure_file("menoh.pc.in" "menoh.pc" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/menoh.pc" DESTINATION "${INSTALL_PKGCONFIG_DIR}")
