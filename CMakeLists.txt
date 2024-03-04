cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tthresh VERSION "0.0.8" LANGUAGES CXX)

#correct was to set a default build type
# https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE 
    STRING "Choose the type to build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()


option(BUILD_SHARED_LIBS "build shared libraries" ON)
include(GNUInstallDirs)
enable_testing()
include(CTest)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(LibPressio REQUIRED)
find_package(std_compat REQUIRED)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)


# tthresh_legacy
add_executable(tthresh_classic
  src/classic/io.hpp
  src/classic/decompress.hpp
  src/classic/Slice.hpp
  src/classic/encode.hpp
  src/classic/tucker.hpp
  src/classic/decode.hpp
  src/classic/compress.hpp
  src/classic/tthresh.hpp
  src/classic/tthresh.cpp
  )
target_link_libraries(tthresh_classic 
  PRIVATE
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  )

# tthresh
add_library(libpressio_tthresh 
  src/pressio/pressio_tthresh.cc
  src/pressio/tucker.hpp
  src/pressio/decode.hpp
  src/pressio/pressio_encode.hpp
  src/pressio/tthresh.hpp
  src/pressio/Slice.hpp
  src/pressio/tthresh_lib.h
  src/pressio/pressio_io.hpp
  src/pressio/pressio_decompress.hpp
  src/pressio/compress.hpp
  )
target_include_directories(libpressio_tthresh PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
target_link_libraries(libpressio_tthresh 
  PRIVATE
  Eigen3::Eigen
  LibPressio::libpressio
  OpenMP::OpenMP_CXX
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libpressio_tthresh.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libpressio_tthresh.pc
  @ONLY
  )

install(TARGETS libpressio_tthresh EXPORT LibPressioTThreshTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libpressio_tthresh.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)
install(EXPORT LibPressioTThreshTargets NAMESPACE LibPressioTThresh:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressioTThresh/)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/LibPressioTThreshConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/LibPressioTThreshConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressioTThresh
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LibPressioTThreshConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_tthresh.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LibPressioTThreshConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LibPressioTThreshConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressioTThresh
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
