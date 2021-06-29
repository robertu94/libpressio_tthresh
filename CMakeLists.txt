cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tthresh VERSION "0.66.3" LANGUAGES CXX)

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

export(TARGETS libpressio_tthresh NAMESPACE PressioTThresh:: FILE PressioTThresh.cmake)
install(TARGETS libpressio_tthresh EXPORT PressioTThreshConfig LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT PressioTThreshConfig NAMESPACE PressioTThresh:: DESTINATION share/PressioTThresh/cmake)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libpressio_tthresh.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
