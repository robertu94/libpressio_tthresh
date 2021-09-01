configure_file(GTestCMakeLists.txt.in googletest-download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
)
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
)
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
  ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
  EXCLUDE_FROM_ALL)
include(GoogleTest)

add_executable(
  validate_tthresh
  validate_tthresh.cc
  )
target_link_libraries(
  validate_tthresh
  PRIVATE
    LibPressio::libpressio
    libpressio_tthresh
    gtest_main
    gtest
    gmock
  )
target_compile_definitions(
  validate_tthresh
  PRIVATE
  CMAKE_PROJECT_SOURCE_DIR="${tthresh_SOURCE_DIR}"
  TTHRESH_CLASSIC_CMD="$<TARGET_FILE:tthresh_classic>"
  )
gtest_discover_tests(validate_tthresh)
