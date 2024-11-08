cmake_minimum_required(VERSION 3.13)

project(MX_API)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
option(TEST_CASCADE "Option description" OFF)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_compile_definitions(OS_LINUX)
endif()

get_filename_component(MX_API_HOME_DIR "." REALPATH)
message(STATUS "MX_API_HOME_DIR  set to " ${MX_API_HOME_DIR})
get_filename_component(MX_API_DIR "mx_accl" REALPATH)
message(STATUS "MX_API_DIR  set to " ${MX_API_DIR})
get_filename_component(MX_API_BUILD_DIR "build" REALPATH)
message(STATUS "MX_API_BUILD_DIR  set to " ${MX_API_BUILD_DIR})

set(CMAKE_X86_FLAGS_BASE "-mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mfxsr -mcx16 -msahf -mpclmul")
set(CMAKE_X86_FLAGS_AVX2 "-mavx -mavx2 -mfma -mbmi -mbmi2 -maes -mf16c -mfsgsbase -mlzcnt -mmovbe -mxsave")

# Option to select build type (Release or Debug)
set(BUILD_TYPE "Release" CACHE STRING "Build type (Release or Debug or Packaging)")
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
endif()

if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Packaging")
  find_package(GTest REQUIRED)
  find_package(OpenCV REQUIRED)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(
    CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -g -fsanitize=address -fsanitize-recover=all"
  )
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fsanitize-recover=all"
  )
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
  set(
    CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -Wall -Wextra -pipe -fPIC -O3 ${CMAKE_X86_FLAGS_BASE} ${CMAKE_X86_FLAGS_AVX2} -mtune=generic -fopenmp"
  )
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 ${CMAKE_X86_FLAGS_BASE} ${CMAKE_X86_FLAGS_AVX2} -mtune=generic -fopenmp"
  )
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(AARCH64)|(arm64)|(ARM64)")
  set(
    CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -march=armv8-a+simd -fopenmp"
  )
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -march=armv8-a+simd -fopenmp"
  )
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(riscv64)|(RISCV64)|(RV64)|(rv64gc)")
  set(
    CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -march=rv64gc_zba_zbb_zbs -fopenmp"
  )
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe -fPIC -O3 -march=rv64gc_zba_zbb_zbs -fopenmp"
  )
endif()


#define SUBDIR macro
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


enable_testing()

SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "*")

FOREACH(subdir ${SUBDIRS})
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
    ADD_SUBDIRECTORY(${subdir})
  endif()
ENDFOREACH()


