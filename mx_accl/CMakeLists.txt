include_directories("${MX_API_DIR}/include/")

file(GLOB local_src src/*.cpp src/utils/*.cpp)

add_library(mx_accl_static STATIC ${local_src})
target_link_libraries(mx_accl_static memx pthread dl)

add_library(mx_accl SHARED ${local_src})
target_link_libraries(mx_accl memx pthread dl)

target_compile_options(mx_accl PRIVATE -Werror)
target_compile_options(mx_accl_static PRIVATE -Werror)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  target_link_options(mx_accl
    BEFORE PUBLIC PUBLIC -fsanitize=address
  )
  target_link_options(mx_accl_static
    BEFORE PUBLIC PUBLIC -fsanitize=address
  )
endif()

if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Packaging")

  add_executable(mxaccl_dfp_tests tests/mxaccl_dfp_tests.cpp )
  target_link_libraries(mxaccl_dfp_tests mx_accl gtest gtest_main)
  add_test(NAME mxaccl_dfp_tests COMMAND mxaccl_dfp_tests)

  add_executable(mxaccl_user_tests tests/mxaccl_user_tests.cpp )
  target_link_libraries(mxaccl_user_tests mx_accl gtest gtest_main)
  add_test(NAME mxaccl_user_tests COMMAND mxaccl_user_tests)

  add_executable(mxaccl_dataflow_tests tests/mxaccl_dataflow_tests.cpp )
  target_link_libraries(mxaccl_dataflow_tests mx_accl gtest gtest_main)
  add_test(NAME mxaccl_dataflow_tests COMMAND mxaccl_dataflow_tests)

  add_executable(mxaccl_accuracy_tests tests/mxaccl_accuracy_tests.cpp)
  target_link_libraries(mxaccl_accuracy_tests mx_accl gtest gtest_main ${OpenCV_LIBS})
  add_test(NAME mxaccl_accuracy_tests COMMAND mxaccl_accuracy_tests)

  add_executable(mxaccl_manual_threading_accuracy_tests tests/mxaccl_manual_threading_accuracy.cpp)
  target_link_libraries(mxaccl_manual_threading_accuracy_tests mx_accl gtest gtest_main ${OpenCV_LIBS})
  add_test(NAME mxaccl_manual_threading_accuracy_tests COMMAND mxaccl_manual_threading_accuracy_tests)

  add_executable(mxaccl_prepost_tests tests/mxaccl_prepost_tests.cpp )
  target_link_libraries(mxaccl_prepost_tests mx_accl gtest gtest_main dl)
  add_test(NAME mxaccl_prepost_tests COMMAND mxaccl_prepost_tests)

  add_executable(mxaccl_prepost_manual_tests tests/mxaccl_prepost_manual_tests.cpp )
  target_link_libraries(mxaccl_prepost_manual_tests mx_accl gtest gtest_main dl)
  add_test(NAME mxaccl_prepost_manual_tests COMMAND mxaccl_prepost_manual_tests)

  add_executable(mxaccl_utility_tests tests/mxaccl_utility_tests.cpp )
  target_link_libraries(mxaccl_utility_tests mx_accl gtest gtest_main dl)
  add_test(NAME mxaccl_utility_tests COMMAND mxaccl_utility_tests)

endif()
