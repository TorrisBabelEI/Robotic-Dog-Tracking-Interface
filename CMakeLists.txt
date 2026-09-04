cmake_minimum_required(VERSION 3.5)
project(Robotic-Dog-Interface-AIMS-Lab)

set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
endif()

# check arch and os
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(ARCH amd64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
  set(ARCH arm64)
endif()

include_directories(externals/unitree_legged_sdk/include)
link_directories(externals/unitree_legged_sdk/lib/cpp/${ARCH})

option(PYTHON_BUILD "build python wrapper" OFF)
option(BUILD_SDK_EXAMPLES "build Unitree SDK example executables" OFF)
if(PYTHON_BUILD)
  add_subdirectory(externals/unitree_legged_sdk/python_wrapper)
endif()

set(EXTRA_LIBS -pthread libunitree_legged_sdk.a)
set(CMAKE_CXX_FLAGS "-O3 -fPIC")
set(CMAKE_CXX_STANDARD 14)

find_package(catkin QUIET)
if(${catkin_FOUND})
  catkin_package(
    INCLUDE_DIRS externals/unitree_legged_sdk/include
    LIBRARIES ${PROJECT_SOURCE_DIR}/externals/unitree_legged_sdk/lib/cpp/${ARCH}/libexternals/unitree_legged_sdk.a
  )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Unitree's example programs are optional deployment extras. The experiment
# executable below still receives hardware support on every Linux build.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND BUILD_SDK_EXAMPLES)
  # one pc one process
  add_executable(example_position externals/unitree_legged_sdk/example/example_position.cpp)
  target_link_libraries(example_position ${EXTRA_LIBS})

  add_executable(example_velocity externals/unitree_legged_sdk/example/example_velocity.cpp)
  target_link_libraries(example_velocity ${EXTRA_LIBS})

  add_executable(example_torque externals/unitree_legged_sdk/example/example_torque.cpp)
  target_link_libraries(example_torque ${EXTRA_LIBS})

  add_executable(example_walk externals/unitree_legged_sdk/example/example_walk.cpp)
  target_link_libraries(example_walk ${EXTRA_LIBS})

  add_executable(example_joystick externals/unitree_legged_sdk/example/example_joystick.cpp)
  target_link_libraries(example_joystick ${EXTRA_LIBS})
endif()

# Low-level experiment runner. Unitree's bundled libraries are Linux binaries,
# so non-Linux hosts build the same CLI with dry-run support only.
add_library(go1_kinematics STATIC src/go1_kinematics.cpp)
target_include_directories(go1_kinematics PUBLIC ${PROJECT_SOURCE_DIR}/src)

add_executable(go1_lowlevel_experiment src/go1_lowlevel_experiment.cpp)
target_link_libraries(go1_lowlevel_experiment go1_kinematics)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(go1_lowlevel_experiment PRIVATE GO1_WITH_SDK=1)
  target_link_libraries(go1_lowlevel_experiment ${EXTRA_LIBS})
else()
  message(WARNING
    "go1_lowlevel_experiment is dry-run-only on ${CMAKE_SYSTEM_NAME}; "
    "build on Linux for Unitree hardware access")
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(go1_kinematics_test test/go1_kinematics_test.cpp)
  target_link_libraries(go1_kinematics_test go1_kinematics)
  add_test(NAME go1_kinematics COMMAND go1_kinematics_test)

  add_test(
    NAME go1_lowlevel_dry_torque
    COMMAND go1_lowlevel_experiment --dry-run --mode torque-sine
            --joint FR_1 --amplitude-nm 0.2 --frequency-hz 0.5
            --duration-s 10 --log ${CMAKE_BINARY_DIR}/go1_dry_torque.csv)
  add_test(
    NAME go1_lowlevel_dry_ground_torque
    COMMAND go1_lowlevel_experiment --dry-run --mode torque-sine-ground
            --amplitude-nm 0.2 --frequency-hz 0.5
            --duration-s 1 --log ${CMAKE_BINARY_DIR}/go1_dry_ground_torque.csv)
  add_test(
    NAME go1_lowlevel_dry_squat
    COMMAND go1_lowlevel_experiment --dry-run --mode squat
            --log ${CMAKE_BINARY_DIR}/go1_dry_squat.csv)
  add_test(
    NAME go1_lowlevel_dry_remote_preflight
    COMMAND go1_lowlevel_experiment --dry-run --mode remote-preflight
            --duration-s 3 --log ${CMAKE_BINARY_DIR}/go1_dry_remote.csv)
  add_test(
    NAME go1_lowlevel_dry_ground_handover
    COMMAND go1_lowlevel_experiment --dry-run --mode ground-handover
            --log ${CMAKE_BINARY_DIR}/go1_dry_handover.csv)
  add_test(
    NAME go1_lowlevel_dry_leg_lift
    COMMAND go1_lowlevel_experiment --dry-run --mode leg-lift --leg auto
            --log ${CMAKE_BINARY_DIR}/go1_dry_leg_lift.csv)
  add_test(
    NAME go1_lowlevel_dry_leg_sequence
    COMMAND go1_lowlevel_experiment --dry-run --mode leg-lift-sequence
            --leg auto --log ${CMAKE_BINARY_DIR}/go1_dry_leg_sequence.csv)
  add_test(
    NAME go1_lowlevel_dry_soft_stop
    COMMAND go1_lowlevel_experiment --dry-run --mode ground-handover
            --inject-soft-stop-s 5
            --log ${CMAKE_BINARY_DIR}/go1_dry_soft_stop.csv)
  add_test(
    NAME go1_lowlevel_dry_remote_panic
    COMMAND go1_lowlevel_experiment --dry-run --mode ground-handover
            --inject-panic-s 5
            --log ${CMAKE_BINARY_DIR}/go1_dry_remote_panic.csv)
  add_test(
    NAME go1_lowlevel_dry_double_ctrl_c
    COMMAND go1_lowlevel_experiment --dry-run --mode ground-handover
            --inject-double-ctrl-c-s 5
            --log ${CMAKE_BINARY_DIR}/go1_dry_double_ctrl_c.csv)
  add_test(
    NAME go1_lowlevel_dry_watchdog
    COMMAND go1_lowlevel_experiment --dry-run --mode ground-handover
            --inject-watchdog-s 5
            --log ${CMAKE_BINARY_DIR}/go1_dry_watchdog.csv)
  add_test(
    NAME go1_lowlevel_rejects_excess_torque
    COMMAND go1_lowlevel_experiment --dry-run --mode torque-sine
            --amplitude-nm 1.1 --log ${CMAKE_BINARY_DIR}/should_not_exist.csv)
  set_tests_properties(go1_lowlevel_rejects_excess_torque PROPERTIES WILL_FAIL TRUE)
  add_test(
    NAME go1_lowlevel_rejects_unconfirmed_preflight
    COMMAND go1_lowlevel_experiment --mode preflight
            --log ${CMAKE_BINARY_DIR}/should_not_exist.csv)
  set_tests_properties(go1_lowlevel_rejects_unconfirmed_preflight
                       PROPERTIES WILL_FAIL TRUE)
  add_test(
    NAME go1_lowlevel_rejects_unconfirmed_remote_preflight
    COMMAND go1_lowlevel_experiment --mode remote-preflight
            --log ${CMAKE_BINARY_DIR}/should_not_exist.csv)
  set_tests_properties(go1_lowlevel_rejects_unconfirmed_remote_preflight
                       PROPERTIES WILL_FAIL TRUE)
  add_test(
    NAME go1_lowlevel_rejects_remote_without_chord
    COMMAND go1_lowlevel_experiment --dry-run --mode remote-preflight
            --duration-s 0.2
            --log ${CMAKE_BINARY_DIR}/go1_dry_remote_without_chord.csv)
  set_tests_properties(go1_lowlevel_rejects_remote_without_chord
                       PROPERTIES WILL_FAIL TRUE)
  add_test(
    NAME go1_lowlevel_rejects_unconfirmed_leg_lift
    COMMAND go1_lowlevel_experiment --mode leg-lift --ground-confirmed
            --log ${CMAKE_BINARY_DIR}/should_not_exist.csv)
  set_tests_properties(go1_lowlevel_rejects_unconfirmed_leg_lift
                       PROPERTIES WILL_FAIL TRUE)
endif()


# install
set(GO1_INSTALL_TARGETS go1_lowlevel_experiment)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND BUILD_SDK_EXAMPLES)
  list(APPEND GO1_INSTALL_TARGETS
       example_position example_velocity example_torque example_walk
       example_joystick)
endif()
install(TARGETS ${GO1_INSTALL_TARGETS} DESTINATION bin/unitree)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  install(DIRECTORY externals/unitree_legged_sdk/lib/cpp/${ARCH}/
    DESTINATION externals/unitree_legged_sdk/lib/unitree
    USE_SOURCE_PERMISSIONS)
endif()
