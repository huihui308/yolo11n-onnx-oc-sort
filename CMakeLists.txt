cmake_minimum_required(VERSION 3.16)

project(LIVE_OC_SORT_TRACKER VERSION 1.0 LANGUAGES CXX)

# --------------------------------------------
# C++ Standard and Warnings
# --------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /permissive- /EHsc)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --------------------------------------------
# Dependencies
# --------------------------------------------

# OpenCV
find_package(OpenCV REQUIRED)

# Eigen (header-only)
find_package(Eigen3 REQUIRED)

# --------------------------------------------
# ONNX Runtime (Linux/Jetson)
# --------------------------------------------
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE STRING "ONNX Runtime installation root")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

if(NOT EXISTS "${ONNXRUNTIME_LIB}")
    message(FATAL_ERROR "❌ ONNXRuntime .lib file not found: ${ONNXRUNTIME_LIB}")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)


set_property(TARGET onnxruntime::onnxruntime PROPERTY IMPORTED_IMPLIB "${ONNXRUNTIME_LIB}")
set_property(TARGET onnxruntime::onnxruntime PROPERTY IMPORTED_LOCATION "${ONNXRUNTIME_DLL}")
set_property(TARGET onnxruntime::onnxruntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")

# --------------------------------------------
# Sources
# --------------------------------------------

file(GLOB SRC_LIST CONFIGURE_DEPENDS src/*.cpp)
list(FILTER SRC_LIST EXCLUDE REGEX ".*/test\\.cpp$")

# --------------------------------------------
# OC Sort Library
# --------------------------------------------

add_library(OCLib STATIC ${SRC_LIST})

target_include_directories(OCLib PUBLIC include)
target_link_libraries(OCLib PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    onnxruntime::onnxruntime
)

# --------------------------------------------
# Executable
# --------------------------------------------

add_executable(live_ocsort_tracker main.cpp)
target_include_directories(live_ocsort_tracker PUBLIC include)
target_link_libraries(live_ocsort_tracker PUBLIC
    OCLib
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
)

# --------------------------------------------
# Logs
# --------------------------------------------

message(STATUS "✔ OpenCV version: ${OpenCV_VERSION}")
message(STATUS "✔ ONNX include dir: ${ONNXRUNTIME_ROOT}/include")
message(STATUS "✔ ONNX library: ${ONNXRUNTIME_LIB}")
