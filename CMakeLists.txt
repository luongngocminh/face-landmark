cmake_minimum_required(VERSION 3.10)
project(custom_face_landmark VERSION 0.1.0 LANGUAGES CXX)

# Ensure we're using Emscripten
if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "This project is configured for WebAssembly/Emscripten only. Please use emcmake.")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set necessary flags for Emscripten/WASM threading
# Add -pthread flag for all compilation (this enables atomics and bulk-memory)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pthread -s USE_PTHREADS=1 --bind -s ENVIRONMENT='web,worker'")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -pthread -s USE_PTHREADS=1 --bind -s ENVIRONMENT='web,worker'")

# Include directories for the main project
include_directories(include)

# Include the third-party NCNN library headers
include_directories(${CMAKE_SOURCE_DIR}/third-party/ncnn/include/ncnn)

# Define NCNN library for Emscripten
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES
  IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/third-party/ncnn/lib/wasm/libncnn.a)

# Add library source files
file(GLOB_RECURSE LIB_SOURCES "lib/*.cpp")

# Build the static library
add_library(${PROJECT_NAME} STATIC ${LIB_SOURCES})
target_link_libraries(${PROJECT_NAME} ncnn)

# WebAssembly specific settings
set(CMAKE_EXECUTABLE_SUFFIX ".js")

# WASM build options with pthread support
set(WASM_COMPILE_FLAGS "-pthread -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_NAME=FaceLandmarkModule -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4 -s ENVIRONMENT='web,worker'")

# Add standard exported functions
set(WASM_LINK_FLAGS "${WASM_COMPILE_FLAGS} -s EXPORTED_FUNCTIONS=['_malloc','_free'] -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS','getValue','UTF8ToString'] -s SINGLE_FILE=0")

# Add NCNN specific flags
# set(WASM_LINK_FLAGS "${WASM_LINK_FLAGS} -s -s USE_LIBPNG=1 -s USE_LIBJPEG=1")

# If models directory exists, add it to preload files
if(EXISTS ${CMAKE_SOURCE_DIR}/models)
  message(STATUS "Adding models directory to preload files")
  set(WASM_LINK_FLAGS "${WASM_LINK_FLAGS} --preload-file ${CMAKE_SOURCE_DIR}/models@/models")
endif()

# Create the WASM demo app
add_executable(${PROJECT_NAME}_wasm demo/wasm_app/wasm_bindings.cpp)
target_link_libraries(${PROJECT_NAME}_wasm ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}_wasm PROPERTIES LINK_FLAGS "${WASM_LINK_FLAGS}")

# Copy web demo files to build directory
# file(COPY ${CMAKE_SOURCE_DIR}/demo/wasm_app/web DESTINATION ${CMAKE_BINARY_DIR})
