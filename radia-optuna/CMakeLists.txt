cmake_minimum_required(VERSION 3.20)
project(radia_optuna_kernels LANGUAGES CXX)
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "64-bit pointer ABI required")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(OPTUNA_BUILD_MATLAB "Link plain MEX only after Python kernel verification" OFF)
option(OPTUNA_LINUX_CUDA "Linux: build the CUDA backend (OFF: CUDA-free CPU profile)" ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.0.2 EXACT CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(WIN32)
  # The Windows build driver compiles the nvcc object before either binding links.
  if(NOT MSVC OR NOT MSVC_VERSION EQUAL 1944)
    message(FATAL_ERROR "Windows MSVC 14.44 baseline required")
  endif()
  if(NOT EXISTS "${OPTUNA_CUDA_OBJECT}" OR NOT EXISTS "${OPTUNA_CUDA_ROOT}/include/cuda_runtime.h")
    message(FATAL_ERROR "Explicit nvcc object and bundled CUDA headers required")
  endif()
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
  find_library(OPTUNA_CUDART NAMES cudart PATHS "${OPTUNA_CUDA_ROOT}/lib/x64" NO_DEFAULT_PATH REQUIRED)
  set_source_files_properties("${OPTUNA_CUDA_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  add_library(optuna_kernels STATIC "${OPTUNA_CUDA_OBJECT}")
  set_target_properties(optuna_kernels PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(optuna_kernels PUBLIC "${OPTUNA_CUDART}")
  target_include_directories(optuna_kernels PUBLIC "${OPTUNA_CUDA_ROOT}/include")
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/object-list.txt" CONTENT "${OPTUNA_CUDA_OBJECT}\n")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT OPTUNA_LINUX_CUDA)
  # CUDA-free CPU profile: the shared kernel source through a host C++ compiler.
  if(OPTUNA_BUILD_MATLAB)
    message(FATAL_ERROR "Linux builds support the Python binding only")
  endif()
  if(DEFINED CMAKE_CUDA_COMPILER OR DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "The CUDA-free Linux profile takes no CUDA settings")
  endif()
  add_library(optuna_kernels STATIC native/optuna_host.cpp)
  set_target_properties(optuna_kernels PROPERTIES POSITION_INDEPENDENT_CODE YES)
  target_compile_definitions(optuna_kernels PUBLIC OPTUNA_HOST_ONLY=1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(OPTUNA_BUILD_MATLAB)
    message(FATAL_ERROR "Linux builds support the Python binding only")
  endif()
  if(NOT IS_ABSOLUTE "${CMAKE_CUDA_COMPILER}" OR NOT IS_ABSOLUTE "${CMAKE_CUDA_HOST_COMPILER}"
     OR NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "Set explicit CUDA compiler, host compiler and architectures")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit 12.8 REQUIRED)
  if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
     OR NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "^12[.]8[.]"
     OR NOT CUDAToolkit_VERSION MATCHES "^12[.]8[.]")
    message(FATAL_ERROR "Linux requires NVIDIA nvcc and CUDA Toolkit 12.8")
  endif()
  add_library(optuna_kernels STATIC native/optuna_cuda.cu)
  set_target_properties(optuna_kernels PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES POSITION_INDEPENDENT_CODE YES)
  target_compile_options(optuna_kernels PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-Xcompiler=-fopenmp>")
  target_link_libraries(optuna_kernels PUBLIC CUDA::cudart)
else()
  message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()
target_link_libraries(optuna_kernels PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(optuna_kernels PUBLIC native)

pybind11_add_module(_optuna_kernels NO_EXTRAS native/optuna_python.cpp native/optuna_python_runtime.cpp)
target_link_libraries(_optuna_kernels PRIVATE optuna_kernels ${CMAKE_DL_LIBS})
target_compile_definitions(_optuna_kernels PRIVATE NOMINMAX)
set_target_properties(_optuna_kernels PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/radia_optuna"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/radia_optuna")

if(OPTUNA_BUILD_MATLAB)
  find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY)
  find_library(OPTUNA_MATLAB_GPU_LIBRARY NAMES gpumexbinder
    PATHS "${Matlab_ROOT_DIR}/extern/lib/win64/microsoft" NO_DEFAULT_PATH REQUIRED)
  matlab_add_mex(NAME optuna_mex SRC native/optuna_mex.cpp native/optuna_matlab_runtime.cpp
    LINK_TO optuna_kernels "${OPTUNA_MATLAB_GPU_LIBRARY}" R2018a)
  target_compile_definitions(optuna_mex PRIVATE NOMINMAX
    _CRT_SECURE_NO_DEPRECATE _SCL_SECURE_NO_DEPRECATE _SECURE_SCL=0)
  target_compile_options(optuna_mex PRIVATE /Zp8 /GR /W3 /EHs /MD /O2 /Oy- /openmp)
  target_include_directories(optuna_mex PRIVATE "${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include")
  set_target_properties(optuna_mex PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/matlab"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/matlab")
endif()
