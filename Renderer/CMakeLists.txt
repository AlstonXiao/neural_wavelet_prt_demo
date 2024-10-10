find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED )
find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED)

include_directories(${OptiX_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(BuildConfig)
include(embed_ptx)

embed_ptx(
  CONST 
  RELOCATABLE
  HEADER DeviceKernelPTX.h
  OUTPUT_TARGET
    DeviceKernel
  FOLDER
    OptixKernels/
  PTX_LINK_LIBRARIES
    OptiX::OptiX
    OptiXToolkit::ShaderUtil
  SOURCES 
    device_functions/devicePrograms.cu 
    device_functions/cubeMap.cu
    device_functions/sgMap.cu
    device_functions/latlongMap.cu
    device_functions/constantColorMap.cu
    device_functions/camera.cu
    device_functions/roughPlastic.cu
    device_functions/disneyBSDF.cu
    device_functions/simple_ggx.cu
    device_functions/cookTorrance.cu
    device_functions/cameraShRender.cu
    device_functions/cameraShTmatrix.cu
    device_functions/LaunchParams.h
    device_functions/perRayData.h
    device_functions/perRayDataUtils.h
    device_functions/mathUtils.h
    device_functions/shaderUtils.h
)
list (GET TCNN_CUDA_ARCHITECTURES -1 TCNN_CUDA_ARCHITECTURE)
message(STATUS "Obtained target architecture from CMake variable TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURE}")
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--std=c++17;--extended-lambda;--expt-relaxed-constexpr;--gpu-architecture=compute_${TCNN_CUDA_ARCHITECTURE}" )
cuda_add_library(toneMap
  toneMap.cu)
target_compile_features(toneMap PRIVATE cxx_std_17)
target_link_libraries(toneMap
  OptiXToolkit::ShaderUtil
)

cuda_add_library(TCNNPass STATIC
  tinycudann/tcnn.cu
	tinycudann/utils_kernels.h
	tinycudann/prt_network.h
	tinycudann/prt_globals.h
)
target_compile_features(TCNNPass PRIVATE cxx_std_17)
target_compile_definitions(TCNNPass PUBLIC ${TCNN_DEFINITIONS})
target_compile_options(TCNNPass PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
set_property(TARGET TCNNPass PROPERTY CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURE})
set_property(TARGET TCNNPass PROPERTY CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURE})
target_include_directories(TCNNPass PUBLIC
	${tcnn_dir}/include
	${tcnn_dir}/dependencies
	${tcnn_dir}/dependencies/cutlass/include
	${tcnn_dir}/dependencies/cutlass/tools/util/include
	${tcnn_dir}/dependencies/fmt/include
)
target_link_libraries(TCNNPass
  OptiXToolkit::ShaderUtil
)

add_executable(PathTracer
  optix7.h
  CUDABuffer.h
  RenderEngine.h
  EnvironmentMap.h
  EnvironmentMap.cpp
  npy.h
  RenderEngine.cpp
  Model.cpp
  Model.h
  main.cpp
  Material.h
  Material.cpp
  Scene.h
  Scene.cpp
  OutputWriter.cpp
  OutputWriter.h
  OutputVisualizer.cpp
  OutputVisualizer.h
  table_dist.cpp
  table_dist.h
  HDRloader.cpp
  HDRloader.h
  textures.h
  textures.cpp
  OptixBackend.cpp
  OptixBackend.h
  frames.h
  denoiser.h
  denoiser.cpp
  files.h
  offlineRenderer.h
  offlineRenderer.cpp
  SHRenderer.h
  SHRenderer.cpp
  )

target_compile_features(PathTracer PRIVATE cxx_std_17)
target_link_libraries(PathTracer
  DeviceKernel
  toneMap
  gdt
  TCNNPass

  tiny-cuda-nn
  # optix dependencies, for rendering
  OptiX::OptiX
  OptiXToolkit::ShaderUtil
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}

  # glfw and opengl, for display
  glfWindow
  glfw
  ${OPENGL_gl_LIBRARY}
  )
