include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

file(GLOB CORE_SRC 
    src/*.cpp
    src/*.cu
)

add_library(cuda_ransac SHARED ${CORE_SRC})
target_link_libraries(cuda_ransac PUBLIC ${CUDA_LIBRARIES})
target_link_directories(cuda_ransac PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)

add_subdirectory(app)