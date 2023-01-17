include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

file(GLOB CORE_SRC 
    src/*.cpp
    src/*.cu
)

add_executable(solver_trans_intra_2ac src/solver_trans_intra_2ac.cpp)
add_executable(solver_trans_inter_2ac src/solver_trans_inter_2ac.cpp)