
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(YOLO_HPC_Linux)

# -fopenmp: 允许并行多线程
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -fopenmp")

# 3. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. 寻找 OpenCV 库 (Linux 必备)
find_package(OpenCV REQUIRED)

# 5. 头文件路径 (关键！)
# 告诉编译器去哪里找 yolo_model.h 和 yolo_model_v5.h
# ${CMAKE_CURRENT_SOURCE_DIR} 就是你当前放代码的目录
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(yolo_run main.cpp)
target_link_libraries(yolo_run ${OpenCV_LIBS})

add_executable(yolo_bench benchmark.cpp)
target_link_libraries(yolo_bench ${OpenCV_LIBS})

add_executable(matmul_test matmul_test.cpp)

target_compile_options(matmul_test PRIVATE -O3)

target_compile_options(matmul_test PRIVATE -O3 -mavx2 -mfma)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -fopenmp")

add_executable(test_quant test_quant.cpp)
target_compile_options(test_quant PRIVATE -O3 -mavx2 -fopenmp)
target_link_libraries(test_quant ${OpenCV_LIBS})
add_executable(sim_kalman 
    sim_KalmanFilter.cpp
    MicroKalman.cpp
    MicroKalman.h 
    yolo_model.h
)
target_compile_options(sim_kalman PRIVATE -O3 -mavx2)

add_executable(video_track 
    video_track.cpp 
    MicroKalman.cpp 
    yolo_model_v5.h
    yolo_model.h
    
)
target_compile_options(video_track PRIVATE -O3 -mavx2 -fopenmp)
target_link_libraries(video_track ${OpenCV_LIBS})
