# 1. 规定 CMake 最低版本
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(YOLO_HPC_Linux)

# 2. 🔥【HPC 灵魂】开启狂暴模式
# -O3: 最高级优化
# -mavx2: 允许使用 CPU 的 AVX2 指令集 (待会手写 Intrinsics 要用)
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

# =========================================================
# 🎯 目标一：YOLO 主程序 (yolo_run)
# =========================================================
# 因为实现都在头文件里，所以只需要编译 main.cpp
# 编译器会自动把 include 进来的头文件展开编译
add_executable(yolo_run main.cpp)
target_link_libraries(yolo_run ${OpenCV_LIBS})

# =========================================================
# 🎯 目标二：HPC 跑分程序 (yolo_bench)
# =========================================================
# 同样，只需要编译 benchmark.cpp
add_executable(yolo_bench benchmark.cpp)
target_link_libraries(yolo_bench ${OpenCV_LIBS})

add_executable(matmul_test matmul_test.cpp)
# 必须开 -O3，否则编译器太笨，看不出 Cache 的差距
target_compile_options(matmul_test PRIVATE -O3)