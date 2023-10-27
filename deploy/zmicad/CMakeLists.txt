project(ZmicAD)  # 定义项目名称

# SET(opencv_DIR /home/caoxiatian/.basis/opencv-3.4.16/share/OpenCV)
# find_package(opencv REQUIRED)	# 找到opencv库

set(ENV{PKG_CONFIG_PATH} /home/caoxiatian/.basis/opencv-3.4.16/lib/pkgconfig)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv)
#MESSAGE(STATUS ${OPENCV_LIBRARIES})
if(${OPENCV_FOUND})
    MESSAGE(${PKG_CONFIG_EXECUTABLE})
    MESSAGE(${OPENCV_LIBRARIES})
    MESSAGE(${OPENCV_LINK_LIBRARIES})
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 添加包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/)

# 生成库文件
add_library(zmicad STATIC  read_raw.cpp slice_manager.cpp channeled_norm.cpp pre_process.cpp post_process.cpp)
# add_library(zmicad SHARED  read_raw.cpp slice_manager.cpp channeled_norm.cpp pre_process.cpp post_process.cpp)
target_link_libraries(zmicad ${OpenCV_LIBRARIES})
