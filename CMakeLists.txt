CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

PROJECT(TP4)

ADD_EXECUTABLE(GC graphCuts.cpp image.cpp image.h maxflow/graph.cpp)
TARGET_LINK_LIBRARIES(GC ${OpenCV_LIBS})
