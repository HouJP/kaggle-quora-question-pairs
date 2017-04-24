cmake_minimum_required(VERSION 2.8)
project(kaggle_quora_question_pairs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    houjp/src/btm.cpp houjp/src/StrUtil.cpp houjp/src/StrUtil.h)

add_executable(btm ${SOURCE_FILES})