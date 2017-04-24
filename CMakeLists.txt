cmake_minimum_required(VERSION 2.8)
project(kaggle_quora_question_pairs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    houjp/src/btm.cpp)

add_executable(kaggle_quora_question_pairs ${SOURCE_FILES})