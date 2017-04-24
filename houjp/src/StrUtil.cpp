//
// Created by JianpengHou on 2017/2/14.
//

#include "StrUtil.h"

template <typename T>
void StrUtil::split(const std::string &s, char delim, T result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> StrUtil::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

std::vector<double>* StrUtil::str_to_double_vector(const std::string &s, char delim) {
    std::vector<double> *elems = new std::vector<double>;
    std::stringstream ss;
    ss.str(s);
    std::string num;
    while (std::getline(ss, num, delim)) {
        (*elems).push_back(std::stod(num));
    }
    return elems;
}