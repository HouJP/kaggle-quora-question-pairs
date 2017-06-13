//
// Created by JianpengHou on 2017/2/14.
//

#ifndef MULTI_LABEL_GBM_STRUTIL_H
#define MULTI_LABEL_GBM_STRUTIL_H

#include <string>
#include <sstream>
#include <vector>

class StrUtil {
public:
    /**
     * split a string with the delimiter and store the result
     *
     * @param s raw string which will be splited
     * @param delim the delimiter
     * @param result container stored the result
     */
    template <typename T>
    static void split(const std::string &s, char delim, T result);

    /**
     * split a string with the delimeter and the result stored in a vector
     *
     * @param s raw string which will be splited
     * @param delim the delimiter
     * @return vector stored the result
     */
    static std::vector<std::string> split(const std::string &s, char delim);

    /**
     * parse a string with the delimeter and return a vector with double type
     *
     * @param s raw string which will be parsed
     * @param delim the delimiter
     * @return vector with double type
     */
    static std::vector<double> *str_to_double_vector(const std::string &s, char delim);
};


#endif //MULTI_LABEL_GBM_STRUTIL_H
