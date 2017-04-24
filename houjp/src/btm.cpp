//
// Created by 侯建鹏 on 2017/4/24.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <pthread.h>
#include <stdio.h>
#include <math.h>
#include "StrUtil.h"

#define NUM_THREADS 30
#define LEN_BINS 10
#define EPS (1e-12)

struct ThreadData {
    int begin;
    int t_id;
    int t_begin;
    int t_end;
    std::vector<std::vector<double>* >* vecs;
    std::vector<double>* fs;
    std::vector<double>* lens;
};

double cal_vector_length(std::vector<double>* vec) {
    double length = 0.0;
    for (size_t i = 0; i < vec->size(); ++i) {
        length += (*vec)[i] * (*vec)[i];
    }
    return sqrt(length);
}

std::vector<double>* cal_vector_length(std::vector<std::vector<double>* >* vecs) {
    std::vector<double>* lens = new std::vector<double>();

    for (size_t i = 0; i < vecs->size(); ++i) {
        lens->push_back(cal_vector_length((*vecs)[i]));
    }
    printf("cal btm vector length done, len(lens)=%d\n", (int)lens->size());
    return lens;
}

double cal_vector_dot(std::vector<double>* v1, std::vector<double>* v2) {
    double dot = 0.0;
    if (v1->size() != v2->size()) {
        return -1.0;
    } else {
        for (size_t i = 0; i < v1->size(); ++i) {
            dot += (*v1)[i] * (*v2)[i];
        }
    }
    return dot;
}

double cal_vector_cos_sim(std::vector<double>* v1, std::vector<double>* v2, double l1, double l2) {
    if (l1 < EPS || l2 < EPS) {
        return 0.0;
    } else {
        return cal_vector_dot(v1, v2) / l1 / l2;
    }
}

void* cos_sim(void* argv) {
    struct ThreadData* thread_data = (struct ThreadData *)argv;
    int begin = thread_data->begin;
    int t_id = thread_data->t_id;
    int t_begin = thread_data->t_begin;
    int t_end = thread_data->t_end;
    std::vector<std::vector<double>* >* vecs = thread_data->vecs;
    std::vector<double>* fs = thread_data->fs;
    std::vector<double>* lens = thread_data->lens;
    printf("into thread: begin=%d, t_id=%d, t_begin=%d, t_end=%d\n", begin, t_id, t_begin, t_end);

    for (int i = t_begin; i < t_end; ++i) {
        for (int j = 0; j < vecs->size(); ++j) {
            double cos_sim = cal_vector_cos_sim((*vecs)[begin + i], (*vecs)[j], (*lens)[begin + i], (*lens)[j]);
            if (isnan(cos_sim)) {
                cos_sim = 0.0;
            }
//            printf("into thread: t_id=%d, begin=%d, t_begin=%d, i=%d, j=%d, cos_sim=%f\n", tid, begin, t_begin, i, j, cos_sim);
            int offset = int(cos_sim * LEN_BINS);
            int id_begin = i * (LEN_BINS + 4);
            (*fs)[id_begin + offset] += 1.0;
            (*fs)[id_begin + LEN_BINS + 0] = std::max((*fs)[id_begin + LEN_BINS + 0], cos_sim);
            if (cos_sim > EPS)
                (*fs)[id_begin + LEN_BINS + 1] = std::min((*fs)[id_begin + LEN_BINS + 1], cos_sim);
            (*fs)[id_begin + LEN_BINS + 2] += cos_sim;
            (*fs)[id_begin + LEN_BINS + 3] += cos_sim * cos_sim;
        }
        if (9 == ((i - t_begin) % 10)) {
            printf("into thread: t_id=%d, begin=%d, t_begin=%d, index=%d done\n", t_id, begin ,t_begin, i);
        }
    }

    pthread_exit(NULL);
}

std::vector<std::vector<double>* >* load_vector_file(std::string fp) {
    std::cout << "load btm vector file: " << fp << std::endl;
    std::ifstream ifs(fp.c_str());
    if (!ifs) {
        std::cout << "file not find: " << fp << std::endl;
        exit(-1);
    }
    std::vector<std::vector<double>* >* vectors = new std::vector<std::vector<double>* >();
    std::string line;
    while (std::getline(ifs, line)) {
        std::vector<double>* vec = StrUtil::str_to_double_vector(line, ' ');
        vectors->push_back(vec);
    }
    return vectors;
}

void save_features(std::vector<double>* fs, std::string fp) {
    std::ofstream out(fp);
    if (out.is_open()) {
        for (int i = 0; i < fs->size(); ++i) {
            out << (*fs)[i];
            if (0 == ((i + 1) % (LEN_BINS + 4))) {
                out << "\n";
            } else {
                out << " ";
            }
        }
        out.close();
    }
}

void print_help() {
    std::cout << "btm <btm_vec_fp> <btm_fs_fp> <n_parts> <id_part>" << std::endl;
}

int main(int argc, char* argv[]) {
    if (5 > argc) {
        print_help();
        return -1;
    }
    std::string btm_vec_fp = argv[1];
    std::string btm_fs_fp = argv[2];
    int n_parts = std::stoi(argv[3]);
    int id_part = std::stoi(argv[4]);

    std::vector<std::vector<double>* >* vecs = load_vector_file(btm_vec_fp);
    std::vector<double>* lens = cal_vector_length(vecs);

    int begin = int(round(1.0 * vecs->size() / n_parts * id_part));
    int end = int(round(1.0 * vecs->size() / n_parts * (id_part + 1)));

    printf("n_parts=%d, id_part=%d, begin=%d, end=%d\n", n_parts, id_part, begin, end);

    // num_bins = LEN_BINS, max, min, sum, sum(^2)
    std::vector<double>* fs = new std::vector<double>((unsigned long)((end - begin) * (LEN_BINS + 4)), 0.);
    printf("create fs vector done\n");
    for (size_t i = 0; i < end - begin; ++i) {
        (*fs)[i * (LEN_BINS + 4) + (LEN_BINS + 1)] = 1.0;
    }

    // data
    struct ThreadData thread_data[NUM_THREADS];

    pthread_t thread[NUM_THREADS];
    pthread_attr_t attr;
    int rc, t;
    void* status;

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for (t = 0; t < NUM_THREADS; ++t) {
        int t_begin = int(round(1.0 * (end - begin) / NUM_THREADS * t));
        int t_end = int(round(1.0 * (end - begin) / NUM_THREADS * (t + 1)));
        printf("create thread id=%d, t_begin=%d, t_end=%d\n", t, t_begin, t_end);

        thread_data[t].begin = begin;
        thread_data[t].t_id = t;
        thread_data[t].t_begin = t_begin;
        thread_data[t].t_end = t_end;
        thread_data[t].vecs = vecs;
        thread_data[t].fs = fs;
        thread_data[t].lens = lens;

        rc = pthread_create(&thread[t], &attr, cos_sim, (void *)&thread_data[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    /* Free attribute and wait for the other threads */
    pthread_attr_destroy(&attr);
    for(t = 0; t < NUM_THREADS; ++t) {
        rc = pthread_join(thread[t], &status);
        if (rc) {
            std::cout << "ERROR; return code from pthread_join() is " << rc << std::endl;
            exit(-1);
        }
        printf("Completed join with thread %d status=%ld\n",t, (long)status);
    }

    save_features(fs, btm_fs_fp);

    pthread_exit(NULL);
}