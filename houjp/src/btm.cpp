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
    int source_begin;
    int t_id;
    int t_source_begin;
    int t_source_end;
    std::vector<std::vector<double>* >* source_vecs;
    std::vector<std::vector<double>* >* dest_vecs;
    std::vector<double>* source_fs;
    std::vector<double>* source_lens;
    std::vector<double>* dest_lens;
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
    int source_begin = thread_data->source_begin;
    int t_id = thread_data->t_id;
    int t_source_begin = thread_data->t_source_begin;
    int t_source_end = thread_data->t_source_end;
    std::vector<std::vector<double>* >* source_vecs = thread_data->source_vecs;
    std::vector<std::vector<double>* >* dest_vecs = thread_data->dest_vecs;
    std::vector<double>* source_fs = thread_data->source_fs;
    std::vector<double>* source_lens = thread_data->source_lens;
    std::vector<double>* dest_lens = thread_data->dest_lens;
    printf("into thread: source_begin=%d, t_id=%d, t_source_begin=%d, t_source_end=%d\n", source_begin, t_id, t_source_begin, t_source_end);

    for (int i = t_source_begin; i < t_source_end; ++i) {
        for (int j = 0; j < dest_vecs->size(); ++j) {
            double cos_sim = cal_vector_cos_sim((*source_vecs)[source_begin + i], (*dest_vecs)[j], (*source_lens)[source_begin + i], (*dest_lens)[j]);
            if (isnan(cos_sim)) {
                cos_sim = 0.0;
            }
            // 归一化
            cos_sim = 0.5 * cos_sim + 0.5;

            int offset = int(cos_sim * LEN_BINS);
            if (10 == offset) {
                offset -= 1;
            }
            int id_begin = i * (LEN_BINS + 4);
            printf("into thread: t_id=%d, source_begin=%d, t_source_begin=%d, i=%d, j=%d, cos_sim=%f, offset=%d\n", t_id, source_begin, t_source_begin, i, j, cos_sim, offset);
            (*source_fs)[id_begin + offset] += 1.0;
            (*source_fs)[id_begin + LEN_BINS + 0] = std::max((*source_fs)[id_begin + LEN_BINS + 0], cos_sim);
            if (cos_sim > EPS)
                (*source_fs)[id_begin + LEN_BINS + 1] = std::min((*source_fs)[id_begin + LEN_BINS + 1], cos_sim);
            (*source_fs)[id_begin + LEN_BINS + 2] += cos_sim;
            (*source_fs)[id_begin + LEN_BINS + 3] += cos_sim * cos_sim;
        }
        if (9 == ((i - t_source_begin) % 10)) {
            printf("into thread: t_id=%d, source_begin=%d, t_source_begin=%d, index=%d done\n", t_id, source_begin ,t_source_begin, i);
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
    std::cout << "btm <source_btm_vec_fp> <dest_btm_vec_fp> <btm_fs_fp> <n_parts> <id_part>" << std::endl;
}

int main(int argc, char* argv[]) {
    if (5 > argc) {
        print_help();
        return -1;
    }
    std::string source_btm_vec_fp = argv[1];
    std::string dest_btm_vec_fp = argv[2];
    std::string btm_fs_fp = argv[3];
    int n_parts = std::stoi(argv[4]);
    int id_part = std::stoi(argv[5]);

    std::vector<std::vector<double>* >* source_vecs = load_vector_file(source_btm_vec_fp);
    std::vector<std::vector<double>* >* dest_vecs = load_vector_file(dest_btm_vec_fp);
    std::vector<double>* source_lens = cal_vector_length(source_vecs);
    std::vector<double>* dest_lens = cal_vector_length(dest_vecs);

    int source_begin = int(round(1.0 * source_vecs->size() / n_parts * id_part));
    int source_end = int(round(1.0 * source_vecs->size() / n_parts * (id_part + 1)));

    printf("n_parts=%d, id_part=%d, source_begin=%d, source_end=%d\n", n_parts, id_part, source_begin, source_end);

    // num_bins = LEN_BINS, max, min, sum, sum(^2)
    std::vector<double>* source_fs = new std::vector<double>((unsigned long)((source_end - source_begin) * (LEN_BINS + 4)), 0.);
    printf("create fs vector done\n");
    for (size_t i = 0; i < source_end - source_begin; ++i) {
        (*source_fs)[i * (LEN_BINS + 4) + (LEN_BINS + 1)] = 1.0;
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
        int t_source_begin = int(round(1.0 * (source_end - source_begin) / NUM_THREADS * t));
        int t_source_end = int(round(1.0 * (source_end - source_begin) / NUM_THREADS * (t + 1)));
        printf("create thread id=%d, t_begin=%d, t_end=%d\n", t, t_source_begin, t_source_end);

        thread_data[t].source_begin = source_begin;
        thread_data[t].t_id = t;
        thread_data[t].t_source_begin = t_source_begin;
        thread_data[t].t_source_end = t_source_end;
        thread_data[t].source_vecs = source_vecs;
        thread_data[t].dest_vecs = dest_vecs;
        thread_data[t].source_fs = source_fs;
        thread_data[t].source_lens = source_lens;
        thread_data[t].dest_lens = dest_lens;

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

    save_features(source_fs, btm_fs_fp);

    pthread_exit(NULL);
}