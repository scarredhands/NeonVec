#include "CoreEngine.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <cstdio> // For removing the old WAL file

int main() {
    size_t dim = 128;
    size_t num_vectors = 10000;
    size_t num_queries = 100;
    size_t k = 10;

    // Delete any old log files so we start fresh for the benchmark
    std::remove("../data/db_wal.bin");

    CoreEngine db(dim, num_vectors, "../data/db_wal.bin");

    std::cout << "--- Starting Vector DB Benchmark ---\n";
    std::cout << "Inserting " << num_vectors << " vectors (dim=" << dim << ")...\n";

    // 1. Generate and insert dataset
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> vec(dim);
        for (size_t d = 0; d < dim; ++d) {
            vec[d] = static_cast<float>(rand()) / RAND_MAX;
        }
        db.insert(vec);
        if ((i + 1) % 2500 == 0) {
            std::cout << "Inserted " << (i + 1) << " vectors.\n";
        }
    }

    // 2. Generate random query vectors
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (size_t i = 0; i < num_queries; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            queries[i][d] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // 3. Establish Ground Truth using exact brute-force search
    std::cout << "\nRunning Exact Search (Ground Truth)...\n";
    std::vector<std::vector<size_t>> ground_truth(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        auto results = db.search_exact(queries[i], k);
        for (const auto& res : results) {
            ground_truth[i].push_back(res.second);
        }
    }

    // 4. Measure ANN Search Performance
    std::cout << "Running ANN Search and calculating metrics...\n";
    size_t total_recall = 0;
    
    // Start high-resolution timer
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_queries; ++i) {
        auto ann_results = db.search(queries[i], k);
        
        // Calculate Recall: How many ANN results are in the Ground Truth set?
        std::unordered_set<size_t> true_set(ground_truth[i].begin(), ground_truth[i].end());
        size_t hits = 0;
        for (const auto& res : ann_results) {
            if (true_set.count(res.second)) {
                hits++;
            }
        }
        total_recall += hits;
    }
    
    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // 5. Final Calculations
    double total_time_sec = std::chrono::duration<double>(end_time - start_time).count();
    double qps = num_queries / total_time_sec;
    double average_recall = static_cast<double>(total_recall) / (num_queries * k) * 100.0;

    std::cout << "\n====================================\n";
    std::cout << "         BENCHMARK RESULTS          \n";
    std::cout << "====================================\n";
    std::cout << "Dataset Size    : " << num_vectors << " vectors\n";
    std::cout << "Total Queries   : " << num_queries << "\n";
    std::cout << "Time Taken      : " << total_time_sec << " seconds\n";
    std::cout << "Throughput (QPS): " << qps << " Queries/Sec\n";
    std::cout << "Recall@" << k << "       : " << average_recall << " %\n";
    std::cout << "====================================\n";

    return 0;
}