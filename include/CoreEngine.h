#ifndef CORE_ENGINE_H
#define CORE_ENGINE_H

#include "VectorStorage.h"
#include "HNSWGraph.h"
#include <shared_mutex>
#include <string>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>

class CoreEngine {
private:
    VectorStorage storage;
    HNSWGraph index;
    mutable std::shared_mutex rw_lock;

    std::string wal_path;
    std::ofstream wal_writer;
    
    // Background Threading for Graph Maintenance
    std::thread bg_worker;
    std::atomic<bool> stop_worker;
    void maintenance_loop(); // The infinite loop the thread runs

    // WAL uses OpCodes now: 0 = Insert, 1 = Delete
    void log_operation_to_disk(uint8_t op_code, size_t id); 

public:
    CoreEngine(size_t dim, size_t capacity, const std::string& log_file);
    ~CoreEngine();

    size_t insert(const std::vector<float>& vec);
    void delete_vector(size_t id);
    
    std::vector<std::pair<float, size_t>> search(const std::vector<float>& query, size_t k) const;
    void recover_from_wal();
};

#endif