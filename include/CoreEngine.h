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
    
    std::thread bg_worker;
    std::atomic<bool> stop_worker;
    void maintenance_loop(); 

    void log_operation_to_disk(uint8_t op_code, size_t id); 

public:
    CoreEngine(size_t dim, size_t capacity, const std::string& log_file);
    ~CoreEngine();

    size_t insert(const std::vector<float>& vec);
    void delete_vector(size_t id);
    std::vector<std::pair<float, size_t>> search(const std::vector<float>& query, size_t k) const;
    
    // NEW SNAPSHOT COMPACTION API
    void save_snapshot();
    void load_state(); // Instantly loads snapshot, then replays trailing WAL
};

#endif