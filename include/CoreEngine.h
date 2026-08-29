#ifndef CORE_ENGINE_H
#define CORE_ENGINE_H

#include "VectorStorage.h"
#include "HNSWGraph.h"
#include <shared_mutex>
#include <string>
#include <fstream>
#include <vector>

class CoreEngine {
private:
    VectorStorage storage;
    HNSWGraph index;
    
    // Concurrency control for thread-safe operations
    mutable std::shared_mutex rw_lock;

    // Disk Durability (Write-Ahead Log)
    std::string wal_path;
    std::ofstream wal_writer;
    
    // UPDATED: Now only requires the ID, because it fetches the 8-bit vector internally
    void log_insert_to_disk(size_t id); 

public:
    CoreEngine(size_t dim, size_t capacity, const std::string& log_file);
    ~CoreEngine();

    // Core Database Operations
    size_t insert(const std::vector<float>& vec);
    std::vector<std::pair<float, size_t>> search(const std::vector<float>& query, size_t k) const;

    // Crash Recovery
    void recover_from_wal();
};

#endif