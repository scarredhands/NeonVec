#ifndef CORE_ENGINE_H
#define CORE_ENGINE_H

#include "VectorStorage.h"
#include "HNSWGraph.h" // <-- Now using HNSW
#include <shared_mutex>
#include <fstream>
#include <string>
#include <vector>

class CoreEngine {
private:
    VectorStorage storage;
    HNSWGraph index; // <-- Updated class
    
    mutable std::shared_mutex rw_lock;
    
    std::string wal_path;
    std::ofstream wal_writer;

    void log_insert_to_disk(size_t id, const std::vector<float>& vec);

public:
    CoreEngine(size_t dim, size_t capacity, const std::string& log_file);
    ~CoreEngine();

    size_t insert(const std::vector<float>& vec);
    std::vector<std::pair<float, size_t>> search(const std::vector<float>& query, size_t k) const;
    std::vector<std::pair<float, size_t>> search_exact(const std::vector<float>& query, size_t k) const;
    
    void recover_from_wal();
};

#endif // CORE_ENGINE_H