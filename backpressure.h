#pragma once
#include <atomic>
#include <chrono>
#include <thread>
#include <memory>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <functional>

// 缓存行大小，用于内存对齐
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

// 前向声明
class BackpressureController;
template<typename T> class LockFreeQueue;
template<typename T> class TrackingAllocator;

// 系统内存监控
class SystemMemoryMonitor {
public:
    // 获取当前进程的实际内存使用量（VmRSS）
    static size_t get_current_memory_usage() {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.compare(0, 6, "VmRSS:") == 0) {
                size_t kb = std::stoul(line.substr(6));
                return kb * 1024; // 转换为bytes
            }
        }
        return 0;
    }
    
    // 获取系统总内存
    static size_t get_system_total_memory() {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.compare(0, 9, "MemTotal:") == 0) {
                size_t kb = std::stoul(line.substr(9));
                return kb * 1024; // 转换为bytes
            }
        }
        return 0;
    }
    
    // 获取系统可用内存
    static size_t get_system_available_memory() {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.compare(0, 10, "MemAvailable:") == 0) {
                size_t kb = std::stoul(line.substr(10));
                return kb * 1024; // 转换为bytes
            }
        }
        return 0;
    }
};

// 背压控制器类（增强比例配置功能）
class BackpressureController {
public:
    // 背压级别枚举
    enum class BackpressureLevel {
        NORMAL = 0,
        WARNING = 1,
        CRITICAL = 2,
        EMERGENCY = 3
    };

    // 配置参数结构（支持内存比例配置）
    struct Config {
        double global_memory_ratio;
        double memory_pool_ratio;
        double other_memory_ratio;
        double warning_threshold_ratio;
        double critical_threshold_ratio;
        size_t max_queue_size;
        bool use_system_memory;

        Config() : global_memory_ratio(0.7), memory_pool_ratio(0.4), 
                    other_memory_ratio(0.3), warning_threshold_ratio(0.7), 
                    critical_threshold_ratio(0.9), max_queue_size(500), 
                    use_system_memory(true) {}
    };

    // 背压决策结构
    struct BackpressureDecision {
        bool should_throttle;
        int delay_ms;
        BackpressureLevel level;
        std::string reason;
    };

    // 统计信息结构（新增内存比例相关字段）
    struct Statistics {
        size_t current_memory_usage;     // 当前内存使用 (bytes)
        size_t peak_memory_usage;        // 峰值内存使用 (bytes)
        size_t system_total_memory;      // 系统总内存 (bytes)
        size_t system_available_memory;  // 系统可用内存 (bytes)
        size_t current_queue_size;       // 当前队列大小
        size_t max_queue_size;           // 最大队列大小
        BackpressureLevel current_level; // 当前背压级别
        uint64_t backpressure_triggers;  // 背压触发次数
        uint64_t emergency_stops;        // 紧急停止次数
        bool is_emergency_stopped;       // 是否处于紧急停止状态
        
        // 新增内存比例相关统计
        size_t global_memory_limit;      // 全局内存限制 (bytes)
        size_t memory_pool_limit;        // 内存池限制 (bytes)
        size_t other_memory_limit;       // 其他内存限制 (bytes)
    };

    // 构造函数（使用配置参数初始化）
    BackpressureController(const Config& config = Config());
    
    // 检查背压状态
    BackpressureDecision check_backpressure();
    
    // 更新内存使用统计（仅当不使用系统内存监控时有效）
    void update_memory_usage(size_t usage);
    void add_memory_usage(size_t delta);
    void sub_memory_usage(size_t delta);
    
    // 更新队列大小
    void update_queue_size(size_t size);
    
    // 获取统计信息
    Statistics get_statistics() const;
    
    // 重置紧急停止状态
    void reset_emergency_stop();
    
    // 动态调整最大队列大小
    void adjust_max_queue_size(size_t new_size);
    
    // 动态调整内存比例配置
    void adjust_config(const Config& new_config);
    
    // 获取内存池限制
    size_t get_memory_pool_limit() const;
    
    // 获取全局内存限制
    size_t get_global_memory_limit() const;

private:
    // 初始化内存限制（基于比例配置）
    void initialize_memory_limits();
    
    // 内存使用统计（仅当不使用系统内存监控时有效）
    std::atomic<size_t> current_memory_usage_{0};
    std::atomic<size_t> peak_memory_usage_{0};
    
    // 队列长度统计
    std::atomic<size_t> current_queue_size_{0};
    size_t max_queue_size_;
    
    // 背压阈值配置（基于比例计算）
    size_t memory_warning_threshold_;   // 内存警告阈值 (bytes)
    size_t memory_critical_threshold_;  // 内存临界阈值 (bytes)
    
    // 系统内存相关
    size_t system_total_memory_;        // 系统总内存 (bytes)
    size_t global_memory_limit_;        // 全局内存限制 (bytes)
    size_t memory_pool_limit_;          // 内存池限制 (bytes)
    size_t other_memory_limit_;         // 其他内存限制 (bytes)
    
    // 配置参数
    Config config_;
    
    // 是否使用系统内存监控（替代内部计数器）
    bool use_system_memory_;
    
    // 背压状态
    std::atomic<BackpressureLevel> current_level_{BackpressureLevel::NORMAL};
    std::atomic<bool> emergency_stop_{false};
    
    // 统计信息
    std::atomic<uint64_t> backpressure_triggers_{0};
    std::atomic<uint64_t> emergency_stops_{0};
};

// 内存跟踪分配器
template<typename T>
class TrackingAllocator {
private:
    BackpressureController* controller;
    
public:
    using value_type = T;
    
    explicit TrackingAllocator(BackpressureController* ctrl) : controller(ctrl) {}
    
    template<typename U>
    TrackingAllocator(const TrackingAllocator<U>& other) : controller(other.controller) {}
    
    T* allocate(size_t n) {
        size_t bytes = n * sizeof(T);
        T* ptr = static_cast<T*>(std::malloc(bytes));
        if (!ptr) throw std::bad_alloc();
        
        if (controller) {
            controller->add_memory_usage(bytes);
        }
        return ptr;
    }
    
    void deallocate(T* ptr, size_t n) {
        if (ptr) {
            size_t bytes = n * sizeof(T);
            if (controller) {
                controller->sub_memory_usage(bytes);
            }
            std::free(ptr);
        }
    }
    
    template<typename U>
    bool operator==(const TrackingAllocator<U>& other) const {
        return controller == other.controller;
    }
    
    template<typename U>
    bool operator!=(const TrackingAllocator<U>& other) const {
        return !(*this == other);
    }
};

// 优化的无锁队列实现（修复内存泄漏）
template <typename T>
class LockFreeQueue {
private:
    struct Node {
        T data;
        std::atomic<Node*> next;
        
        Node() : next(nullptr) {}
    };

    alignas(CACHE_LINE_SIZE) std::atomic<Node*> head;
    alignas(CACHE_LINE_SIZE) std::atomic<Node*> tail;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> count;
    const size_t max_size;

    // 垃圾回收机制
    std::atomic<Node*> garbage_head;
    
    // 将节点添加到垃圾列表
    void add_to_garbage(Node* node);
    
    // 清理所有垃圾节点
    void clear_garbage();

public:
    explicit LockFreeQueue(size_t max_size = SIZE_MAX);
    ~LockFreeQueue();

    bool enqueue(const T& value);
    bool dequeue(T& value);
    bool empty() const;
    size_t size() const;
    
    // 定期清理垃圾
    void collect_garbage();
};

// **缺少的类定义开始**

// MonitoredQueue 类 - 带背压控制的队列
template<typename T>
class MonitoredQueue {
private:
    LockFreeQueue<T> queue_;
    BackpressureController* controller_;
    std::atomic<size_t> enqueue_count_{0};
    std::atomic<size_t> dequeue_count_{0};
    std::atomic<size_t> throttled_count_{0};

public:
    explicit MonitoredQueue(size_t max_size, BackpressureController* controller)
        : queue_(max_size), controller_(controller) {}

    // 带背压控制的入队
    bool enqueue(const T& value);
    
    // 出队
    bool dequeue(T& value);
    
    // 获取队列大小
    size_t size() const { return queue_.size(); }
    
    // 是否为空
    bool empty() const { return queue_.empty(); }
    
    // 获取统计信息
    struct Statistics {
        size_t current_size;
        size_t enqueue_count;
        size_t dequeue_count;
        size_t throttled_count;
    };
    
    Statistics get_statistics() const;
    
    // 垃圾回收
    void collect_garbage() { queue_.collect_garbage(); }
};

// ImprovedMemoryPool 类 - 改进的内存池
class ImprovedMemoryPool {
public:
    struct Statistics {
        size_t total_allocated;
        size_t total_freed;
        size_t current_blocks;
        size_t peak_blocks;
        size_t allocation_count;
        size_t deallocation_count;
        size_t pool_hits;
        size_t pool_misses;
    };

private:
    struct Block {
        size_t size;
        bool is_free;
        Block* next;
        alignas(CACHE_LINE_SIZE) char data[];
    };

    BackpressureController* controller_;
    std::mutex mutex_;
    Block* free_blocks_;
    size_t pool_size_;
    size_t block_size_;
    
    // 统计信息
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> total_freed_{0};
    std::atomic<size_t> current_blocks_{0};
    std::atomic<size_t> peak_blocks_{0};
    std::atomic<size_t> allocation_count_{0};
    std::atomic<size_t> deallocation_count_{0};
    std::atomic<size_t> pool_hits_{0};
    std::atomic<size_t> pool_misses_{0};

public:
    ImprovedMemoryPool(size_t pool_size, size_t block_size, BackpressureController* controller);
    ~ImprovedMemoryPool();

    // 分配内存
    void* allocate(size_t size);
    
    // 释放内存
    void deallocate(void* ptr, size_t size);
    
    // 获取统计信息
    Statistics get_statistics() const;
    
    // 清理内存池
    void cleanup();
    
    // 检查内存池状态
    bool is_healthy() const;
};

// ImprovedBatchTask 类 - 改进的批处理任务
template<typename T>
class ImprovedBatchTask {
public:
    using TaskFunction = std::function<void(const std::vector<T>&)>;
    
    struct Config {
        size_t batch_size;
        std::chrono::milliseconds timeout;
        size_t max_pending_batches;
        bool enable_adaptive_batching;
        
        Config() : batch_size(100), timeout(std::chrono::milliseconds(1000)),
                  max_pending_batches(10), enable_adaptive_batching(true) {}
    };
    
    struct Statistics {
        size_t total_items_processed;
        size_t total_batches_processed;
        size_t current_batch_size;
        size_t pending_items;
        size_t dropped_items;
        double average_batch_size;
        std::chrono::milliseconds average_processing_time;
    };

private:
    Config config_;
    TaskFunction task_function_;
    BackpressureController* controller_;
    MonitoredQueue<T>* input_queue_;
    
    std::vector<T> current_batch_;
    std::mutex batch_mutex_;
    std::condition_variable batch_cv_;
    std::atomic<bool> running_{false};
    std::thread worker_thread_;
    
    // 统计信息
    std::atomic<size_t> total_items_processed_{0};
    std::atomic<size_t> total_batches_processed_{0};
    std::atomic<size_t> dropped_items_{0};
    std::atomic<double> total_processing_time_{0.0};

public:
    ImprovedBatchTask(const Config& config, TaskFunction task_func, 
                     BackpressureController* controller, MonitoredQueue<T>* queue);
    ~ImprovedBatchTask();

    // 启动批处理
    void start();
    
    // 停止批处理
    void stop();
    
    // 添加任务项
    bool add_item(const T& item);
    
    // 获取统计信息
    Statistics get_statistics() const;
    
    // 调整配置
    void adjust_config(const Config& new_config);

private:
    // 工作线程函数
    void worker_function();
    
    // 处理当前批次
    void process_current_batch();
    
    // 自适应调整批次大小
    void adaptive_batch_sizing();
};

// **全局变量声明**
extern std::unique_ptr<BackpressureController> g_backpressure_controller;
extern std::unique_ptr<ImprovedMemoryPool> g_memory_pool;
extern std::atomic<bool> g_shutdown_requested;
extern std::atomic<size_t> g_total_memory_usage;
extern std::atomic<uint64_t> g_total_processed_items;

// **函数声明**

// 初始化系统
bool initialize_system(const BackpressureController::Config& config = BackpressureController::Config());

// 清理系统
void cleanup_system();

// 内存管理函数
void* tracked_malloc(size_t size);
void tracked_free(void* ptr, size_t size);
void* tracked_realloc(void* ptr, size_t old_size, size_t new_size);

// 系统监控函数
void start_monitoring_thread();
void stop_monitoring_thread();
void print_system_statistics();

// 性能优化函数
void optimize_memory_usage();
void adjust_system_parameters();

// 错误处理函数
void handle_memory_pressure();
void handle_emergency_stop();

// 工具函数
size_t align_size(size_t size, size_t alignment = CACHE_LINE_SIZE);
bool is_power_of_two(size_t n);
size_t next_power_of_two(size_t n);

// 测试和调试函数
void run_memory_stress_test();
void run_performance_benchmark();
void validate_system_integrity();

// 配置管理函数
bool load_config_from_file(const std::string& filename, BackpressureController::Config& config);
bool save_config_to_file(const std::string& filename, const BackpressureController::Config& config);

// 日志函数
void log_info(const std::string& message);
void log_warning(const std::string& message);
void log_error(const std::string& message);
void log_debug(const std::string& message);

//新加的各种实现
BackpressureController::BackpressureController(const Config& config)
    : config_(config),
      use_system_memory_(config.use_system_memory),
      max_queue_size_(config.max_queue_size) {
    system_total_memory_ = SystemMemoryMonitor::get_system_total_memory();
    initialize_memory_limits();
}

void BackpressureController::initialize_memory_limits() {
    global_memory_limit_ = static_cast<size_t>(system_total_memory_ * config_.global_memory_ratio);
    memory_pool_limit_ = static_cast<size_t>(global_memory_limit_ * config_.memory_pool_ratio);
    other_memory_limit_ = static_cast<size_t>(global_memory_limit_ * config_.other_memory_ratio);
    memory_warning_threshold_ = static_cast<size_t>(global_memory_limit_ * config_.warning_threshold_ratio);
    memory_critical_threshold_ = static_cast<size_t>(global_memory_limit_ * config_.critical_threshold_ratio);
}

BackpressureController::BackpressureDecision BackpressureController::check_backpressure() {
    size_t current_memory = use_system_memory_ ? SystemMemoryMonitor::get_current_memory_usage() : current_memory_usage_.load();
    size_t current_queue = current_queue_size_.load();

    if (current_memory > peak_memory_usage_) {
        peak_memory_usage_ = current_memory;
    }

    BackpressureLevel level = BackpressureLevel::NORMAL;
    std::string reason = "Normal";
    int delay_ms = 0;
    bool should_throttle = false;

    if (current_memory >= memory_critical_threshold_) {
        level = BackpressureLevel::EMERGENCY;
        reason = "Memory usage exceeded critical threshold";
        delay_ms = 3000;
        should_throttle = true;
        emergency_stops_++;
        emergency_stop_ = true;
    } else if (current_memory >= memory_warning_threshold_) {
        level = BackpressureLevel::CRITICAL;
        reason = "Memory usage exceeded warning threshold";
        delay_ms = 1000;
        should_throttle = true;
        backpressure_triggers_++;
    } else if (current_queue >= max_queue_size_) {
        level = BackpressureLevel::WARNING;
        reason = "Queue size exceeded maximum";
        delay_ms = 500;
        should_throttle = true;
        backpressure_triggers_++;
    }

    current_level_ = level;

    return BackpressureDecision {
        .should_throttle = should_throttle,
        .delay_ms = delay_ms,
        .level = level,
        .reason = reason
    };
}

void BackpressureController::update_memory_usage(size_t usage) {
    if (!use_system_memory_) {
        current_memory_usage_ = usage;
        if (usage > peak_memory_usage_) {
            peak_memory_usage_ = usage;
        }
    }
}

void BackpressureController::add_memory_usage(size_t delta) {
    if (!use_system_memory_) {
        auto new_usage = current_memory_usage_.fetch_add(delta) + delta;
        if (new_usage > peak_memory_usage_) {
            peak_memory_usage_ = new_usage;
        }
    }
}

void BackpressureController::sub_memory_usage(size_t delta) {
    if (!use_system_memory_) {
        current_memory_usage_.fetch_sub(delta);
    }
}

void BackpressureController::update_queue_size(size_t size) {
    current_queue_size_ = size;
}

BackpressureController::Statistics BackpressureController::get_statistics() const {
    size_t memory_usage = use_system_memory_ ? SystemMemoryMonitor::get_current_memory_usage() : current_memory_usage_.load();
    size_t available_memory = use_system_memory_ ? SystemMemoryMonitor::get_system_available_memory() : 0;

    return Statistics{
        .current_memory_usage = memory_usage,
        .peak_memory_usage = peak_memory_usage_.load(),
        .system_total_memory = system_total_memory_,
        .system_available_memory = available_memory,
        .current_queue_size = current_queue_size_.load(),
        .max_queue_size = max_queue_size_,
        .current_level = current_level_.load(),
        .backpressure_triggers = backpressure_triggers_.load(),
        .emergency_stops = emergency_stops_.load(),
        .is_emergency_stopped = emergency_stop_.load(),
        .global_memory_limit = global_memory_limit_,
        .memory_pool_limit = memory_pool_limit_,
        .other_memory_limit = other_memory_limit_
    };
}

void BackpressureController::reset_emergency_stop() {
    emergency_stop_ = false;
}

void BackpressureController::adjust_max_queue_size(size_t new_size) {
    max_queue_size_ = new_size;
}

void BackpressureController::adjust_config(const Config& new_config) {
    config_ = new_config;
    initialize_memory_limits();
}

size_t BackpressureController::get_memory_pool_limit() const {
    return memory_pool_limit_;
}

size_t BackpressureController::get_global_memory_limit() const {
    return global_memory_limit_;
}

// === LockFreeQueue Implementation ===

template<typename T>
LockFreeQueue<T>::LockFreeQueue(size_t max_size_)
    : max_size(max_size_), count(0) {
    Node* dummy = new Node();
    head = tail = dummy;
    garbage_head = nullptr;
}

template<typename T>
LockFreeQueue<T>::~LockFreeQueue() {
    T tmp;
    while (dequeue(tmp)) {}
    clear_garbage();
    delete head.load();
}

template<typename T>
bool LockFreeQueue<T>::enqueue(const T& value) {
    if (count.load(std::memory_order_acquire) >= max_size) return false;
    Node* node = new Node();
    node->data = value;
    node->next.store(nullptr);

    Node* prev_tail = tail.exchange(node);
    prev_tail->next.store(node);
    count.fetch_add(1);
    return true;
}

template<typename T>
bool LockFreeQueue<T>::dequeue(T& value) {
    Node* old_head = head.load();
    Node* next = old_head->next.load();
    if (!next) return false;

    value = next->data;
    head.store(next);
    add_to_garbage(old_head);
    count.fetch_sub(1);
    return true;
}

template<typename T>
bool LockFreeQueue<T>::empty() const {
    return count.load() == 0;
}

template<typename T>
size_t LockFreeQueue<T>::size() const {
    return count.load();
}

template<typename T>
void LockFreeQueue<T>::add_to_garbage(Node* node) {
    node->next.store(garbage_head.load());
    while (!garbage_head.compare_exchange_weak(node->next, node)) {}
}

template<typename T>
void LockFreeQueue<T>::clear_garbage() {
    Node* node = garbage_head.exchange(nullptr);
    while (node) {
        Node* next = node->next.load();
        delete node;
        node = next;
    }
}

template<typename T>
void LockFreeQueue<T>::collect_garbage() {
    clear_garbage();
}

// === MonitoredQueue Implementation ===

template<typename T>
bool MonitoredQueue<T>::enqueue(const T& value) {
    if (controller_) {
        auto decision = controller_->check_backpressure();
        if (decision.should_throttle) {
            throttled_count_++;
            std::this_thread::sleep_for(std::chrono::milliseconds(decision.delay_ms));
        }
    }
    bool result = queue_.enqueue(value);
    if (result) enqueue_count_++;
    controller_->update_queue_size(queue_.size());
    return result;
}

template<typename T>
bool MonitoredQueue<T>::dequeue(T& value) {
    bool result = queue_.dequeue(value);
    if (result) dequeue_count_++;
    controller_->update_queue_size(queue_.size());
    return result;
}

template<typename T>
typename MonitoredQueue<T>::Statistics MonitoredQueue<T>::get_statistics() const {
    return Statistics{
        .current_size = queue_.size(),
        .enqueue_count = enqueue_count_.load(),
        .dequeue_count = dequeue_count_.load(),
        .throttled_count = throttled_count_.load()
    };
}

// === ImprovedBatchTask Implementation ===

template<typename T>
ImprovedBatchTask<T>::ImprovedBatchTask(const Config& config, TaskFunction task_func, BackpressureController* controller, MonitoredQueue<T>* queue)
    : config_(config), task_function_(task_func), controller_(controller), input_queue_(queue) {}

template<typename T>
ImprovedBatchTask<T>::~ImprovedBatchTask() {
    stop();
}

template<typename T>
void ImprovedBatchTask<T>::start() {
    running_ = true;
    worker_thread_ = std::thread(&ImprovedBatchTask::worker_function, this);
}

template<typename T>
void ImprovedBatchTask<T>::stop() {
    running_ = false;
    batch_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

template<typename T>
bool ImprovedBatchTask<T>::add_item(const T& item) {
    return input_queue_->enqueue(item);
}

template<typename T>
void ImprovedBatchTask<T>::adjust_config(const Config& new_config) {
    config_ = new_config;
}

template<typename T>
typename ImprovedBatchTask<T>::Statistics ImprovedBatchTask<T>::get_statistics() const {
    return Statistics {
        .total_items_processed = total_items_processed_.load(),
        .total_batches_processed = total_batches_processed_.load(),
        .current_batch_size = current_batch_.size(),
        .pending_items = input_queue_ ? input_queue_->size() : 0,
        .dropped_items = dropped_items_.load(),
        .average_batch_size = total_batches_processed_ > 0 ? static_cast<double>(total_items_processed_) / total_batches_processed_ : 0.0,
        .average_processing_time = std::chrono::milliseconds(static_cast<size_t>(total_processing_time_ / std::max<size_t>(1, total_batches_processed_)))
    };
}

template<typename T>
void ImprovedBatchTask<T>::worker_function() {
    while (running_) {
        std::vector<T> batch;
        T item;

        auto start_time = std::chrono::steady_clock::now();
        while (batch.size() < config_.batch_size && input_queue_->dequeue(item)) {
            batch.push_back(item);
        }

        if (!batch.empty()) {
            auto begin = std::chrono::steady_clock::now();
            task_function_(batch);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end - begin;
            total_processing_time_.fetch_add(elapsed.count() * 1000.0, std::memory_order_relaxed);
            total_batches_processed_++;
            total_items_processed_ += batch.size();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

template<typename T>
void ImprovedBatchTask<T>::process_current_batch() {
    task_function_(current_batch_);
    total_batches_processed_++;
    total_items_processed_ += current_batch_.size();
    current_batch_.clear();
}

template<typename T>
void ImprovedBatchTask<T>::adaptive_batch_sizing() {
    // 可选：根据延迟与负载调整 config_.batch_size
}
