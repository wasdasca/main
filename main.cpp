// main.cpp

#include <cstring>
#include <csignal>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <array>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <unordered_set>
#include <sys/mman.h>
#include <limits>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <openssl/provider.h>
#include <openssl/conf.h>
#include <openssl/engine.h>
#include <sodium.h>
#include <secp256k1.h>
#include <secp256k1_recovery.h>
#include <libbase58.h>
#include "backpressure.h"

extern "C" {
#include <openssl/sha.h>
#include "KeccakHash.h"
#include "KeccakP-1600-SnP.h"
#include "align.h"
#include "brg_endian.h"
#include "KeccakSponge.h"
#include "config.h"
#include "KeccakP-1600-times4-SnP.h"
#include "SIMD256-config.h"
}


// 内存池配置
static size_t calculate_memory_pool_block_size() {
    size_t system_total = SystemMemoryMonitor::get_system_total_memory();
    if (system_total > static_cast<size_t>(16ULL * 1024 * 1024 * 1024)) {
        return 512 * 1024;
    } else if (system_total > static_cast<size_t>(4ULL * 1024 * 1024 * 1024)) {
        return 256 * 1024;
    } else {
        return 128 * 1024;
    }
}

// 计算内存池最大容量
size_t calculate_memory_pool_max_size(BackpressureController* controller) {
    if (!controller) return 2ULL * 1024 * 1024 * 1024;  // 默认 2GB
    return controller->get_memory_pool_limit();  // 使用背压控制器建议的上限
}

// 配置参数
const size_t HARDWARE_THREADS = std::thread::hardware_concurrency();
const size_t PRODUCER_THREADS = HARDWARE_THREADS > 2 ? HARDWARE_THREADS / 2 : 1;
const size_t CONSUMER_THREADS = HARDWARE_THREADS - PRODUCER_THREADS;
constexpr size_t MONITOR_THREADS = 1;
constexpr size_t BATCH_SIZE = 64;
constexpr size_t MAX_QUEUE_SIZE = 500;

// BIP44路径配置
constexpr uint32_t BIP44_PURPOSE = 0x8000002C;
constexpr uint32_t BIP44_COIN_TYPE = 0x800000C3;
constexpr uint32_t BIP44_ACCOUNT = 0x80000000;
constexpr uint32_t BIP44_CHANGE = 0x00000000;
constexpr uint32_t BIP44_START_INDEX = 0;
constexpr size_t DERIVED_KEYS_PER_SEED = 5;

// BIP32扩展密钥结构
struct BIP32Key {
    uint32_t version;
    uint8_t depth;
    uint32_t parent_fingerprint;
    uint32_t child_number;
    uint8_t chain_code[32];
    uint8_t private_key[32];
    uint8_t public_key[65];
    bool is_private;

    BIP32Key() : version(0), depth(0), parent_fingerprint(0), 
                child_number(0), is_private(false) {
        memset(chain_code, 0, 32);
        memset(private_key, 0, 32);
        memset(public_key, 0, 65);
    }
};

// 内存对齐分配器
template <typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using size_type = std::size_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() = default;

    template <typename U, size_t OtherAlignment>
    AlignedAllocator(const AlignedAllocator<U, OtherAlignment>&) {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        void* ptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) {
        free(p);
    }
};

// Block结构体定义
struct Block {
    void* memory;
    size_t size;
    size_t used;
    std::unique_ptr<Block> next;
    
    Block(size_t block_size) : memory(nullptr), size(block_size), used(0) {
        memory = malloc(size);
        if (!memory) throw std::bad_alloc();
    }
    
    ~Block() {
        if (memory) {
            sodium_memzero(memory, size);
            free(memory);
        }
    }
};

// 改进的线程本地内存池
class ImprovedThreadLocalMemoryPool {
private:
    struct LargeAllocation {
        void* ptr;
        size_t size;
        std::unique_ptr<LargeAllocation> next;
        
        LargeAllocation(void* p, size_t s) : ptr(p), size(s) {}
        
        ~LargeAllocation() {
            if (ptr) {
                sodium_memzero(ptr, size);
                free(ptr);
            }
        }
    };
    
    std::unique_ptr<Block> current_block;
    std::unique_ptr<LargeAllocation> large_allocations;
    const size_t block_size;
    std::atomic<size_t> total_allocated{0};
    
public:
    explicit ImprovedThreadLocalMemoryPool(size_t size = calculate_memory_pool_block_size())
        : block_size(size) {
        current_block = allocate_block();
    }
    
    ~ImprovedThreadLocalMemoryPool() {
        cleanup_large_allocations();
    }

    void* allocate(size_t size) {
        // 检查背压状态
        if (g_backpressure_controller) {
            auto decision = g_backpressure_controller->check_backpressure();
            if (decision.level >= BackpressureController::BackpressureLevel::WARNING) {
                cleanup_large_allocations();
                if (decision.level == BackpressureController::BackpressureLevel::EMERGENCY) {
                    throw std::bad_alloc();
                }
            }
        }

        size_t aligned_size = (size >= 64 || size % 64 == 0) ? size : ((size / 64) + 1) * 64;

        if (aligned_size > block_size / 2) {
            return allocate_large(aligned_size);
        }

        return allocate_small(aligned_size);
    }

    void periodic_cleanup() {
        cleanup_large_allocations();
    }

    size_t get_allocated_size() const {
        return total_allocated.load();
    }

private:
    void* allocate_large(size_t aligned_size) {
         if (aligned_size == 0) return nullptr;
        void* ptr = aligned_alloc(64, aligned_size);
        if (!ptr) {
        g_backpressure_controller->reset_emergency_cleanup();
        throw std::bad_alloc();
    }

        size_t new_total = total_allocated.fetch_add(aligned_size) + aligned_size;
        
        if (g_backpressure_controller) {
            g_backpressure_controller->add_memory_usage(aligned_size);
            if (new_total > g_backpressure_controller->get_memory_pool_limit()) {
                total_allocated.fetch_sub(aligned_size);
                free(ptr);
                throw std::bad_alloc();
            }
        }

        large_allocations = std::make_unique<LargeAllocation>(ptr, aligned_size, std::move(large_allocations));
        return ptr;
    }

    void* allocate_small(size_t aligned_size) {
        Block* suitable_block = find_suitable_block(aligned_size);
        if (suitable_block) {
            return use_existing_block(suitable_block, aligned_size);
        }

        return allocate_new_block(aligned_size);
    }

    Block* find_suitable_block(size_t aligned_size) {
        Block* block = current_block.get();
        while (block) {
            if (block->used + aligned_size <= block->size) {
                return block;
            }
            block = block->next.get();
        }
        return nullptr;
    }

    void* use_existing_block(Block* block, size_t aligned_size) {
        void* ptr = static_cast<char*>(block->memory) + block->used;
        block->used += aligned_size;
        size_t new_total = total_allocated.fetch_add(aligned_size) + aligned_size;
        
        if (g_backpressure_controller) {
            g_backpressure_controller->add_memory_usage(aligned_size);
            if (new_total > g_backpressure_controller->get_memory_pool_limit()) {
                block->used -= aligned_size;
                total_allocated.fetch_sub(aligned_size);
                throw std::bad_alloc();
            }
        }
        return ptr;
    }

    void* allocate_new_block(size_t aligned_size) {
        auto new_block = allocate_block();
        if (!new_block) throw std::bad_alloc();

        new_block->used = aligned_size;
        new_block->next = std::move(current_block);
        current_block = std::move(new_block);
        
        size_t new_total = total_allocated.fetch_add(aligned_size) + aligned_size;
        
        if (g_backpressure_controller) {
            g_backpressure_controller->add_memory_usage(aligned_size);
            if (new_total > g_backpressure_controller->get_memory_pool_limit()) {
                current_block = std::move(current_block->next);
                total_allocated.fetch_sub(aligned_size);
                throw std::bad_alloc();
            }
        }
        
        return static_cast<char*>(current_block->memory);
    }

    std::unique_ptr<Block> allocate_block() {
        try {
            auto block = std::make_unique<Block>(block_size);
            if (g_backpressure_controller) {
                g_backpressure_controller->add_memory_usage(block_size);
            }
            return block;
        } catch (const std::bad_alloc&) {
            throw;
        }
    }
    
    void cleanup_large_allocations() {
        large_allocations.reset(); // unique_ptr会自动清理链表
    }
};

// 全局状态(按缓存行对齐)
struct alignas(CACHE_LINE_SIZE) GlobalState {
    std::atomic<bool> running{true};
    std::atomic<uint64_t> keys_generated{0}; 
    std::atomic<uint64_t> addresses_checked{0}; 
    std::atomic<uint64_t> matches_found{0}; 
    std::unordered_set<std::string> target_addresses;
    mutable std::shared_mutex target_mutex;
    std::string result_file;
    mutable std::mutex file_mutex;
} global_state;

// 信号处理函数
void signal_handler(int signum) {
    global_state.running = false;
    std::cout << "\nThread " << std::this_thread::get_id() << " received signal " << signum 
              << ", initiating shutdown...\n";
}

// 私钥有效性验证
bool is_valid_private_key(const uint8_t* private_key) {
    static const uint8_t max_privkey[32] = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
    };

    for (int i = 0; i < 32; i++) {
        if (private_key[i] != 0) {
            break;
        }
        if (i == 31) return false;
    }

    for (int i = 0; i < 32; i++) {
        if (private_key[i] < max_privkey[i]) return true;
        if (private_key[i] > max_privkey[i]) return false;
    }
    return false;
}

// 私钥转公钥
bool private_to_public(const uint8_t* private_key, uint8_t* public_key, secp256k1_context* ctx) {
    if (!is_valid_private_key(private_key)) {
        return false;
    }

    secp256k1_pubkey pubkey;
    int ret = secp256k1_ec_pubkey_create(ctx, &pubkey, private_key);

    if (ret) {
        size_t pubkey_len = 65;
        secp256k1_ec_pubkey_serialize(ctx, public_key, &pubkey_len, &pubkey, SECP256K1_EC_UNCOMPRESSED);
    }

    return ret != 0;
}

// 从私钥生成公钥
std::vector<uint8_t> private_to_public_avx2(const std::vector<uint8_t>& private_key, secp256k1_context* ctx) {
    std::vector<uint8_t> public_key(65);
    if (!private_to_public(private_key.data(), public_key.data(), ctx)) {
        throw std::runtime_error("Invalid private key or failed to generate public key");
    }
    return public_key;
}

// SHA-3-256
std::vector<uint8_t> sha3_256(const std::vector<uint8_t>& data) {
    Keccak_HashInstance instance;
    std::vector<uint8_t> hash(32);

    if (Keccak_HashInitialize(&instance, 1344, 256, 256, 0x06) != KECCAK_SUCCESS) {
        throw std::runtime_error("SHA-3 initialization failed");
    }

    if (Keccak_HashUpdate(&instance, data.data(), data.size() * 8) != KECCAK_SUCCESS) {
        throw std::runtime_error("SHA-3 update failed");
    }

    if (Keccak_HashFinal(&instance, hash.data()) != KECCAK_SUCCESS) {
        throw std::runtime_error("SHA-3 finalization failed");
    }

    return hash;
}

// 公钥转波场地址
std::string public_to_tron(const std::vector<uint8_t>& public_key) {
    if (public_key.size() != 65 || public_key[0] != 0x04) {
        throw std::invalid_argument("TRON address generation requires uncompressed public key (65 bytes starting with 0x04)");
    }

    std::vector<uint8_t> sha3_digest = sha3_256(
        std::vector<uint8_t>(public_key.begin() + 1, public_key.end()));

    std::vector<uint8_t> address_hash(sha3_digest.end() - 20, sha3_digest.end());

    std::vector<char> encoded(50, 0);
    size_t encoded_len = encoded.size();

    if (!b58check_enc(encoded.data(), &encoded_len, 
                      0x41,  
                      address_hash.data(), address_hash.size())) {
        throw std::runtime_error("Base58Check encoding failed");
    }

    return std::string(encoded.data(), encoded_len);
}

// HMAC - SHA512
std::vector<uint8_t> hmac_sha512(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len) {
    std::vector<uint8_t> result(64);
    unsigned int len = 64;
    if (HMAC(EVP_sha512(), key, key_len, data, data_len, result.data(), &len) == nullptr) {
        throw std::runtime_error("HMAC-SHA512 failed: " + std::string(ERR_error_string(ERR_get_error(), NULL)));
    }
    return result;
}

// 从种子生成主密钥
BIP32Key generate_master_key(const std::vector<uint8_t>& seed, secp256k1_context* ctx) {
    BIP32Key master_key;

    // 计算HMAC-SHA512("Bitcoin seed", seed)
    auto hmac_result = hmac_sha512(reinterpret_cast<const uint8_t*>("Bitcoin seed"), 12, seed.data(), seed.size());

    // 前32字节作为私钥
    memcpy(master_key.private_key, hmac_result.data(), 32);

    // 验证私钥有效性
    if (!is_valid_private_key(master_key.private_key)) {
        throw std::runtime_error("Generated master key is invalid");
    }

    // 后32字节作为链码
    memcpy(master_key.chain_code, hmac_result.data() + 32, 32);

    // 生成对应的公钥
    if (!private_to_public(master_key.private_key, master_key.public_key, ctx)) {
        throw std::runtime_error("Failed to generate public key for master key");
    }

    // 设置主密钥的其他参数
    master_key.version = 0x0488ADE4; // xprv版本
    master_key.depth = 0;
    master_key.parent_fingerprint = 0;
    master_key.child_number = 0;
    master_key.is_private = true;

    return master_key;
}

// 计算密钥指纹
uint32_t calculate_key_fingerprint(const uint8_t* public_key) {
    // 1. 对公钥进行SHA-256哈希
    uint8_t sha256_digest[32];
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create EVP_MD_CTX");
    }

    const EVP_MD* md = EVP_sha256();
    if (!md) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("SHA256 not available");
    }

    if (EVP_DigestInit_ex(ctx, md, nullptr) != 1 ||
        EVP_DigestUpdate(ctx, public_key, 65) != 1 ||
        EVP_DigestFinal_ex(ctx, sha256_digest, nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("SHA256 computation failed");
    }

    EVP_MD_CTX_free(ctx);

    // 2. 对SHA-256结果进行RIPEMD160哈希
    uint8_t ripemd160_digest[20];
    ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create EVP_MD_CTX");
    }

    md = EVP_ripemd160();
    if (!md) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("RIPEMD160 not available");
    }

    if (EVP_DigestInit_ex(ctx, md, nullptr) != 1 ||
        EVP_DigestUpdate(ctx, sha256_digest, 32) != 1 ||
        EVP_DigestFinal_ex(ctx, ripemd160_digest, nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("RIPEMD160 computation failed");
    }

    EVP_MD_CTX_free(ctx);

    // 返回前4字节作为指纹
    return (ripemd160_digest[0] << 24) | 
           (ripemd160_digest[1] << 16) | 
           (ripemd160_digest[2] << 8) | 
            ripemd160_digest[3];
}

// 派生子密钥
BIP32Key derive_child_key(const BIP32Key& parent_key, uint32_t child_index, secp256k1_context* ctx) {
    BIP32Key child_key;
    bool is_hardened = (child_index & 0x80000000) != 0;
    uint8_t data[69];

    if (is_hardened) {
        data[0] = 0x00;
        memcpy(data + 1, parent_key.private_key, 32);
        memcpy(data + 33, &child_index, 4);
    } else {
        memcpy(data, parent_key.public_key, 65);
        memcpy(data + 65, &child_index, 4);
    }

    auto hmac_result = hmac_sha512(parent_key.chain_code, 32, data, is_hardened ? 37 : 69);

    static const uint8_t curve_n[32] = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
    };
    bool il_less_than_n = true;
    for (int i = 0; i < 32; i++) {
        if (hmac_result[i] < curve_n[i]) break;
        if (hmac_result[i] > curve_n[i]) {
            il_less_than_n = false;
            break;
        }
    }
    if (!il_less_than_n) {
        throw std::runtime_error("Derived private key is invalid (IL >= n)");
    }

    if (parent_key.is_private) {
        uint8_t child_private_key[32];
        memcpy(child_private_key, parent_key.private_key, 32);
        if (!secp256k1_ec_seckey_tweak_add(ctx, child_private_key, hmac_result.data())) {
            throw std::runtime_error("Failed to add private keys");
        }
        memcpy(child_key.private_key, child_private_key, 32);
        
        secp256k1_pubkey pubkey;
        if (!secp256k1_ec_pubkey_create(ctx, &pubkey, child_private_key)) {
            throw std::runtime_error("Failed to generate public key for child key");
        }
        size_t pubkey_len = 65;
        secp256k1_ec_pubkey_serialize(ctx, child_key.public_key, &pubkey_len, &pubkey, SECP256K1_EC_UNCOMPRESSED);
        child_key.is_private = true;
    } else {
        secp256k1_pubkey parent_pubkey;
        if (!secp256k1_ec_pubkey_parse(ctx, &parent_pubkey, parent_key.public_key, 65)) {
            throw std::runtime_error("Failed to parse parent public key");
        }
        if (!secp256k1_ec_pubkey_tweak_add(ctx, &parent_pubkey, hmac_result.data())) {
            throw std::runtime_error("Failed to derive child public key");
        }
        size_t pubkey_len = 65;
        secp256k1_ec_pubkey_serialize(ctx, child_key.public_key, &pubkey_len, &parent_pubkey, SECP256K1_EC_UNCOMPRESSED);
        child_key.is_private = false;
    }

    memcpy(child_key.chain_code, hmac_result.data() + 32, 32);

    child_key.version = parent_key.version;
    child_key.depth = parent_key.depth + 1;
    child_key.parent_fingerprint = calculate_key_fingerprint(parent_key.public_key);
    child_key.child_number = child_index;

    return child_key;
}

// 从种子派生HD密钥
std::vector<std::vector<uint8_t>> derive_hd_keys(const std::vector<uint8_t>& seed, secp256k1_context* ctx) {
    std::vector<std::vector<uint8_t>> keys;
    
    try {
        // 生成主密钥
        BIP32Key master_key = generate_master_key(seed, ctx);
        
        // 存储根密钥
        std::vector<uint8_t> root_private_key(32);
        memcpy(root_private_key.data(), master_key.private_key, 32);
        keys.push_back(root_private_key);
        
        // 派生路径基础部分: m/44'/195'/0'/0
        BIP32Key derived_key = master_key;
        
        // 硬化派生: 44'
        derived_key = derive_child_key(derived_key, BIP44_PURPOSE | 0x80000000, ctx);
        
        // 硬化派生: 195' (TRX)
        derived_key = derive_child_key(derived_key, BIP44_COIN_TYPE | 0x80000000, ctx);
        
        // 硬化派生: 0' (账户)
        derived_key = derive_child_key(derived_key, BIP44_ACCOUNT | 0x80000000, ctx);
        
        // 非硬化派生: 0 (外部链)
        derived_key = derive_child_key(derived_key, BIP44_CHANGE, ctx);
        
        // 派生4个子密钥 (索引0-3)
        for (uint32_t i = 0; i < 4; ++i) {
            BIP32Key child_key = derive_child_key(derived_key, i, ctx);
            
            // 提取子私钥
            std::vector<uint8_t> child_private_key(32);
            memcpy(child_private_key.data(), child_key.private_key, 32);
            keys.push_back(child_private_key);
        }
    } catch (const std::exception& e) {
        std::cerr << "HD key derivation error: " << e.what() << std::endl;
        // 安全清理已生成的部分密钥
        for (auto& key : keys) {
            sodium_memzero(key.data(), key.size());
        }
        keys.clear();
        throw;
    }
    
    return keys;
}

// 从熵生成种子
std::vector<uint8_t> generate_seed() {
    std::vector<uint8_t> entropy(32);
    randombytes_buf(entropy.data(), 32);
    
    std::vector<uint8_t> seed(64);
    crypto_generichash(seed.data(), seed.size(), entropy.data(), entropy.size(), NULL, 0);
    sodium_memzero(entropy.data(), entropy.size());
    return seed;
}

// AVX2优化的批量密钥生成
class AVX2KeyGenerator {
private:
    struct ThreadLocalState {
        std::vector<uint8_t> seed_buffer;
        secp256k1_context* secp_context;
        
        ThreadLocalState() : 
            secp_context(secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY)) {
            seed_buffer.resize(64);
        }
        
        ~ThreadLocalState() {
            if (secp_context) {
                secp256k1_context_destroy(secp_context);
            }
            sodium_memzero(seed_buffer.data(), seed_buffer.size());
        }
    };
    
    static thread_local std::unique_ptr<ThreadLocalState> tls_state;
    
public:
    static void generate_batch(ImprovedBatchTask& batch, ImprovedThreadLocalMemoryPool& pool) {
        if (!tls_state) {
            tls_state = std::make_unique<ThreadLocalState>();
        }
        
        size_t keys_added = 0;
        
        while (keys_added < BATCH_SIZE * DERIVED_KEYS_PER_SEED) {
            try {
                // 生成种子
                auto seed = generate_seed();
                
                // 派生HD密钥（1根+4子）
                auto derived_keys = derive_hd_keys(seed, tls_state->secp_context);
                
                // 安全擦除种子
                sodium_memzero(seed.data(), seed.size());
                
                // 存储到批量任务
                for (const auto& key : derived_keys) {
                    if (keys_added >= BATCH_SIZE * DERIVED_KEYS_PER_SEED) break;
                    
                    // 生成对应的公钥
                    auto public_key = private_to_public_avx2(key, tls_state->secp_context);
                    
                    // 存储私钥和公钥
                    batch.private_keys.push_back(std::move(key));
                    batch.public_keys.push_back(std::move(public_key));
                    
                    keys_added++;
                }
            } catch (const std::exception& e) {
                std::cerr << "Key generation error: " << e.what() << " - using fallback\n";
                
                // 生成随机密钥作为fallback
                std::vector<uint8_t> private_key(32);
                do {
                    randombytes_buf(private_key.data(), 32);
                } while (!is_valid_private_key(private_key.data()));
                
                // 生成公钥
                auto public_key = private_to_public_avx2(private_key, tls_state->secp_context);
                
                // 存储结果
                if (keys_added < BATCH_SIZE * DERIVED_KEYS_PER_SEED) {
                    batch.private_keys.push_back(std::move(private_key));
                    batch.public_keys.push_back(std::move(public_key));
                    keys_added++;
                }
            }
        }
        
        // 更新统计
        global_state.keys_generated.fetch_add(keys_added, std::memory_order_relaxed);
    }
    
    static void thread_cleanup() {
        tls_state.reset();
    }
};

// 初始化线程本地变量
thread_local std::unique_ptr<AVX2KeyGenerator::ThreadLocalState> AVX2KeyGenerator::tls_state;

// 公钥有效性验证
bool is_valid_public_key(const std::vector<uint8_t>& key) {
    return key.size() == 65 && key[0] == 0x04;
}

// AVX2优化的地址生成
class AVX2AddressGenerator {
private:
    static thread_local std::unique_ptr<secp256k1_context> tls_secp_context;

public:
    static void process_batch(ImprovedBatchTask& batch) {
        if (!tls_secp_context) {
            tls_secp_context.reset(secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY));
        }

        for (size_t i = 0; i < batch.private_keys.size(); ++i) {
            try {
                // 确保每个public_key都有效
                const auto& private_key = batch.private_keys[i];
                auto public_key = private_to_public_avx2(private_key, tls_secp_context.get());
                
                if (!is_valid_public_key(public_key)) {
                    throw std::runtime_error("Invalid public key format");
                }
                
                batch.public_keys[i] = std::move(public_key);
                
                // 从公钥生成地址
                auto address = public_to_tron(batch.public_keys[i]);
                
                // 存储地址
                batch.addresses.push_back(std::move(address));
                
                // 更新统计
                global_state.addresses_checked.fetch_add(1, std::memory_order_relaxed);
            } catch (const std::exception& e) {
                std::cerr << "Address generation error: " << e.what() << std::endl;
                
                // 生成随机地址作为fallback
                std::vector<uint8_t> random_private(32);
                do {
                    randombytes_buf(random_private.data(), 32);
                } while (!is_valid_private_key(random_private.data()));
                
                auto public_key = private_to_public_avx2(random_private, tls_secp_context.get());
                
                // 确保fallback的public_key有效
                if (!is_valid_public_key(public_key)) {
                    std::cerr << "Fallback public key is still invalid, this should not happen\n";
                    continue;
                }
                
                // 存储fallback结果
                batch.private_keys[i] = std::move(random_private);
                batch.public_keys[i] = std::move(public_key);
                
                // 再次尝试生成地址
                auto address = public_to_tron(batch.public_keys[i]);
                batch.addresses[i] = std::move(address);
                
                // 更新统计
                global_state.addresses_checked.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    
    static void thread_cleanup() {
        tls_secp_context.reset();
    }
};

// 初始化线程本地变量
thread_local std::unique_ptr<secp256k1_context> AVX2AddressGenerator::tls_secp_context;

// 改进的生产者线程，添加背压控制和内存预检查
void improved_producer_thread(MonitoredQueue<std::shared_ptr<ImprovedBatchTask>>& task_queue) {
    // 添加空指针检查
    if (!g_backpressure_controller) {
        std::cerr << "错误: 背压控制器未初始化" << std::endl;
        return;
    }
ImprovedThreadLocalMemoryPool memory_pool;

while (global_state.running) {
    try {
        // 检查背压状态
        auto decision = g_backpressure_controller->check_backpressure();
        
        if (decision.should_throttle) {
            // 根据背压级别设置不同的休眠时间
            switch (decision.level) {
                case BackpressureController::BackpressureLevel::WARNING:
                    std::this_thread::sleep_for(std::chrono::seconds(30));
                    break;
                case BackpressureController::BackpressureLevel::CRITICAL:
                    std::this_thread::sleep_for(std::chrono::minutes(1));
                    break;
                case BackpressureController::BackpressureLevel::EMERGENCY:
                    std::this_thread::sleep_for(std::chrono::minutes(3));
                    break;
                default:
                    if (decision.delay_ms > 0) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(decision.delay_ms));
                    }
                    break;
            }
            
            // 休眠结束后再次检查背压状态
            decision = g_backpressure_controller->check_backpressure();
            if (decision.level >= BackpressureController::BackpressureLevel::CRITICAL) {
                continue;
            }
        }
        
        // 创建新批次前检查内存使用
        size_t estimated_batch_size = BATCH_SIZE * DERIVED_KEYS_PER_SEED * (32 + 65 + 50); // 估算批处理内存需求
        auto stats = g_backpressure_controller->get_statistics();
        size_t memory_limit = g_backpressure_controller->get_memory_pool_limit();
        
        // 如果当前内存使用量 + 预估批次大小超过80%限制，则延迟生产
        if (stats.current_memory_usage + estimated_batch_size > memory_limit * 0.8) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // 创建新批次任务
        auto batch = std::make_shared<ImprovedBatchTask>();
        if (!batch) {
            std::cerr << "错误: 无法创建批处理任务" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // 生成密钥前再次检查内存状态（双重保险）
        decision = g_backpressure_controller->check_backpressure();
        if (decision.level >= BackpressureController::BackpressureLevel::CRITICAL) {
            continue;
        }
        
        // 生成密钥
        try {
            AVX2KeyGenerator::generate_batch(*batch, memory_pool);
            
            // 生成后立即更新背压控制器的内存使用情况
            g_backpressure_controller->add_memory_usage(estimated_batch_size);
            
        } catch (const std::bad_alloc& e) {
            std::cerr << "内存分配失败，触发紧急清理: " << e.what() << std::endl;
            // 触发紧急清理
            g_backpressure_controller->trigger_emergency_cleanup();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        } catch (const std::exception& e) {
            std::cerr << "密钥生成错误: " << e.what() << std::endl;
            continue;
        }
        
        // 尝试加入队列
        int retry_count = 0;
        const int max_retries = 20;
        bool enqueued = false;
        
        while (retry_count < max_retries && global_state.running) {
            if (task_queue.enqueue(batch)) {
                enqueued = true;
                break;
            }
            
            // 检查是否因为背压原因导致入队失败
            auto current_decision = g_backpressure_controller->check_backpressure();
            if (current_decision.level >= BackpressureController::BackpressureLevel::WARNING) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            retry_count++;
        }
        
        // 如果重试太多次，说明消费跟不上，需要减慢生产
        if (!enqueued && global_state.running) {
            // 回滚内存使用记录，因为任务没有成功入队
            g_backpressure_controller->add_memory_usage(-static_cast<int64_t>(estimated_batch_size));
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "生产者错误: " << e.what() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } catch (...) {
        std::cerr << "生产者未知错误" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

// 清理线程本地资源
try {
    AVX2KeyGenerator::thread_cleanup();
} catch (const std::exception& e) {
    std::cerr << "线程清理错误: " << e.what() << std::endl;
}

std::cout << "生产者线程正常退出" << std::endl;

// 改进的消费者线程
void improved_consumer_thread(MonitoredQueue<std::shared_ptr<ImprovedBatchTask>>& task_queue) {
size_t processed_count = 0;
size_t expired_count = 0;

while (global_state.running) {
    std::shared_ptr<ImprovedBatchTask> batch;
    
    // 尝试从队列获取任务
    if (!task_queue.dequeue(batch)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
    }

    // 计算批次的内存使用量（用于后续释放通知）
    size_t batch_memory_usage = batch ? batch->estimate_memory_usage() : 0;

    try {
        // 检查任务是否过期
        if (!batch || batch->get_age_seconds() > 30.0) {
            expired_count++;
            
            if (batch) {
                // 安全清理过期任务
                batch->addresses.clear();
                batch->private_keys.clear();
                batch->addresses.shrink_to_fit();
                batch->private_keys.shrink_to_fit();
                
                // 通知背压控制器释放过期任务的内存
                if (g_backpressure_controller && batch_memory_usage > 0) {
                    g_backpressure_controller->add_memory_usage(-static_cast<int64_t>(batch_memory_usage));
                }
            }
            continue;
        }

        // 处理批次
        try {
            AVX2AddressGenerator::process_batch(*batch);
        } catch (const std::exception& e) {
            std::cerr << "地址生成错误: " << e.what() << std::endl;
            // 即使处理失败，也要释放内存
            if (g_backpressure_controller && batch_memory_usage > 0) {
                g_backpressure_controller->add_memory_usage(-static_cast<int64_t>(batch_memory_usage));
            }
            continue;
        }

        // 检查匹配
        {
            std::shared_lock lock(global_state.target_mutex);
            for (size_t i = 0; i < batch->addresses.size() && i < batch->private_keys.size(); ++i) {
                if (global_state.target_addresses.count(batch->addresses[i]) > 0) {
                    global_state.matches_found.fetch_add(1, std::memory_order_relaxed);
                    
                    // 线程安全的文件写入
                    {
                        std::lock_guard<std::mutex> lock(global_state.file_mutex);
                        
                        std::ofstream outfile(global_state.result_file, std::ios::app);
                        if (outfile.is_open()) {
                            std::ostringstream ss;
                            ss << "匹配发现: " << batch->addresses[i] << "\n";
                            ss << "私钥: ";
                            for (uint8_t byte : batch->private_keys[i]) {
                                ss << std::hex << std::setw(2) << std::setfill('0')
                                   << static_cast<int>(byte);
                            }
                            ss << "\n" << std::string(50, '-') << "\n";
                            outfile << ss.str();
                            outfile.flush(); // 确保立即写入
                        } else {
                            std::cerr << "无法打开结果文件进行写入" << std::endl;
                        }
                    }
                }
            }
        }

        processed_count++;
        
        // 定期报告处理进度，包括内存使用情况
        if (processed_count % 100 == 0) {
            std::cout << "消费者已处理 " << processed_count 
                      << " 个批次，丢弃 " << expired_count << " 个过期批次";
            
            // 添加内存使用统计
            if (g_backpressure_controller) {
                auto stats = g_backpressure_controller->get_statistics();
                std::cout << " [内存: " << (stats.current_memory_usage / 1024 / 1024) << "MB/"
                          << (g_backpressure_controller->get_memory_pool_limit() / 1024 / 1024) << "MB]";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "消费者处理错误: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "消费者未知错误" << std::endl;
    }
    
    // 无论如何都要清理 batch 并通知背压控制器
    if (batch) {
        try {
            batch->addresses.clear();
            batch->private_keys.clear();
            batch->addresses.shrink_to_fit();
            batch->private_keys.shrink_to_fit();
            
            // 通知背压控制器内存释放（正常处理完成的情况）
            if (g_backpressure_controller && batch_memory_usage > 0) {
                g_backpressure_controller->add_memory_usage(-static_cast<int64_t>(batch_memory_usage));
            }
            
        } catch (const std::exception& e) {
            std::cerr << "批次清理错误: " << e.what() << std::endl;
        }
    }
}

// 清理线程本地资源
AVX2AddressGenerator::thread_cleanup();

std::cout << "消费者线程正常退出，共处理 " << processed_count 
          << " 个批次，丢弃 " << expired_count << " 个过期批次" << std::endl;
}

// 改进的内存监控线程
void improved_memory_monitor_thread(ImprovedMemoryPool* memory_pool) {
if (!memory_pool || !g_backpressure_controller) {
std::cerr << "错误: 内存池或背压控制器指针为空" << std::endl;
return;
}

auto last_report_time = std::chrono::steady_clock::now();
auto last_cleanup_time = std::chrono::steady_clock::now();
auto last_brief_report_time = std::chrono::steady_clock::now();

while (global_state.running) {
    try {
        auto stats = g_backpressure_controller->get_statistics();
        auto now = std::chrono::steady_clock::now();
        
        // 每10秒生成简要背压状态报告
        if (std::chrono::duration<double>(now - last_brief_report_time).count() > 10.0) {
            std::stringstream brief_report;
            brief_report << "\n=== 背压系统状态报告 ===\n";
            brief_report << "内存使用: " << stats.current_memory_usage / 1024 / 1024 << "MB/" 
                       << stats.global_memory_limit / 1024 / 1024 << "MB";
            
            // 计算内存使用百分比
            double memory_usage_percent = (static_cast<double>(stats.current_memory_usage) / stats.global_memory_limit) * 100.0;
            brief_report << " (" << std::fixed << std::setprecision(1) << memory_usage_percent << "%)\n";
            
            brief_report << "队列大小: " << stats.current_queue_size << "/" << stats.max_queue_size << "\n";
            brief_report << "当前背压级别: ";
            
            switch (stats.current_level) {
                case BackpressureController::BackpressureLevel::NORMAL: 
                    brief_report << "正常 ✅"; break;
                case BackpressureController::BackpressureLevel::WARNING: 
                    brief_report << "警告 ⚠️"; break;
                case BackpressureController::BackpressureLevel::CRITICAL: 
                    brief_report << "临界 🚨"; break;
                case BackpressureController::BackpressureLevel::EMERGENCY: 
                    brief_report << "紧急 🔥"; break;
            }
            
            brief_report << "\n背压触发次数: " << stats.backpressure_triggers;
            brief_report << " | 匹配发现: " << global_state.matches_found.load() << "\n";
            brief_report << "========================\n";
            
            std::cout << brief_report.str();
            last_brief_report_time = now;
        }
        
        // 每30秒报告一次详细状态
        if (std::chrono::duration<double>(now - last_report_time).count() > 30.0) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "=== 详细系统内存状态报告 ===" << std::endl;
            std::cout << "当前内存使用: " << stats.current_memory_usage / (1024*1024) << " MB" << std::endl;
            std::cout << "峰值内存使用: " << stats.peak_memory_usage / (1024*1024) << " MB" << std::endl;
            std::cout << "全局内存限制: " << stats.global_memory_limit / (1024*1024) << " MB" << std::endl;
            std::cout << "内存池限制: " << stats.memory_pool_limit / (1024*1024) << " MB" << std::endl;
            
            // 详细队列状态
            std::cout << "队列状态: " << stats.current_queue_size << "/" << stats.max_queue_size;
            if (stats.max_queue_size > 0) {
                double queue_usage_percent = (static_cast<double>(stats.current_queue_size) / stats.max_queue_size) * 100.0;
                std::cout << " (" << std::fixed << std::setprecision(1) << queue_usage_percent << "%)";
            }
            std::cout << std::endl;
            
            // 背压系统统计
            std::cout << "背压级别: " << static_cast<int>(stats.current_level);
            switch (stats.current_level) {
                case BackpressureController::BackpressureLevel::NORMAL: 
                    std::cout << " (正常 ✅)"; break;
                case BackpressureController::BackpressureLevel::WARNING: 
                    std::cout << " (警告 ⚠️)"; break;
                case BackpressureController::BackpressureLevel::CRITICAL: 
                    std::cout << " (临界 🚨)"; break;
                case BackpressureController::BackpressureLevel::EMERGENCY: 
                    std::cout << " (紧急 🔥)"; break;
            }
            std::cout << std::endl;
            
            std::cout << "背压触发次数: " << stats.backpressure_triggers << std::endl;
            std::cout << "紧急停止次数: " << stats.emergency_stops << std::endl;
            std::cout << "匹配找到数量: " << global_state.matches_found.load() << std::endl;
            
            // 系统健康度评估
            std::cout << "系统健康度: ";
            if (stats.current_level == BackpressureController::BackpressureLevel::NORMAL && 
                memory_usage_percent < 70.0) {
                std::cout << "良好 💚" << std::endl;
            } else if (stats.current_level <= BackpressureController::BackpressureLevel::WARNING) {
                std::cout << "注意 💛" << std::endl;
            } else {
                std::cout << "警戒 ❤️" << std::endl;
            }
            
            std::cout << std::string(60, '=') << std::endl;
            last_report_time = now;
        }
        
        // 每10秒清理一次内存池
        if (std::chrono::duration<double>(now - last_cleanup_time).count() > 10.0) {
            try {
                memory_pool->periodic_cleanup();
                
                // 清理后再次检查内存状态
                auto post_cleanup_stats = g_backpressure_controller->get_statistics();
                if (post_cleanup_stats.current_memory_usage < stats.current_memory_usage) {
                    size_t freed_memory = stats.current_memory_usage - post_cleanup_stats.current_memory_usage;
                }
            } catch (const std::exception& e) {
                std::cerr << "内存池清理错误: " << e.what() << std::endl;
            }
            last_cleanup_time = now;
        }
        
        // 如果处于紧急状态，尝试清理
        if (stats.current_level >= BackpressureController::BackpressureLevel::CRITICAL) {
            try {
                memory_pool->periodic_cleanup();
                
                // 如果是紧急状态，可能需要更激进的清理
                if (stats.current_level >= BackpressureController::BackpressureLevel::EMERGENCY) {
                    // 触发背压控制器的紧急清理
                    g_backpressure_controller->trigger_emergency_cleanup();
                }
            } catch (const std::exception& e) {
                std::cerr << "紧急清理错误: " << e.what() << std::endl;
            }
        }
        
        // 动态调整监控频率：背压级别越高，监控越频繁
        std::chrono::seconds sleep_duration(5);
        switch (stats.current_level) {
            case BackpressureController::BackpressureLevel::NORMAL:
                sleep_duration = std::chrono::seconds(5);
                break;
            case BackpressureController::BackpressureLevel::WARNING:
                sleep_duration = std::chrono::seconds(3);
                break;
            case BackpressureController::BackpressureLevel::CRITICAL:
                sleep_duration = std::chrono::seconds(2);
                break;
            case BackpressureController::BackpressureLevel::EMERGENCY:
                sleep_duration = std::chrono::seconds(1);
                break;
        }
        
        std::this_thread::sleep_for(sleep_duration);
        
    } catch (const std::exception& e) {
        std::cerr << "内存监控错误: " << e.what() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    } catch (...) {
        std::cerr << "内存监控未知错误" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

std::cout << "内存监控线程正常退出" << std::endl;
}

// 监控线程函数
void monitor_thread() {
uint64_t last_checked = 0;
uint64_t last_matches = 0;
auto start_time = std::chrono::high_resolution_clock::now();

while (global_state.running) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    auto current_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(current_time - start_time).count();
    int elapsed_int = static_cast<int>(elapsed);
    int hours = elapsed_int / 3600;
    int minutes = (elapsed_int % 3600) / 60;
    int seconds = elapsed_int % 60;
    
    double speed = (global_state.addresses_checked.load() - last_checked) / 5.0;
    std::cout << "\rSpeed: " << std::fixed << std::setprecision(2) << speed << " addr/s | "
              << "Total: " << global_state.addresses_checked.load() << " | "
              << "Matches: " << global_state.matches_found.load() << " | "
              << "Uptime: " << hours << "h " << minutes << "m " << seconds << "s"
              << std::flush;
    
    last_checked = global_state.addresses_checked.load();
    last_matches = global_state.matches_found.load();
}
std::cout << std::endl;
}

// 加载目标地址
bool load_target_addresses(const std::string& filename) {
std::ifstream file(filename);
if (!file.is_open()) {
std::cerr << "Failed to open target addresses file: " << filename << std::endl;
return false;
}

std::string line;
while (std::getline(file, line)) {
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
    line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
    line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
    
    if (!line.empty()) {
        std::lock_guard<std::shared_mutex> lock(global_state.target_mutex);
        global_state.target_addresses.insert(line);
    }
}

file.close();
std::cout << "Loaded " << global_state.target_addresses.size() << " target addresses" << std::endl;
return true;
}

bool my_sha256(void* digest, const void* data, size_t len) {
return SHA256(static_cast<const unsigned char>(data), len, static_cast<unsigned char>(digest)) != nullptr;
}

// 清理OpenSSL资源
void cleanup_openssl() {
EVP_cleanup();
ERR_free_strings();
CRYPTO_cleanup_all_ex_data();
CONF_modules_unload(1);
CONF_modules_free();
ENGINE_cleanup();
}

// 主函数
int main(int argc, char* argv[]) {
// 设置 Base58 的 SHA256 实现
b58_sha256_impl = my_sha256;
if (!b58_sha256_impl) {
std::cerr << "Error: SHA256 implementation not set for Base58!" << std::endl;
return 1;
}

// 初始化加密库
OpenSSL_add_all_algorithms();
ERR_load_crypto_strings();

if (sodium_init() < 0) {
    std::cerr << "Failed to initialize libsodium" << std::endl;
    return 1;
}

// 注册信号处理
signal(SIGINT, signal_handler);
signal(SIGTERM, signal_handler);

// 命令行参数解析
std::string target_file = "targets.txt";
global_state.result_file = "matches.txt";

for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-t" && i + 1 < argc) {
        target_file = argv[++i];
    } else if (std::string(argv[i]) == "-o" && i + 1 < argc) {
        global_state.result_file = argv[++i];
    } else if (std::string(argv[i]) == "-h") {
        std::cout << "Usage: " << argv[0] << " [-t targets_file] [-o output_file]\n";
        return 0;
    }
}

// 加载目标地址
if (!load_target_addresses(target_file)) {
    return 1;
}

// 创建背压控制器配置
BackpressureController::Config config;
config.global_memory_ratio = 0.7;
config.memory_pool_ratio = 0.4;
config.other_memory_ratio = 0.3;
config.max_queue_size = 10000;
config.warning_threshold_ratio = 0.7;
config.critical_threshold_ratio = 0.9;
config.use_system_memory = true;

// 初始化背压控制器和内存池
g_backpressure_controller = std::make_unique<BackpressureController>(config);
size_t block_size = calculate_memory_pool_block_size();
ImprovedMemoryPool memory_pool(block_size);

// 打印系统配置信息
std::cout << "\n=== 系统配置信息 ===" << std::endl;
std::cout << "系统总内存: " << SystemMemoryMonitor::get_system_total_memory() / (1024*1024) << "MB" << std::endl;
std::cout << "全局内存限制: " << g_backpressure_controller->get_global_memory_limit() / (1024*1024) << "MB" << std::endl;
std::cout << "内存池块大小: " << block_size / 1024 << "KB" << std::endl;
std::cout << "生产者线程数: " << PRODUCER_THREADS << std::endl;
std::cout << "消费者线程数: " << CONSUMER_THREADS << std::endl;
std::cout << "===================\n" << std::endl;

// 创建监控队列
MonitoredQueue<std::shared_ptr<ImprovedBatchTask>> task_queue(MAX_QUEUE_SIZE);

// 创建线程容器
std::vector<std::thread> producer_threads;
std::vector<std::thread> consumer_threads;
std::vector<std::thread> monitor_threads;

// 启动内存监控线程
monitor_threads.emplace_back(improved_memory_monitor_thread, &memory_pool);

// 启动生产者线程
for (size_t i = 0; i < PRODUCER_THREADS; ++i) {
    producer_threads.emplace_back(Improved_producer_thread, std::ref(task_queue));
}

// 启动消费者线程
for (size_t i = 0; i < CONSUMER_THREADS; ++i) {
    consumer_threads.emplace_back(improved_consumer_thread, std::ref(task_queue));
}

// 启动监控线程
monitor_threads.emplace_back(monitor_thread);

std::cout << "所有线程已启动，开始处理..." << std::endl;

// 等待生产者线程结束
for (auto& t : producer_threads) {
    if (t.joinable()) {
        t.join();
    }
}

std::cout << "所有生产者线程已完成" << std::endl;

// 清空队列
size_t cleared_tasks = 0;
while (!task_queue.empty()) {
    std::shared_ptr<ImprovedBatchTask> batch;
    if (task_queue.dequeue(batch)) {
        // 安全清理私钥
        if (batch) {
            for (auto& key : batch->private_keys) {
                sodium_memzero(key.data(), key.size());
            }
        }
        cleared_tasks++;
    }
}

if (cleared_tasks > 0) {
    std::cout << "清理了 " << cleared_tasks << " 个剩余任务" << std::endl;
}

// 停止消费者和监控线程
global_state.running = false;

// 等待消费者线程结束
for (auto& t : consumer_threads) {
    if (t.joinable()) {
        t.join();
    }
}

std::cout << "所有消费者线程已完成" << std::endl;

// 等待监控线程结束
for (auto& t : monitor_threads) {
    if (t.joinable()) {
        t.join();
    }
}

std::cout << "所有监控线程已完成" << std::endl;

// 清理资源
g_backpressure_controller.reset();
cleanup_openssl();

std::cout << "程序正常终止。总共找到匹配: " << global_state.matches_found.load() << std::endl;

return 0;
}
