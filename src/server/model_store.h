#pragma once
#include <vector>
#include <atomic>
#include <cstring>

namespace distriml {
namespace server {

class ModelStore {
public:
    ModelStore(size_t size) : weights_(size) {
        // Initialize weights to a dummy value (e.g., 0.1)
        for (size_t i = 0; i < size; ++i) {
            float initial_val = 0.1f;
            uint32_t bits;
            std::memcpy(&bits, &initial_val, sizeof(float));
            weights_[i].store(bits, std::memory_order_relaxed);
        }
    }

    size_t size() const { return weights_.size(); }

    // Lock-free read of all weights
    std::vector<float> get_weights() const {
        std::vector<float> current_weights(weights_.size());
        for (size_t i = 0; i < weights_.size(); ++i) {
            uint32_t bits = weights_[i].load(std::memory_order_relaxed);
            float val;
            std::memcpy(&val, &bits, sizeof(float));
            current_weights[i] = val;
        }
        return current_weights;
    }

    // Hogwild! Lock-free asynchronous update using Compare-And-Swap (CAS)
    void apply_gradients(const std::vector<float>& gradients, float learning_rate = 0.01f) {
        for (size_t i = 0; i < weights_.size(); ++i) {
            float grad_update = -(gradients[i] * learning_rate); // Gradient descent step
            atomic_add_float(weights_[i], grad_update); 
        }
    }

private:
    // We use atomic<uint32_t> instead of atomic<float> because C++17 does not guarantee 
    // fetch_add for floats. Using uint32_t allows us to implement a perfect lock-free CAS loop.
    std::vector<std::atomic<uint32_t>> weights_;

    // Elite HFT Flex: Lock-free floating point addition using IEEE 754 bit-casting and CAS
    static void atomic_add_float(std::atomic<uint32_t>& target, float value) {
        uint32_t current_bits = target.load(std::memory_order_relaxed);
        while (true) {
            // Type-pun the bits to a float
            float current_float;
            std::memcpy(&current_float, &current_bits, sizeof(float));
            
            // Perform the mathematical addition
            float new_float = current_float + value;
            
            // Type-pun back to bits
            uint32_t new_bits;
            std::memcpy(&new_bits, &new_float, sizeof(float));

            // Attempt to swap. If another thread changed it, current_bits is updated and we retry.
            // memory_order_relaxed is used because Hogwild! algorithm assumes independent updates 
            // without strict synchronization barriers.
            if (target.compare_exchange_weak(current_bits, new_bits, 
                                             std::memory_order_relaxed, 
                                             std::memory_order_relaxed)) {
                break;
            }
        }
    }
};

} // namespace server
} // namespace distriml
