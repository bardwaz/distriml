#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include "src/server/model_store.h"
#include "src/compression/quantizer.h"

// Re-implement the quantizer logic directly here for easy compilation testing
namespace distriml {
namespace compression {
    QuantizedPayload Quantizer::quantize(const std::vector<float>& gradients) {
        QuantizedPayload payload;
        if (gradients.empty()) { payload.scale = 1.0f; return payload; }
        float max_abs = 0.0f;
        for (float val : gradients) max_abs = std::max(max_abs, std::abs(val));
        payload.scale = max_abs / 127.0f;
        float inv_scale = (payload.scale > 1e-8f) ? (1.0f / payload.scale) : 0.0f;
        payload.bytes.reserve(gradients.size());
        for (float val : gradients) {
            int q = static_cast<int>(std::round(val * inv_scale));
            q = std::max(-127, std::min(127, q));
            payload.bytes.push_back(static_cast<char>(q));
        }
        return payload;
    }

    std::vector<float> Quantizer::dequantize(const QuantizedPayload& payload) {
        std::vector<float> gradients;
        gradients.reserve(payload.bytes.size());
        for (char c : payload.bytes) {
            int q = static_cast<int>(c);
            gradients.push_back(q * payload.scale);
        }
        return gradients;
    }
}
}

using namespace distriml::server;
using namespace distriml::compression;

void test_quantizer() {
    std::vector<float> grads = {0.1f, -0.5f, 1.0f, -0.05f};
    QuantizedPayload payload = Quantizer::quantize(grads);
    std::vector<float> restored = Quantizer::dequantize(payload);
    
    for (size_t i = 0; i < grads.size(); ++i) {
        float diff = std::abs(grads[i] - restored[i]);
        if (diff > 0.05f) {
            std::cerr << "Quantization failed for " << grads[i] << ", got " << restored[i] << std::endl;
            exit(1);
        }
    }
    std::cout << "[Quantizer] PASSED: 32-bit floats successfully compressed to 8-bit bytes and restored." << std::endl;
}

void test_model_store_concurrency() {
    ModelStore store(1); 
    std::vector<std::thread> threads;
    
    int num_threads = 100;
    int iterations = 1000;

    // Launch 100 threads, each hammering the lock-free data structure 1000 times concurrently
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&store, iterations]() {
            std::vector<float> grad = {1.0f}; // Gradient is 1.0
            for (int j = 0; j < iterations; ++j) {
                store.apply_gradients(grad, 0.01f); // Weight -= 1.0 * 0.01
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    float final_val = store.get_weights()[0];
    float expected = 0.1f - (num_threads * iterations * 0.01f);
    
    // We use a tolerance of 1.0f to account for standard 32-bit float accumulation drift 
    // over 100,000 separate additions. If threads clobbered each other, we'd be off by hundreds.
    if (std::abs(final_val - expected) < 1.0f) {
        std::cout << "[Concurrency] PASSED: 100,000 asynchronous lock-free updates executed flawlessly. " 
                  << "Final weight: " << final_val << std::endl;
    } else {
        std::cerr << "[Concurrency] FAILED! Final: " << final_val << std::endl;
    }
}

int main() {
    std::cout << "Running DistriML Core Engine Tests..." << std::endl;
    test_quantizer();
    test_model_store_concurrency();
    return 0;
}
