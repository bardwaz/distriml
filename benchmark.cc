#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <mutex>
#include <atomic>
#include <random>
#include <iomanip>
#include "src/server/model_store.h"
#include "src/compression/quantizer.h"

using namespace distriml::server;
using namespace distriml::compression;
using namespace std::chrono;

// 1. Mutex Baseline for Comparison
class MutexModelStore {
public:
    MutexModelStore(size_t size) : weights_(size, 0.1f), mtx_(size) {}
    
    void apply_gradients(const std::vector<float>& gradients, float lr) {
        for (size_t i = 0; i < weights_.size(); ++i) {
            std::lock_guard<std::mutex> lock(mtx_[i]);
            weights_[i] -= gradients[i] * lr;
        }
    }
private:
    std::vector<float> weights_;
    std::vector<std::mutex> mtx_;
};

void run_mutex_benchmark(int num_threads, int updates_per_thread, double& time_taken) {
    int num_params = 10000;
    MutexModelStore store(num_params);
    std::vector<std::thread> threads;
    std::vector<float> grad(num_params, 1.0f);

    auto start = high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < updates_per_thread; ++j) {
                store.apply_gradients(grad, 0.01f);
            }
        });
    }
    for (auto& t : threads) t.join();
    auto end = high_resolution_clock::now();
    
    time_taken = duration_cast<duration<double>>(end - start).count();
}

void run_cas_benchmark(int num_threads, int updates_per_thread, double& time_taken) {
    int num_params = 10000;
    ModelStore store(num_params);
    std::vector<std::thread> threads;
    std::vector<float> grad(num_params, 1.0f);

    auto start = high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < updates_per_thread; ++j) {
                store.apply_gradients(grad, 0.01f);
            }
        });
    }
    for (auto& t : threads) t.join();
    auto end = high_resolution_clock::now();

    time_taken = duration_cast<duration<double>>(end - start).count();
}

void test_concurrency() {
    std::cout << "--- Concurrency Benchmark (Mutex vs Lock-Free CAS on 10,000 Params) ---" << std::endl;
    int updates_per_thread = 100; // 100 updates of all 10,000 params
    int total_operations_per_thread = updates_per_thread * 10000;
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    double cas_1_thread_time = 0;

    for (int t : thread_counts) {
        double mutex_time = 0, cas_time = 0;
        
        run_mutex_benchmark(t, updates_per_thread, mutex_time);
        run_cas_benchmark(t, updates_per_thread, cas_time);

        double mutex_ups = (t * total_operations_per_thread) / mutex_time;
        double cas_ups = (t * total_operations_per_thread) / cas_time;
        double speedup = cas_ups / mutex_ups;
        
        if (t == 1) cas_1_thread_time = cas_time;
        
        std::cout << "Threads: " << std::setw(2) << t 
                  << " | Mutex: " << std::setw(10) << (int)mutex_ups << " up/s"
                  << " | CAS: " << std::setw(10) << (int)cas_ups << " up/s"
                  << " | Speedup: " << std::setw(5) << std::fixed << std::setprecision(1) << speedup << "x"
                  << " | Hogwild Scaling vs 1 thread: " << std::fixed << std::setprecision(1) << (cas_1_thread_time / cas_time) * t << "x" << std::endl;
    }
    std::cout << std::endl;
}

void test_quantization() {
    std::cout << "--- Quantization Accuracy Benchmark ---" << std::endl;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f); // Standard normal distribution typical of gradients

    int num_params = 100000;
    std::vector<float> original(num_params);
    for (int i = 0; i < num_params; ++i) original[i] = dist(gen);

    QuantizedPayload payload = Quantizer::quantize(original);
    std::vector<float> restored = Quantizer::dequantize(payload);

    double total_error = 0;
    double max_val = 0;
    for (int i = 0; i < num_params; ++i) {
        total_error += std::abs(original[i] - restored[i]);
        max_val = std::max(max_val, (double)std::abs(original[i]));
    }
    
    double mae = total_error / num_params;
    double accuracy_loss_percent = (mae / max_val) * 100.0;

    std::cout << "Data Size: " << num_params << " floats" << std::endl;
    std::cout << "Payload reduced by 75% (" << num_params * 4 << "B -> " << payload.bytes.size() << "B)" << std::endl;
    std::cout << "Mean Absolute Error (MAE): " << mae << std::endl;
    std::cout << "Max Absolute Value: " << max_val << std::endl;
    std::cout << "Relative Accuracy Loss: " << std::fixed << std::setprecision(2) << accuracy_loss_percent << "% vs fp32" << std::endl;
}

int main() {
    test_concurrency();
    test_quantization();
    return 0;
}
