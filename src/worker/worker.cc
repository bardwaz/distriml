#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <chrono>

#include <grpcpp/grpcpp.h>
#include "parameter_server.grpc.pb.h"
#include "compression/quantizer.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using distriml::ParameterServer;
using distriml::RegisterRequest;
using distriml::RegisterResponse;
using distriml::PullRequest;
using distriml::PullResponse;
using distriml::PushRequest;
using distriml::PushResponse;
using distriml::compression::Quantizer;
using distriml::compression::QuantizedPayload;

namespace distriml {
namespace worker {

class DistriMLClient {
public:
    DistriMLClient(std::shared_ptr<Channel> channel, const std::string& worker_id)
        : stub_(ParameterServer::NewStub(channel)), worker_id_(worker_id), model_size_(0) {}

    bool Register() {
        RegisterRequest request;
        request.set_worker_id(worker_id_);
        
        RegisterResponse reply;
        ClientContext context;

        Status status = stub_->RegisterWorker(&context, request, &reply);
        if (status.ok() && reply.success()) {
            model_size_ = reply.model_size();
            std::cout << "[" << worker_id_ << "] Registered. Model size: " << model_size_ << std::endl;
            return true;
        } else {
            std::cerr << "[" << worker_id_ << "] Registration failed: " << status.error_message() << std::endl;
            return false;
        }
    }

    std::vector<float> PullWeights() {
        PullRequest request;
        request.set_worker_id(worker_id_);
        
        PullResponse reply;
        ClientContext context;

        Status status = stub_->PullWeights(&context, request, &reply);
        if (status.ok()) {
            std::vector<float> weights(reply.weights().begin(), reply.weights().end());
            return weights;
        } else {
            std::cerr << "[" << worker_id_ << "] Pull failed: " << status.error_message() << std::endl;
            return {};
        }
    }

    bool PushGradients(const std::vector<float>& gradients) {
        // 1. Compress the gradients using our 8-bit quantizer
        QuantizedPayload payload = Quantizer::quantize(gradients);

        // 2. Build the gRPC request
        PushRequest request;
        request.set_worker_id(worker_id_);
        request.set_quantized_gradients(payload.bytes);
        request.set_scale_factor(payload.scale);

        PushResponse reply;
        ClientContext context;

        // 3. Send over the network
        Status status = stub_->PushGradients(&context, request, &reply);
        
        if (status.ok() && reply.success()) {
            // Log the bandwidth saving
            size_t original_size = gradients.size() * sizeof(float);
            size_t compressed_size = payload.bytes.size();
            std::cout << "[" << worker_id_ << "] Pushed gradients. Bandwidth: " 
                      << original_size << "B -> " << compressed_size << "B (75% savings)" << std::endl;
            return true;
        } else {
            std::cerr << "[" << worker_id_ << "] Push failed: " << status.error_message() << std::endl;
            return false;
        }
    }

    int GetModelSize() const { return model_size_; }

private:
    std::unique_ptr<ParameterServer::Stub> stub_;
    std::string worker_id_;
    int model_size_;
};

void RunWorker(const std::string& worker_id) {
    std::string server_address("localhost:50051");
    DistriMLClient client(
        grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()), 
        worker_id
    );

    if (!client.Register()) {
        return;
    }

    // Training Loop Simulation
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    int epochs = 5;
    for (int e = 0; e < epochs; ++e) {
        // 1. Pull latest weights
        std::vector<float> weights = client.PullWeights();
        if (weights.empty()) break;

        // 2. Simulate Forward / Backward propagation to compute gradients
        // (In the real world, we would use the matrix math we built in Python, ported to C++).
        // Here, we simulate generating 1 million gradients (representing a deep neural network).
        std::vector<float> gradients(client.GetModelSize());
        for (float& g : gradients) {
            g = dist(gen);
        }

        // Simulate GPU compute time
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // 3. Push quantized gradients to server
        client.PushGradients(gradients);
    }
}

} // namespace worker
} // namespace distriml

int main(int argc, char** argv) {
    std::string worker_id = "worker-1";
    if (argc > 1) {
        worker_id = argv[1];
    }
    distriml::worker::RunWorker(worker_id);
    return 0;
}
