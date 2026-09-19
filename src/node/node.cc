#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <random>

#include <grpcpp/grpcpp.h>
#include "ring_node.grpc.pb.h"
#include "compression/quantizer.h"
#include "server/model_store.h" // We reuse our legendary lock-free store!

using grpc::Channel;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ClientContext;
using grpc::Status;

using distriml::RingNode;
using distriml::ChunkRequest;
using distriml::ChunkResponse;
using distriml::compression::Quantizer;
using distriml::compression::QuantizedPayload;
using distriml::server::ModelStore;

namespace distriml {
namespace ring {

// 1. The Server portion of the Node (Receives chunks from the left neighbor)
class RingNodeServiceImpl final : public RingNode::Service {
public:
    RingNodeServiceImpl(std::shared_ptr<ModelStore> store) : model_store_(store) {}

    Status PushRingChunk(ServerContext* context, const ChunkRequest* request,
                         ChunkResponse* reply) override {
        // 1. Reconstruct the quantized payload from the network request
        QuantizedPayload payload;
        payload.bytes = request->quantized_data();
        payload.scale = request->scale_factor();

        // 2. Dequantize the 8-bit bytes back to 32-bit floats
        std::vector<float> chunk_updates = Quantizer::dequantize(payload);

        // 3. THE MAGIC: The network thread applies these updates LOCK-FREE
        // while the local compute threads are simultaneously updating the store!
        model_store_->apply_gradients(chunk_updates, 1.0f); 

        reply->set_success(true);
        return Status::OK;
    }

private:
    std::shared_ptr<ModelStore> model_store_;
};

// 2. The Node Manager (Orchestrates Server, Client, and Compute Threads)
class PeerNode {
public:
    PeerNode(const std::string& my_address, const std::string& right_neighbor_address) 
        : my_address_(my_address), right_neighbor_address_(right_neighbor_address) {
        
        // Initialize the lock-free model store with 100,000 parameters
        model_store_ = std::make_shared<ModelStore>(100000);
    }

    void Start() {
        // Start the gRPC Server to listen to left neighbor
        std::thread server_thread(&PeerNode::RunServer, this);

        // Wait a second to let all nodes in the cluster boot up
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Start the background network thread to push syncs to the right neighbor
        std::thread sync_thread(&PeerNode::RunRingSync, this);

        // Start multiple local Compute threads (Hogwild! style)
        std::vector<std::thread> compute_threads;
        for (int i = 0; i < 4; ++i) { // Spawning 4 concurrent compute threads per node
            compute_threads.emplace_back(&PeerNode::RunCompute, this, i);
        }

        // Block main thread
        server_thread.join();
    }

private:
    std::string my_address_;
    std::string right_neighbor_address_;
    std::shared_ptr<ModelStore> model_store_;

    void RunServer() {
        RingNodeServiceImpl service(model_store_);
        ServerBuilder builder;
        builder.AddListeningPort(my_address_, grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        std::unique_ptr<Server> server(builder.BuildAndStart());
        std::cout << "[Node] Listening for left neighbor on " << my_address_ << std::endl;
        server->Wait();
    }

    void RunRingSync() {
        auto channel = grpc::CreateChannel(right_neighbor_address_, grpc::InsecureChannelCredentials());
        auto stub = RingNode::NewStub(channel);

        std::cout << "[Node] Starting async ring sync with right neighbor: " << right_neighbor_address_ << std::endl;

        while (true) {
            // Take a lock-free snapshot of the local weights
            std::vector<float> snapshot = model_store_->get_weights();

            // Quantize to 8-bit to save 75% bandwidth
            QuantizedPayload payload = Quantizer::quantize(snapshot);

            ChunkRequest request;
            request.set_chunk_index(0);
            request.set_quantized_data(payload.bytes);
            request.set_scale_factor(payload.scale);

            ChunkResponse reply;
            ClientContext context;

            Status status = stub->PushRingChunk(&context, request, &reply);
            if (status.ok()) {
                std::cout << "[Network Thread] Pushed " << snapshot.size() * sizeof(float) 
                          << "B chunk compressed to " << payload.bytes.size() << "B to neighbor." << std::endl;
            } else {
                // If neighbor is down, wait and retry
                std::cerr << "[Network Thread] Waiting for neighbor..." << std::endl;
            }

            // Sync every 500ms
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void RunCompute(int thread_id) {
        std::mt19937 gen(thread_id);
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

        while (true) {
            // Simulate ML forward/backward propagation computing gradients
            std::vector<float> local_gradients(model_store_->size());
            for (float& g : local_gradients) {
                g = dist(gen);
            }

            // Apply gradients to the lock-free model store using IEEE-754 CAS loops.
            // This executes completely concurrently with the network thread 
            // AND the other compute threads!
            model_store_->apply_gradients(local_gradients, 0.01f);

            // Simulate GPU compute time
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

} // namespace ring
} // namespace distriml

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./ring_node <my_ip:port> <right_neighbor_ip:port>" << std::endl;
        std::cerr << "Example: ./ring_node 0.0.0.0:50051 localhost:50052" << std::endl;
        return 1;
    }

    std::string my_address = argv[1];
    std::string neighbor_address = argv[2];

    distriml::ring::PeerNode node(my_address, neighbor_address);
    node.Start();

    return 0;
}
