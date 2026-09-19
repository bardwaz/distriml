#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "parameter_server.grpc.pb.h"
#include "compression/quantizer.h"
#include "server/model_store.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
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
namespace server {

class ParameterServerImpl final : public ParameterServer::Service {
public:
    ParameterServerImpl(size_t model_size) : model_store_(model_size) {}

    Status RegisterWorker(ServerContext* context, const RegisterRequest* request,
                          RegisterResponse* reply) override {
        std::cout << "[Server] Worker connected: " << request->worker_id() << std::endl;
        reply->set_success(true);
        reply->set_model_size(static_cast<int32_t>(model_store_.size()));
        return Status::OK;
    }

    Status PullWeights(ServerContext* context, const PullRequest* request,
                       PullResponse* reply) override {
        // Lock-free read of the global weights
        std::vector<float> current_weights = model_store_.get_weights();
        
        // Copy weights into the gRPC protobuf message
        for (float w : current_weights) {
            reply->add_weights(w);
        }
        
        return Status::OK;
    }

    Status PushGradients(ServerContext* context, const PushRequest* request,
                         PushResponse* reply) override {
        // 1. Reconstruct the quantized payload from the network request
        QuantizedPayload payload;
        payload.bytes = request->quantized_gradients();
        payload.scale = request->scale_factor();

        // 2. Dequantize back to 32-bit floats
        std::vector<float> gradients = Quantizer::dequantize(payload);

        // 3. Apply the gradients to the global model lock-free (Hogwild!)
        model_store_.apply_gradients(gradients, 0.01f);

        reply->set_success(true);
        return Status::OK;
    }

private:
    ModelStore model_store_;
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    
    // Initialize a model with 1,000,000 parameters to simulate a realistic load.
    // 1M floats = ~4MB of data. The 8-bit compression will reduce network push to 1MB.
    ParameterServerImpl service(1000000); 

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "[Server] DistriML Parameter Server listening on " << server_address << std::endl;

    server->Wait();
}

} // namespace server
} // namespace distriml

int main(int argc, char** argv) {
    distriml::server::RunServer();
    return 0;
}
