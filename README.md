# DistriML: High-Performance Distributed Parameter Server

**DistriML** is a high-performance, distributed machine learning parameter server written in C++ and gRPC. It is designed to tackle two of the most critical bottlenecks in distributed deep learning: **Network Bandwidth** and **Concurrency Contention**.

## Key Features

1. **Lock-Free Concurrency (Hogwild! Algorithm):** 
   Instead of relying on slow `std::mutex` locks that serialize memory access, the `ModelStore` utilizes C++ `std::atomic` and Compare-And-Swap (CAS) loops with `std::memory_order_relaxed`. This allows hundreds of worker threads to apply gradient updates concurrently with zero lock contention.
   
2. **Network Bandwidth Optimization (Gradient Quantization):** 
   Gradients are dynamically quantized from 32-bit floats to 8-bit integers before network transmission. This slashes the gRPC payload size by **75%**, significantly reducing the network uplink bottleneck when thousands of workers are syncing data.

## Architecture

* **RPC Layer:** Protobuf and gRPC are used to define the contract between the Parameter Server and the Worker nodes. The network payload is optimized to send raw `bytes` rather than standard float arrays.
* **Server:** A highly concurrent C++ gRPC server that maintains the global model state in a lock-free memory store.
* **Worker:** C++ clients that pull the latest weights, simulate forward/backward propagation to compute gradients, quantize the payload, and stream updates back to the server.

## Building from Source

### Prerequisites
* C++17 compliant compiler (GCC 7+, Clang 5+, MSVC)
* CMake 3.15+
* gRPC & Protobuf

### Compilation
```bash
mkdir build && cd build
cmake ..
make -j
```

## Running the Cluster

**1. Start the Parameter Server:**
```bash
./parameter_server
```
*(The server will initialize a lock-free model store and listen on port 50051).*

**2. Start a Worker Node:**
```bash
./worker worker-1
```
*(You can spawn multiple workers in different terminals to see the lock-free concurrency in action).*
