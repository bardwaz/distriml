cmake_minimum_required(VERSION 3.15)
project(DistriML CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Protobuf and gRPC
find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# Set up generated files directory
set(GENERATED_PROTO_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${GENERATED_PROTO_DIR})

# Retrieve the gRPC C++ plugin path
get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)

# Custom command to invoke protoc and generate C++ stubs for the Ring protocol
add_custom_command(
    OUTPUT "${GENERATED_PROTO_DIR}/ring_node.pb.cc"
           "${GENERATED_PROTO_DIR}/ring_node.pb.h"
           "${GENERATED_PROTO_DIR}/ring_node.grpc.pb.cc"
           "${GENERATED_PROTO_DIR}/ring_node.grpc.pb.h"
    COMMAND protobuf::protoc
    ARGS --grpc_out="${GENERATED_PROTO_DIR}"
         --cpp_out="${GENERATED_PROTO_DIR}"
         -I "${CMAKE_CURRENT_SOURCE_DIR}/proto"
         --plugin=protoc-gen-grpc="${grpc_cpp_plugin_location}"
         "${CMAKE_CURRENT_SOURCE_DIR}/proto/ring_node.proto"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/proto/ring_node.proto"
)

# Common library for generated proto stubs
add_library(distriml_proto
    "${GENERATED_PROTO_DIR}/ring_node.pb.cc"
    "${GENERATED_PROTO_DIR}/ring_node.grpc.pb.cc"
)
target_include_directories(distriml_proto PUBLIC "${GENERATED_PROTO_DIR}")
target_link_libraries(distriml_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)

# Compression Module (for 8-bit Quantization)
add_library(distriml_compression src/compression/quantizer.cc)
target_include_directories(distriml_compression PUBLIC src)

# Decentralized Peer-to-Peer Node Executable
add_executable(ring_node src/node/node.cc)
target_include_directories(ring_node PRIVATE src)
target_link_libraries(ring_node distriml_proto distriml_compression)
