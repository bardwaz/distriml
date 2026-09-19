#pragma once
#include <vector>
#include <string>

namespace distriml {
namespace compression {

// Represents the compressed payload sent over the network
struct QuantizedPayload {
    std::string bytes; // 8-bit quantized data 
    float scale;       // Scale factor required to reconstruct the floats
};

class Quantizer {
public:
    // Compress a vector of 32-bit floats into 8-bit integers (75% bandwidth reduction)
    static QuantizedPayload quantize(const std::vector<float>& gradients);

    // Decompress the 8-bit integers back into 32-bit floats on the server side
    static std::vector<float> dequantize(const QuantizedPayload& payload);
};

} // namespace compression
} // namespace distriml
