#include "compression/quantizer.h"
#include <cmath>
#include <algorithm>

namespace distriml {
namespace compression {

QuantizedPayload Quantizer::quantize(const std::vector<float>& gradients) {
    QuantizedPayload payload;
    if (gradients.empty()) {
        payload.scale = 1.0f;
        return payload;
    }

    // 1. Find the maximum absolute value to determine the scale
    float max_abs = 0.0f;
    for (float val : gradients) {
        max_abs = std::max(max_abs, std::abs(val));
    }

    // 2. Map [-max_abs, max_abs] to [-127, 127] (signed 8-bit integer range)
    payload.scale = max_abs / 127.0f;
    
    // Prevent division by zero if all gradients are perfectly 0
    float inv_scale = (payload.scale > 1e-8f) ? (1.0f / payload.scale) : 0.0f;

    // 3. Quantize the floats into bytes
    payload.bytes.reserve(gradients.size());
    for (float val : gradients) {
        // Round to nearest integer and cast to char (int8)
        int q = static_cast<int>(std::round(val * inv_scale));
        // Clamp to [-127, 127] to prevent any overflow edge cases
        q = std::max(-127, std::min(127, q));
        payload.bytes.push_back(static_cast<char>(q));
    }

    return payload;
}

std::vector<float> Quantizer::dequantize(const QuantizedPayload& payload) {
    std::vector<float> gradients;
    gradients.reserve(payload.bytes.size());

    // Restore the floats using the scale factor
    for (char c : payload.bytes) {
        int q = static_cast<int>(c); // Reinterpret char as signed int
        gradients.push_back(q * payload.scale);
    }

    return gradients;
}

} // namespace compression
} // namespace distriml
