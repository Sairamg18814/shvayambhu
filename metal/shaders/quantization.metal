#include <metal_stdlib>
using namespace metal;

// INT8 Quantization kernel
kernel void quantize_int8(
    device const float* input [[buffer(0)]],
    device char* output [[buffer(1)]],
    device float* scale [[buffer(2)]],
    device float* zero_point [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    
    // Dynamic quantization per block
    const uint block_size = 256;
    uint block_id = tid / block_size;
    uint block_start = block_id * block_size;
    uint block_end = min(block_start + block_size, size);
    
    // Find min/max in block
    float min_val = INFINITY;
    float max_val = -INFINITY;
    
    for (uint i = block_start; i < block_end; i++) {
        min_val = min(min_val, input[i]);
        max_val = max(max_val, input[i]);
    }
    
    // Calculate scale and zero point
    float range = max_val - min_val;
    float s = range / 255.0f;
    float z = -min_val / s;
    
    if (tid == block_start) {
        scale[block_id] = s;
        zero_point[block_id] = z;
    }
    
    // Quantize
    float quantized = round(input[tid] / s + z);
    quantized = clamp(quantized, -128.0f, 127.0f);
    output[tid] = char(quantized);
}

// INT8 Dequantization kernel
kernel void dequantize_int8(
    device const char* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* zero_point [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    
    const uint block_size = 256;
    uint block_id = tid / block_size;
    
    float s = scale[block_id];
    float z = zero_point[block_id];
    
    output[tid] = (float(input[tid]) - z) * s;
}

// INT4 Quantization kernel
kernel void quantize_int4(
    device const float* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],  // Packed INT4
    device float* scale [[buffer(2)]],
    device float* zero_point [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint idx = tid * 2;  // Each thread handles 2 values
    if (idx >= size) return;
    
    // Similar quantization logic but pack 2 INT4 values per byte
    // Implementation details...
}