#include <metal_stdlib>
using namespace metal;

// Flash Attention kernel for Metal
kernel void flash_attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_length [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    // Thread indices
    uint batch_idx = tid.z;
    uint head_idx = tid.y;
    uint seq_idx = tid.x;
    
    if (seq_idx >= seq_length || head_idx >= num_heads) {
        return;
    }
    
    uint head_dim = hidden_dim / num_heads;
    
    // Compute attention scores
    float sum = 0.0;
    float max_score = -INFINITY;
    
    // First pass: compute max for numerical stability
    for (uint i = 0; i < seq_length; i++) {
        float score = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            uint q_idx = batch_idx * seq_length * hidden_dim + seq_idx * hidden_dim + head_idx * head_dim + d;
            uint k_idx = batch_idx * seq_length * hidden_dim + i * hidden_dim + head_idx * head_dim + d;
            score += query[q_idx] * key[k_idx];
        }
        score /= sqrt(float(head_dim));
        max_score = max(max_score, score);
    }
    
    // Second pass: compute softmax
    for (uint i = 0; i < seq_length; i++) {
        float score = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            uint q_idx = batch_idx * seq_length * hidden_dim + seq_idx * hidden_dim + head_idx * head_dim + d;
            uint k_idx = batch_idx * seq_length * hidden_dim + i * hidden_dim + head_idx * head_dim + d;
            score += query[q_idx] * key[k_idx];
        }
        score /= sqrt(float(head_dim));
        sum += exp(score - max_score);
    }
    
    // Third pass: compute weighted values
    for (uint d = 0; d < head_dim; d++) {
        float result = 0.0;
        for (uint i = 0; i < seq_length; i++) {
            float score = 0.0;
            for (uint d2 = 0; d2 < head_dim; d2++) {
                uint q_idx = batch_idx * seq_length * hidden_dim + seq_idx * hidden_dim + head_idx * head_dim + d2;
                uint k_idx = batch_idx * seq_length * hidden_dim + i * hidden_dim + head_idx * head_dim + d2;
                score += query[q_idx] * key[k_idx];
            }
            score /= sqrt(float(head_dim));
            float weight = exp(score - max_score) / sum;
            
            uint v_idx = batch_idx * seq_length * hidden_dim + i * hidden_dim + head_idx * head_dim + d;
            result += weight * value[v_idx];
        }
        
        uint out_idx = batch_idx * seq_length * hidden_dim + seq_idx * hidden_dim + head_idx * head_dim + d;
        output[out_idx] = result;
    }
}