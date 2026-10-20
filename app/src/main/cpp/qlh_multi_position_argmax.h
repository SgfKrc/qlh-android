#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Keep the caller's output untouched unless every requested logit row is valid.
// Strict > preserves llama's first-index tie breaking, including signed zero.
template <typename RowLookup>
bool qlh_multi_position_argmax(int32_t n_tokens, int32_t n_vocab,
                               RowLookup row_lookup, std::vector<int32_t> & output) {
    if (n_tokens <= 0 || n_vocab <= 0) return false;
    std::vector<int32_t> tokens((std::size_t) n_tokens);
    for (int32_t row = 0; row < n_tokens; ++row) {
        const float * logits = row_lookup(row);
        if (logits == nullptr) return false;
        int32_t best = 0;
        for (int32_t col = 0; col < n_vocab; ++col) {
            if (!std::isfinite(logits[col])) return false;
            if (logits[col] > logits[best]) best = col;
        }
        tokens[(std::size_t) row] = best;
    }
    output.swap(tokens);
    return true;
}

// Explicit positions determine the effective base, independently of root.pos_base.
inline bool qlh_multi_position_positions(const std::vector<int32_t> & positions, uint32_t n_ctx) {
    if (positions.empty() || positions.front() < 0 || (int64_t) positions.back() >= (int64_t) n_ctx) return false;
    for (std::size_t i = 1; i < positions.size(); ++i) {
        if ((int64_t) positions[i] != (int64_t) positions.front() + (int64_t) i) return false;
    }
    return true;
}
