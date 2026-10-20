#include "qlh_multi_position_argmax.h"

#include <cassert>
#include <iostream>
#include <limits>

int main() {
    int cases = 0;
    std::vector<std::vector<float>> rows = {{1, 4, 2}, {-5, -3, -4}, {9, 2, 9}};
    std::vector<int32_t> output = {77};
    auto lookup = [&](int32_t row) { return rows[(size_t) row].data(); };
    assert(qlh_multi_position_argmax(3, 3, lookup, output));
    assert((output == std::vector<int32_t>{1, 1, 0}));
    ++cases;
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
                float saved = rows[(size_t) row][(size_t) col];
                rows[(size_t) row][(size_t) col] = bad;
                output = {77, 88};
                assert(!qlh_multi_position_argmax(3, 3, lookup, output));
                assert((output == std::vector<int32_t>{77, 88}));
                rows[(size_t) row][(size_t) col] = saved;
                ++cases;
            }
        }
    }
    for (int32_t missing = 0; missing < 3; ++missing) {
        output = {77, 88};
        assert(!qlh_multi_position_argmax(3, 3, [&](int32_t row) -> const float * {
            return row == missing ? nullptr : lookup(row);
        }, output));
        assert((output == std::vector<int32_t>{77, 88}));
        ++cases;
    }
    for (auto shape : std::vector<std::vector<int32_t>>{{0, 3}, {-1, 3}, {3, 0}, {3, -1}}) {
        output = {77, 88};
        assert(!qlh_multi_position_argmax(shape[0], shape[1], lookup, output));
        assert((output == std::vector<int32_t>{77, 88}));
        ++cases;
    }
    rows = {{-0.0f, 0.0f, -0.0f}};
    assert(qlh_multi_position_argmax(1, 3, lookup, output));
    assert((output == std::vector<int32_t>{0}));
    ++cases;
    for (auto positions : std::vector<std::vector<int32_t>>{{41, 42, 43}, {0}, {63}, {61, 62, 63}}) {
        assert(qlh_multi_position_positions(positions, 64));
        ++cases;
    }
    for (auto positions : std::vector<std::vector<int32_t>>{{}, {-1, 0}, {41, 43}, {42, 41},
                                                          {63, 64}, {2147483647}}) {
        assert(!qlh_multi_position_positions(positions, 64));
        ++cases;
    }
    assert(!qlh_multi_position_positions({0}, 0));
    ++cases;
    std::cout << cases << " native argmax cases passed\n";
}
