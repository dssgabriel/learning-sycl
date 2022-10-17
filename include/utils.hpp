#pragma once

#include <chrono>
#include <random>
#include <vector>

using namespace std::chrono;
using f32 = float;
using f64 = double;

constexpr size_t META_REPS = 31;
constexpr size_t WARMUP_REPS = 1000;
constexpr size_t REPS = 1000;

namespace utils {
    namespace stats {
        auto mean(const std::vector<f64> &samples) -> duration<f64, std::micro>;
        auto stddev(const std::vector<f64> &samples) -> duration<f64, std::micro>;
    } // namespace stats

    template <typename T = f32>
    auto vector_rand_init(size_t len) -> std::vector<T> {
        std::mt19937 rng(0);
        std::uniform_real_distribution<T> distrib(0.0, 1.0);

        std::vector<T> self(len);
        for (size_t i = 0; i < self.capacity(); ++i) {
            self[i] = distrib(rng);
        }

        return self;
    }
} // namespace utils
