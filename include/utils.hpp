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

namespace stats {
    auto mean(const std::vector<f64> &samples) -> duration<f64, std::micro>;
    auto stddev(const std::vector<f64> &samples) -> duration<f64, std::micro>;
}
auto vector_rand_init(size_t len) -> std::vector<f32>;
