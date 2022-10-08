#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace stats {
    auto mean(const std::vector<f64> &samples) -> duration<f64, std::micro>
    {
        f64 mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        return duration<f64, std::micro>(mean);
    }

    auto stddev(const std::vector<f64> &samples) -> duration<f64, std::micro>
    {
        f64 mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        f64 acc = 0.0;
        std::for_each(samples.begin(), samples.end(), [&](const f64 val) {
            acc += (val - mean) * (val - mean);
        });
        f64 stddev = sqrt(acc / samples.size());
        return duration<f64, std::micro>(stddev);
    }
}

auto vector_rand_init(size_t len) -> std::vector<f32>
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<f32> distrib(0.0, 1.0);

    std::vector<f32> self(len);
    for (size_t i = 0; i < self.capacity(); ++i) {
        self[i] = distrib(rng);
    }
    return self;
}