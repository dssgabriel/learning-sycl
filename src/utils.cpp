#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace utils {
    namespace stats {
        auto mean(const std::vector<f64> &samples) -> duration<f64, std::micro> {
            f64 mean = std::reduce(samples.begin(), samples.end(), 0.0) / samples.size();
            return duration<f64, std::micro>(mean);
        }

        auto stddev(const std::vector<f64> &samples) -> duration<f64, std::micro> {
            f64 mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
            f64 acc = 0.0;
            std::for_each(samples.begin(), samples.end(),
                          [&](const f64 val) { acc += (val - mean) * (val - mean); });
            f64 stddev = sqrt(acc / samples.size());
            return duration<f64, std::micro>(stddev);
        }
    } // namespace stats
} // namespace utils