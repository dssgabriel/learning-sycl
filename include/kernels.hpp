#pragma once

#include <cstddef>
#include <vector>

template <typename T>
void gaxpy(size_t n, T alpha, std::vector<T> const &x, std::vector<T> &y)
{
    for (size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}