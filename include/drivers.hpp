#pragma once

#include "kernels.hpp"
#include "utils.hpp"

#include <chrono>
#include <iostream>
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

constexpr size_t MAX_THREADS = 64;

template <typename T = f32>
void driver_gaxpy(size_t n, T alpha, std::vector<T> const &x, std::vector<T> &y)
{
    std::cout << "SAXPY with " << n << " elements on CPU" << std::endl;
    std::vector<f64> samples(META_REPS);
    high_resolution_clock::time_point start, end;

    // Warmup loop
    for (size_t _ = 0; _ < WARMUP_REPS; ++_) {
        gaxpy(n, alpha, x, y);
    }

    // Main benchmarking loop
    for (size_t i = 0; i < META_REPS; ++i) {
        start = high_resolution_clock::now();
        for (size_t _ = 0; _ < REPS; ++_) {
            gaxpy(n, alpha, x, y);
        }
        end = high_resolution_clock::now();
        samples[i] = ((end - start).count()) / (double)(REPS);
    }

    auto mean = stats::mean(samples);
    auto stddev = stats::stddev(samples);
    std::cout << "Average: " << mean.count() << " +/- " << stddev.count() << " µs" << std::endl;
}

template <typename T = f32>
void driver_sycl_gaxpy(size_t n, T alpha, std::vector<T> const &x, std::vector<T> &y)
{
    sycl::queue q(sycl::gpu_selector{});
    auto sx = sycl::malloc_shared<T>(n, q);
    auto sy = sycl::malloc_shared<T>(n, q);
    for (size_t i = 0; i < n; ++i) {
        sx[i] = x[i];
        sy[i] = y[i];
    }

    std::cout << "SAXPY with " << n << " elements on "
              << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    std::vector<double> samples(META_REPS);
    high_resolution_clock::time_point start, end;
    try {
        for (size_t w = 0; w < WARMUP_REPS; ++w) {
            q.parallel_for(sycl::range<1>(MAX_THREADS), [=](sycl::id<1> i) {
                sy[i] += alpha * sx[i];
            });
        }
        for (size_t m = 0; m < META_REPS; ++m) {
            start = high_resolution_clock::now();
            for (size_t j = 0; j < REPS; ++j) {
                q.parallel_for(sycl::nd_range<1>(MAX_THREADS, 64), [=](sycl::id<1> i) {
                    sy[i] += alpha * sx[i];
                });
            }
            end = high_resolution_clock::now();
            samples[m] = ((end - start).count()) / (double)(REPS);
        }
        q.wait();
    } catch (sycl::exception &e) {
        std::cerr << e.what() << std::endl;
        std::terminate();
    }

    auto mean = stats::mean(samples);
    auto stddev = stats::stddev(samples);
    std::cout << "Average: " << mean.count() << " +/- " << stddev.count() << " µs" << std::endl;

    sycl::free(sx, q);
    sycl::free(sy, q);
}