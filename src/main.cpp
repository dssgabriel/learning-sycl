#include "drivers.hpp"
#include "utils.hpp"

#include <iostream>
#include <vector>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

int main(int argc, char *argv[argc + 1])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <LEN>" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const size_t len = std::atoi(argv[1]);
    std::vector<f32> x = vector_rand_init(len);
    std::vector<f32> y = vector_rand_init(len);
    const f32 alpha = 1.0;

    driver_gaxpy(len, alpha, x, y);
    driver_sycl_gaxpy(len, alpha, x, y);

    return 0;
}
