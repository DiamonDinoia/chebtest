#include <cmath>       // for std::abs
#include <immintrin.h> // for AVX
#include <iostream>
#include <limits>
#include <nanobench.h>
#include <random>
#include <stdexcept>

double cheb_eval_generic(int order, double x, const double *c);
double cheb_eval_if(int order, double x, const double *coeffs);
double cheb_eval_switch(int order, double x, const double *c);

template <int ORDER, typename T>
inline T cheb_eval(T x, const T *c) {
    //        __asm(";cheb_eval start");
    const T x2 = 2 * x;

    T c0 = c[0];
    T c1 = c[1];
    for (int i = 2; i < ORDER; ++i) {
        T tmp = c1;
        c1 = c[i] - c0;
        c0 = tmp + c0 * x2;
    }
    //    __asm(";cheb_eval end");
    return c1 + c0 * x;
}

template <int ORDER, typename T>
T cheb_eval_fast(T x, const T *c) {
    return cheb_eval<ORDER>(x, c);
}


template <typename T>
__attribute__((always_inline)) inline T cheb_eval_vector_order_12(T x, const T *c) {
    const __m256d x2 = _mm256_set1_pd(2 * x);

    __m256d c0 = _mm256_loadu_pd(c);
    __m256d c1 = _mm256_loadu_pd(c + 4);

    __m256d tmp = c1;
    c1 = _mm256_sub_pd(_mm256_loadu_pd(c), c0);
    c0 = _mm256_fmadd_pd(c0, x2, tmp);

    __m128d c2 = _mm_loadu_pd(c);
    __m128d c3 = _mm_loadu_pd(c + 2);

    __m128d tmp2 = c3;
    c3 = _mm_sub_pd(_mm_loadu_pd(c + 4), c2);
    c2 =  _mm_fmadd_pd(c2, _mm_set1_pd(x), tmp2);

    c0 = _mm256_add_pd(c0, _mm256_castpd128_pd256(c2));
    c1 = _mm256_add_pd(c1, _mm256_castpd128_pd256(c3));
    // Combine the results
    __m256d result =  _mm256_fmadd_pd(c0, x2, c1);
    return result[0] + result[1] + result[2] + result[3];
}

template <typename T>
__attribute__((always_inline)) inline T cheb_eval_vector_order_8(T x, const T *c) {
    const __m256d x2 = _mm256_set1_pd(2 * x);

    __m256d c0 = _mm256_loadu_pd(c);
    __m256d c1 = _mm256_set1_pd(0);

    __m256d tmp = _mm256_loadu_pd(c + 4);
    c0 = _mm256_fmadd_pd(c0, x2, tmp);

    tmp = c1;
    c1 = _mm256_sub_pd(_mm256_loadu_pd(c + 4), c0);
    c0 = _mm256_fmadd_pd(c0, x2, tmp);
    // Combine the results
    __m256d result = _mm256_fmadd_pd(c0, x2, c1);
    return result[0] + result[1] + result[2] + result[3];
}

template <typename T>
__attribute__((always_inline)) inline T cheb_eval_vector_order_4(T x, const T *c) {
    const T x2 = 2 * x;
    // Load the first two coefficients
    __m256d c0_c1 = _mm256_set_pd(0, 0, c[1], c[0]);
    // Load the next pair of coefficients
    __m256d c_i_c_i1 = _mm256_set_pd(0, 0, c[3], c[2]);
    // Compute c1
    __m256d c1 = _mm256_sub_pd(c_i_c_i1, c0_c1);
    // Compute c0
    c0_c1 = _mm256_fmadd_pd(c0_c1, _mm256_set1_pd(x2), _mm256_permute_pd(c0_c1, 0x05));
    // Update c0_c1 for the next iteration
    c0_c1 = _mm256_blend_pd(c1, c0_c1, 0b1100);
    // Extract the final result from c0_c1
    return c0_c1[0] + c0_c1[1];
}

template <int ORDER, typename T>
inline T cheb_eval_vector(T x, const T *c) {
    if constexpr (ORDER == 12) {
        return cheb_eval_vector_order_12(x, c);
    } else if constexpr (ORDER == 8) {
        return cheb_eval_vector_order_8(x, c);
    } else if constexpr (ORDER == 4) {
        return cheb_eval_vector_order_4(x, c);
    } else {
        return cheb_eval<ORDER>(x, c);
    }
}
template <int ORDER, typename T>
bool test_correctness(T x, const T *c) {
    T result_cheb_eval = cheb_eval<ORDER>(x, c);
    T result_cheb_eval_generic = cheb_eval_fast<ORDER>(x, c);
    auto valid = (std::abs(result_cheb_eval - result_cheb_eval_generic) /
                  std::max(result_cheb_eval, result_cheb_eval_generic)) < 1e-14;
    if (!valid) {
        std::cout << "order " << ORDER << " " << result_cheb_eval << " " << result_cheb_eval_generic << std::endl;
    }
    // Consider two results as equal if the absolute difference is less than a small threshold
    return valid;
}

template <int ORDER, typename T>
bool test_correctness_vector(T x, const T *c) {
    T result_cheb_eval_vector = cheb_eval_vector<ORDER>(x, c);
    T result_cheb_eval_generic = cheb_eval_generic(ORDER, x, c);

    // Consider two results as equal if the absolute difference is less than a small threshold
    return (std::abs(result_cheb_eval_vector - result_cheb_eval_generic) /
            std::max(result_cheb_eval_vector, result_cheb_eval_generic)) < 1e-14;
}

template <typename T, int... Orders>
void run_tests(T x, const T *c) {
    std::string failed_orders; // Store orders for which tests failed
    // Iterate over each order in the variadic list
    (((test_correctness<Orders>(x, c) && test_correctness_vector<Orders>(x, c)) ||
      (failed_orders += std::to_string(Orders) + ", ", false)),
     ...);

    if (!failed_orders.empty()) {
        failed_orders.pop_back(); // Remove the last comma
        throw std::runtime_error("Tests failed for order(s): " + failed_orders);
    }
}

template <typename T, int... Orders>
void run_benchmarks(T x, const T *c) {
    using ankerl::nanobench::doNotOptimizeAway;
    (ankerl::nanobench::Bench()
         .unit("eval")
         .title("order " + std::to_string(Orders))
         .minEpochIterations(10000000)
         .run("generic", [&] { doNotOptimizeAway(cheb_eval_generic(Orders, x, c)); })
         .run("if", [&] { doNotOptimizeAway(cheb_eval_if(Orders, x, c)); })
         .run("switch", [&] { doNotOptimizeAway(cheb_eval_switch(Orders, x, c)); })
         .run("template", [&] { doNotOptimizeAway(cheb_eval<Orders>(x, c)); })
         .run("fast", [&] { doNotOptimizeAway(cheb_eval_fast<Orders>(x, c)); })
         .run("vector", [&] { doNotOptimizeAway(cheb_eval_vector<Orders>(x, c)); }),
     ...);
}

int main() {
    std::mt19937 gen(1);
    std::uniform_real_distribution<> dis(-1, 1);
    alignas(sizeof(double) * 4) double c[128];
    for (double &i : c)
        i = dis(gen);

    double x{0.1};

    run_tests<double, 4, 6, 8, 10, 12>(x, c);
    run_benchmarks<double, 4, 6, 8, 10, 12>(x, c);

    return 0;
}
