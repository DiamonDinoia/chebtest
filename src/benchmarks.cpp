#include <iostream>
#include <nanobench.h>
#include <random>
#include <stdexcept>

#include <vectorclass.h>
#include <xsimd/xsimd.hpp>

double cheb_eval_generic(int order, double x, const double *c);
double cheb_eval_if(int order, double x, const double *coeffs);
double cheb_eval_switch(int order, double x, const double *c);

double cheb_eval_4(double x, const double *__restrict__ c) {
    // note (RB): uses clenshaw's method to avoid direct calculation of recurrence relation of
    // T_i, where res = \Sum_i T_i c_i
    // reference code:
    /*
    const auto x20 = 2 * x * c[0];
    const auto c0 = c[2] - c[0] + c[1] + x20;
    const auto c1 = c[3] - c[1] + x20;
    return std::fma(x, c0, c1);
     */
    /*
    const double x20 = 2 * x * c[0];
    using batch_t = xsimd::make_sized_batch_t<double,4>;
    const batch_t c0_v{c[2], - c[0] , c[1] , x20};
    const batch_t c1_v{c[3], - c[1] , x20 , 0.0};
    return xsimd::reduce_add(xsimd::fma(batch_t{x}, c0_v, c1_v));
    */
    /*
    const double x20 = 2 * x * c[0];
    const Vec4d c0_v(c[2], -c[0], c[1], x20);
    const Vec4d c1_v(c[3], -c[1], x20, 0.0);
    const Vec4d result = mul_add(Vec4d(x), c0_v, c1_v);
    const Vec4d sum = result+result;
    return horizontal_add(sum.get_low());
    */

    using batch_t = xsimd::make_sized_batch_t<double, 2>;
    const auto c23 = batch_t::load_aligned(c + 2) - batch_t::load_aligned(c);
    const auto x2 = batch_t{2 * x};
    const auto c0 = batch_t{c[0]};
    const auto res = xsimd::fma(c0, x2, c23);
    return xsimd::fma(res.get(0) + c[1], x, res.get(1));

    /*
    const auto x20 = Vec2d{2 * x * c[0]};
    const auto c1 = Vec2d{c[1], 0};
    const auto c01 = Vec2d{}.load(c);
    const auto c23 = Vec2d{}.load(c+2);
    const auto res = c23 - c01 + c1 + x20;
    return xsimd::fma(res[0], x, res[1]);
    */
}

struct reverse {
    // helper struct to get the lower half of a SIMD register and zip it with itself
    // it returns index 0, 0, 1, 1, ... N/2, N/2
    static constexpr unsigned get(unsigned index, unsigned size) { return size - index - 1; }
};

template <class arch_t, class T>
constexpr auto reverse_index = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>, arch_t, reverse>();

double cheb_eval_6(double x, const double *__restrict__ c) {
    using T = double;
    /*
    const auto x2 = 2 * x;
    const auto x20 = x2 * c[0];

    // Two additional iterations
    const auto c00 = c[2] - c[0] + c[1] + x20;
    const auto c10 = c[3] - c[1] + x20;

    const auto c0 = c[4] - c00 + c10 * x2;
    const auto c1 = c[5] - c10 + c00 * x2;

    return std::fma(x, c0, c1);
    */
    /*
    using batch_t = xsimd::make_sized_batch_t<double, 2>;
    using arch_t = batch_t::arch_type;
    const auto x2 = batch_t{2 * x};
    const auto x20 = batch_t{x2 * c[0]};
    const auto c1 = batch_t{c[1], 0};
    using batch_t6 = xsimd::make_sized_batch_t<double, 8>;
    union vec{
        batch_t6 vec;
        batch_t scalar[4];
    };

    const auto v = vec {batch_t6::load_aligned(c)};
    // Populate c01, c23, and c45 using the scalar member of the union
    const auto c01 = v.scalar[0];
    const auto c23 = v.scalar[1];
    const auto c45 = v.scalar[2];

    const auto order4 = c23 - c01 + c1 + x20;
    const auto reverse4 = xsimd::swizzle(order4, reverse_index<arch_t, double>);
    const auto res = xsimd::fma(reverse4, x2, c45 - order4);
    return xsimd::fma(res.get(0), x, res.get(1));
    */
    using batch_t = xsimd::make_sized_batch_t<double, 2>;
    using arch_t = batch_t::arch_type;

    // Load all required values at once
    const auto c23 = batch_t::load_aligned(c + 2) - batch_t::load_aligned(c);

    // Precompute common expressions
    const auto x2 = batch_t{2 * x};
    const auto c0 = batch_t(c[0]);
    const auto c1 = c23 - batch_t{c[1], 0};

    // Compute order4 and reverse4
    const auto order4 = xsimd::fma(x2, c0, c1);
    const auto c45 = batch_t::load_aligned(c + 4) - order4;
    const auto reverse4 = xsimd::swizzle(order4, reverse_index<arch_t, double>);

    // Compute the final result
    const auto res = xsimd::fma(reverse4, x2, c45);
    return xsimd::fma(res.get(0), x, res.get(1));
}

template <int ORDER, typename T>
inline T cheb_eval(T x, const T *c) {
    const T x2 = 2 * x;

    T c0 = c[0];
    T c1 = c[1];
    for (int i = 2; i < ORDER; ++i) {
        T tmp = c1;
        c1 = c[i] - c0;
        c0 = tmp + c0 * x2;
    }
    return c1 + c0 * x;
}

template <int ORDER, typename T>
T cheb_eval_fast(T x, const T *c) {
    return cheb_eval<ORDER>(x, c);
}

template <int ORDER, typename T>
inline T cheb_eval_vector(T x, const T *c) {
    if constexpr (ORDER == 4) {
        return cheb_eval_4(x, c);
    }
    if constexpr (ORDER == 6) {
        return cheb_eval_6(x, c);
    }
    return cheb_eval<ORDER>(x, c);
}

template <int ORDER, typename T>
bool test_correctness(T x, const T *c) {
    T result_cheb_eval = cheb_eval<ORDER>(x, c);
    T result_cheb_eval_generic = cheb_eval_fast<ORDER>(x, c);
    auto valid = 1 - result_cheb_eval / result_cheb_eval_generic < 1e-14;
    if (!valid) {
        std::cout << "order " << ORDER << " " << result_cheb_eval << " " << result_cheb_eval_generic << std::endl;
    }
    // Consider two results as equal if the absolute difference is less than a small threshold
    return valid;
}

template <int ORDER, typename T>
bool test_correctness_vector(T x, const T *c) {
    T result_cheb_eval_vector = cheb_eval_generic(ORDER, x, c);
    T result_cheb_eval_generic = cheb_eval_vector<ORDER>(x, c);

    // Consider two results as equal if the absolute difference is less than a small threshold
    return 1 - result_cheb_eval_generic / result_cheb_eval_vector < 1e-14;
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
         .minEpochIterations(108543525)
         .run("generic", [&] { doNotOptimizeAway(cheb_eval_generic(Orders, x, c)); })
         .run("if", [&] { doNotOptimizeAway(cheb_eval_if(Orders, x, c)); })
         .run("switch", [&] { doNotOptimizeAway(cheb_eval_switch(Orders, x, c)); })
         .run("template", [&] { doNotOptimizeAway(cheb_eval<Orders>(x, c)); })
         .run("fast", [&] { doNotOptimizeAway(cheb_eval_fast<Orders>(x, c)); })
         .run("vector", [&] { doNotOptimizeAway(cheb_eval_vector<Orders>(x, c)); }),
     ...);
}

int main() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1, 1);
    alignas(sizeof(double) * 8) double c[128];
    for (double &i : c)
        i = dis(gen);

    double x{0.1};

    run_tests<double, 4, 6, 8, 10, 12>(x, c);
    run_benchmarks<double, 4, 6, 8, 10, 12>(x, c);

    return 0;
}
