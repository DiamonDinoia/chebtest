#include <nanobench.h>
#include <random>

double cheb_eval_generic(int order, double x, const double *c);
double cheb_eval_if(int order, double x, const double *coeffs);
double cheb_eval_switch(int order, double x, const double *c);

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

int main() {
    std::mt19937 gen(1);
    std::uniform_real_distribution<> dis(-1, 1);
    double c[32];
    for (size_t i = 0; i < 32; ++i)
        c[i] = dis(gen);

    double x{0.1};
    using ankerl::nanobench::doNotOptimizeAway;
    {
        auto b = ankerl::nanobench::Bench().unit("eval").title("order 6").minEpochIterations(10000000);
        b.run("generic", [&] { doNotOptimizeAway(cheb_eval_generic(6, x, c)); });
        b.run("template", [&] { doNotOptimizeAway(cheb_eval<6>(x, c)); });
        b.run("if", [&] { doNotOptimizeAway(cheb_eval_if(6, x, c)); });
        b.run("switch", [&] { doNotOptimizeAway(cheb_eval_switch(6, x, c)); });
    }
    {
        auto b = ankerl::nanobench::Bench().unit("eval").title("order 8").minEpochIterations(10000000);
        b.run("generic", [&] { doNotOptimizeAway(cheb_eval_generic(8, x, c)); });
        b.run("template", [&] { doNotOptimizeAway(cheb_eval<8>(x, c)); });
        b.run("if", [&] { doNotOptimizeAway(cheb_eval_if(8, x, c)); });
        b.run("switch", [&] { doNotOptimizeAway(cheb_eval_switch(8, x, c)); });
    }
    {
        auto b = ankerl::nanobench::Bench().unit("eval").title("order 10").minEpochIterations(10000000);
        b.run("generic", [&] { doNotOptimizeAway(cheb_eval_generic(10, x, c)); });
        b.run("template", [&] { doNotOptimizeAway(cheb_eval<10>(x, c)); });
        b.run("if", [&] { doNotOptimizeAway(cheb_eval_if(10, x, c)); });
        b.run("switch", [&] { doNotOptimizeAway(cheb_eval_switch(10, x, c)); });
    }
    {
        auto b = ankerl::nanobench::Bench().unit("eval").title("order 12").minEpochIterations(10000000);
        b.run("generic", [&] { doNotOptimizeAway(cheb_eval_generic(12, x, c)); });
        b.run("template", [&] { doNotOptimizeAway(cheb_eval<12>(x, c)); });
        b.run("if", [&] { doNotOptimizeAway(cheb_eval_if(12, x, c)); });
        b.run("switch", [&] { doNotOptimizeAway(cheb_eval_switch(12, x, c)); });
    }

    return 0;
}
