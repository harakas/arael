// A program using the shared library in shared.cpp.
#include <cstdio>

extern "C" double consumer_fit_slope();

int main() {
    double m = consumer_fit_slope();
    std::printf("m %.6f\n", m);
    return m > 1.9 && m < 2.1 ? 0 : 1;
}
