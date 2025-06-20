#include <bits/stdc++.h>

double truncate(double number_val, int n)
{
    bool negative = false;
    if (number_val == 0) {
        return 0;
    } else if (number_val < 0) {
        number_val = -number_val;
        negative = true;
    }
    double pre_digits = std::log10(number_val) + 1;
    if (pre_digits < 17) {
        double post_digits = 17 - pre_digits;
        double factor = std::pow(10, post_digits);
        number_val = std::round(number_val * factor) / factor;
        factor = std::pow(10, n);
        number_val = std::trunc(number_val * factor) / factor;
    } else {
        number_val = std::round(number_val);
    }
    if (negative) {
        number_val = -number_val;
    }
    return number_val;
}