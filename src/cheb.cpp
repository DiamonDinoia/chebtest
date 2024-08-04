double cheb_eval_generic(int order, double x, const double *__restrict__ c) {
    // note (RB): uses clenshaw's method to avoid direct calculation of recurrence relation of
    // T_i, where res = \Sum_i T_i c_i
    const double x2 = 2 * x;

    double c0 = c[0];
    double c1 = c[1];

    for (int i = 2; i < order; ++i) {
        double tmp = c1;
        c1 = c[i] - c0;
        c0 = tmp + c0 * x2;
    }

    return c1 + c0 * x;
}


double cheb_eval_if(int order, double x, const double *__restrict__ c) {
    if (order == 6)
        return cheb_eval_generic(6, x, c);
    else if (order == 8)
        return cheb_eval_generic(8, x, c);
    else if (order == 10)
        return cheb_eval_generic(10, x, c);
    else if (order == 12)
        return cheb_eval_generic(12, x, c);

    return cheb_eval_generic(order, x, c);
}

double cheb_eval_switch(int order, double x, const double *__restrict__ c) {
    switch (order) {
    case 6: {
        return cheb_eval_generic(6, x, c);
        break;
    }
    case 8: {
        return cheb_eval_generic(8, x, c);
        break;
    }
    case 10: {
        return cheb_eval_generic(10, x, c);
        break;
    }
    case 12: {
        return cheb_eval_generic(12, x, c);
        break;
    }
    default:
        return cheb_eval_generic(order, x, c);
    }
}


