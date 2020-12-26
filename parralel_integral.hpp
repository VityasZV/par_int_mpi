#pragma once
#include <fstream>

namespace parallel_integral {

    struct Limits {
        double left;
        double right;
        Limits(std::ifstream& input_file);
    };
    struct ResultAndTime {
        double result;
        double time;
        ResultAndTime(const double &result,const double &time): result(result), time(time){}
    };
    // Parallel computing of integral
    ResultAndTime ComputeIntegral(int *argc_ptr, char ***argv_ptr);

}// parallel_integral
