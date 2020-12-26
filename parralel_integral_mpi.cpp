// parallel_integral_mpi.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include "parralel_integral.hpp"
#include <iostream>
#include <ctime>
#include <mpi.h>	
int main(int argc, char **argv)
{
    
    parallel_integral::ResultAndTime result_and_time = parallel_integral::ComputeIntegral(&argc, &argv);
    if (result_and_time.time == -1) {
        std::cout << "MPI Init returned non-zero";
    }
    else {
        std::cout << "result = " << result_and_time.result;
        std::cout << "\ntime = " << result_and_time.time << std::endl;
    }
    MPI_Finalize();
    return 0;
}