#include "parralel_integral.hpp"

#include "mpi_info.hpp"
#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <cmath>
#include <stdio.h>
#include <cstdlib>

namespace parallel_integral {

    namespace {
        static const double kAccuracy = 0.000001;

        double Function(const double& arg) {
            return 3*std::pow(arg, -1) + 2;
        }

        struct AccuracyParameters {
            unsigned long parts;
            double step;
            const Limits limits;
            AccuracyParameters(std::ifstream& input_file): limits(input_file){
                parts = 2;
                step = (limits.right - limits.left) / parts;
            }
            void Increment() {
                parts *= 2;
                step = (limits.right - limits.left) / parts;
            }
            operator double() {
                return step;
            }
        };

    } //namespace

    Limits::Limits(std::ifstream& input_file) {
        input_file.open("input.txt");
        input_file >> left >> right;
        input_file.close();
    }

    ResultAndTime ComputeIntegral(int *argc_ptr, char ***argv_ptr) {
        std::ifstream input_file;
        AccuracyParameters accuracy_parameters(input_file);
        double previous_result = 0, result = 0;
        unsigned long i = 0;
        mpi_info::MPI mpi_statistics(argc_ptr, argv_ptr);
        double mpi_time = MPI_Wtime();
        if (mpi_statistics.ierr != MPI_SUCCESS) {
            return ResultAndTime(-1, -1); //костыли потому что не получилось указать компилятору использование std::exception
        }
        //
        //  Get the number of processes.
        //
        mpi_statistics.ierr = MPI_Comm_size(MPI_COMM_WORLD, &mpi_statistics.amount_of_processes);
        //
        //  Get the individual process ID.
        //
        mpi_statistics.ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_statistics.process_id);
        /*
            Giving pieces from accuracy_parameters.parts for each process,
            and then in for-cycle each process is using threads for computation of sum. 
        */
        do {
            if (mpi_statistics.process_id == 0){
                //std::cout << "PROCESS: "<<mpi_statistics.process_id << "changes params and broadcast" << std::endl;
                previous_result = result;
                result = 0;
                accuracy_parameters.Increment();
                //std::cout<< "Process: " << mpi_statistics.process_id <<" has accuracy - " << accuracy_parameters.parts << std::endl;
                double temp = 0;
            }
            else {
                double temp = 0;
                previous_result = result;
                result = 0;
                accuracy_parameters.Increment();
                //std::cout<< "Process: " << mpi_statistics.process_id <<" has accuracy - " << accuracy_parameters.parts << std::endl;
            }
            double result_in_process = 0;
            unsigned long for_work = accuracy_parameters.parts / mpi_statistics.amount_of_processes;
            unsigned long remains = accuracy_parameters.parts % mpi_statistics.amount_of_processes;
            unsigned long start = mpi_statistics.process_id * for_work;
            unsigned long finish = start + for_work;// + mpi_statistics.process_id == 0 ? remains : 0;
            //std::cout << "ID:" << mpi_statistics.process_id <<" start:" << start << " finish:" << finish << std::endl;
            #pragma omp parallel shared(accuracy_parameters) reduction(+:result_in_process)
            {
                #pragma omp for
                for (i = start; i < finish; ++i) {
                    double x = accuracy_parameters.limits.left + i * accuracy_parameters.step;
                    result_in_process += accuracy_parameters.step * Function(x + accuracy_parameters.step / 2);
                }
            }
            if (mpi_statistics.process_id != 0){
                //std::cout << "PROCESS: "<<mpi_statistics.process_id << "sending result: " <<result_in_process<< " to master" << std::endl;
                MPI_Send(&result_in_process, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); //sending result to master
            }
            if (mpi_statistics.process_id == 0) {
                result += result_in_process; //added masters part
                int count = 0;
                //std::cout << "PROCESS: "<<mpi_statistics.process_id << "is waiting for others finally" << std::endl; //here was infinite loop last time
                for (int c = 1; c < mpi_statistics.amount_of_processes; ++c){
                    MPI_Recv(&result_in_process, 1, MPI_DOUBLE, c, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    result += result_in_process;
                    count = c;
                }
                //std::cout << "PROCESS: "<<mpi_statistics.process_id << "waited for count=" << count << std::endl;
                //MPI_Waitall(mpi_statistics.amount_of_processes - 1, &requests[1], &statuses[1]); // дождался всех 
                //std::cout << "DEBUG: Master waited for all " << mpi_statistics.amount_of_processes - 1 << std::endl;
                //std::cout << "RESULT = " << result << std::endl;
                for (int c = 1; c < mpi_statistics.amount_of_processes; ++c){
                    MPI_Send(&result, 1, MPI_DOUBLE, c, 0, MPI_COMM_WORLD); //sending result to rest processes
                }
            }
            else {
                MPI_Recv(&result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //receiving result
            }

        } while (fabs(result - previous_result) >= kAccuracy);
        mpi_time = MPI_Wtime() - mpi_time;
        if (mpi_statistics.process_id == 0){
            return ResultAndTime(result, mpi_time);
        }
        else return ResultAndTime(0, mpi_time);
    }
}

