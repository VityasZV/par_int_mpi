#pragma once

namespace mpi_info {
    struct MPI {
        int process_id;
        int amount_of_processes;
        double wtime;
        int ierr;
        MPI(int *argc_ptr, char ***argv_ptr);
        ~MPI();
    };
}// mpi_info
