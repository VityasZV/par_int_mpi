#include "mpi_info.hpp"
#include <mpi.h>

namespace mpi_info {
    MPI::MPI(int *argc_ptr, char ***argv_ptr){
        ierr = MPI_Init (argc_ptr, argv_ptr);
        //  Initialize MPI.
    }
    MPI::~MPI(){
        //nothing
    }
    int MPI::amount_of_processes;
}// mpi_info
	
