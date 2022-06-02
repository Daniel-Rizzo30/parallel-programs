/*******************************************************************************
Title : hwk3_Daniel.Rizzo30.c
Author : Daniel Rizzo
Class : CSCI 49365 Parallel Computing, Stewart Weiss
Created on : October 15th, 2021
Description : Parallel program using MPI to output the saddle point of a matrix,
which is provided by naming an input file, in binary. Saddle point is the 
smallest in its column and the largest in its row.
Purpose : First MPI assignment in CSCI 49365.
Usage : saddle_point <matrixfile>  (implied usage of mpirun -np N, mpirun -H...)
Build with : mpicc -Wall -g -o saddle_point hwk3_Daniel.Rizzo30.c
Debug on : mpicc -g -DDEBUG_ON -Wall -o saddle_point hwk3_Daniel.Rizzo30.c
Modifications: Many; October 20th, 2021: Added error handling and a reduce 
function to collect any error messages when allocating data in all processes
October 24th, 2021: Testing, playing with creating matrices, realized how 
precise the format of the input must be
October 26th, 2021: Took care of all compiler warnings
October 27th, 2021: Finished testing, left in (but commented out) all printf
statements where I was testing, so this file can be useful in future studying
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

typedef double  Element_type; // The elements taken are stored as doubles
#define MPI_TYPE   MPI_DOUBLE // As specified in the assignment instructions
#define ROOT                0

#define PROMPT_MSG          0
#define RESPONSE_MSG        1

#define MPI_INIT_ERROR     -1 // MPI not init'd by program
#define SUCCESS             0 // No error
#define MALLOC_ERROR        1 // Error allocating storage
#define OPEN_FILE_ERROR     2 // Error with OPENING given file
#define FILE_READ_ERROR     3 // Error with contents of file
#define BAD_MATRIX_SIZE     4 // Error constructing matrix based on size
#define BAD_ELEMENT_SIZE    5 // Not used since const type, still implemented
#define USAGE_ERROR         6 // Error with usage



/**
 * @brief read_and_distribute_matrix ( ... ) will open a file and read in a 
 * matrix to the ROOT process, which will then distribute the full matrix to
 * all other processes. 
 * Only Process 0, the ROOT, will read in the matrix.
 * This function was modeled heavily after the function 
 * read_and_distribute_matrix_byrows ( ... ) found in Professor Weiss' 
 * Chapter 5 Lecture Notes
 * This function was taken/modified from free software under copyright of 
 * Stewart Weiss (found in /common folder).
 */
void read_and_distribute_matrix(
        char *filename, /* [ IN ] name of file to read */
        void ***matrix, /* [ OUT ] matrix to fill with data */
        void **matrix_storage, /* [ OUT ] linear storage for the matrix */
        MPI_Datatype dtype, /* [ IN ] matrix element type */
        int *nrows, /* [ OUT ] number of rows in matrix */
        int *ncols, /* [ OUT ] number of columns in matrix */
        int *errval, /* [ OUT ] success / error code on return */
        MPI_Comm comm); /* [ IN ] communicator handle */

/** 
 *  @brief alloc_matrix(r,c,e, &Mstorage, &M, &err)
 *  If &err is SUCCESS, on return it allocated storage for two arrays in
 *  the heap. Mstorage is a linear array large enough to hold the elements of
 *  an r by c 2D matrix whose elements are e bytes long. The other, M, is a 2D
 *  matrix such that M[i][j] is the element in row i and column j.
 *  This function was taken/modified from free software under copyright of 
 *  Stewart Weiss (found in /common folder).
 */
void alloc_matrix(
        int     nrows,          /* number of rows in matrix                   */
        int     ncols,          /* number of columns in matrix                */
        size_t  element_size,   /* number of bytes per matrix element         */
        void  **matrix_storage, /* address of linear storage array for matrix */
        void ***matrix,         /* address of start of matrix                 */
        int    *errvalue);      /* return code for error, if any              */

/**
 * @brief collect_and_print_saddle_point(dtype, comm) will use the root process
 * to collect all of the other processes' findings on where the saddle point 
 * is. If P0 hasn't found the saddle point itself, then each process will send
 * its findings, through handshaking, to P0. The 'findings' will either be a 
 * pair of -1's, indicating no saddle point found, or two coordinates, 
 * indicating the saddle point was found at that coordinate by the sender.
 * Based off of collect_and_print_matrix() in Ch.5 Lecture Notes
 * This function was taken/modified from free software under copyright of 
 * Stewart Weiss (found in /common folder).
 */
void collect_and_print_saddle_point(
        int *saddle_point,    /* [ IN ] saddle point info to be sent */
        MPI_Datatype dtype,    /* [ IN ] matrix element type          */
        MPI_Comm comm);          /* [ IN ] communicator handle          */

/**
 * @brief print_error(errvalue) will print out an error statement
 * based on the errvalue given.
 */
void print_error( 
        int errvalue); /* Is the value that determines the err statement */



int main(int argc, char *argv [])
{
    int id; /* Process rank */
    int p; /* Total num of processes */
    Element_type ** matrix_pointers; /* Doubly-subscripted array, use M[][] */
    Element_type * matrix_storage; /* Local linear copies of array elements */
    int nrows; /* Rows in matrix */
    int ncols; /* Columns in matrix */
    int error;

    int i; // for loop vars
    int j;
    int k;
    Element_type extremum; // For storing the max/min in the row/column
    int saddle_point[] = {-1, -1}; // Coordinates, -1's signal 'not found'
    int saddle_point_flag = 0;

    MPI_Init(&argc, &argv);

    // Debug code here
    #ifdef DEBUG_ON
        /* To debug, compile this program with the -DDEBUG_ON option,
        which defines the symbol DEBUG_ON, and run the program as usual
        with mpirun.
        When the output appears on the terminal, listing the pids of the
        processes and which hosts they are on, choose the lowest
        pid P on the machine you are connected to.
        Open a new terminal window and in that window issue the command
        gdb --pid P
        (or gdb -p P on some systems)
        and after gdb starts, go up the stack to main by entering the
        command
        up 3
        (main will be three stacks frames above your current frame,
        which should be nanosleep.)
        Then enter the command
        set var z = 1
        to break the while loop. You can now run ordinary gdb commands to
        debug this process. This should be process 0.
        Repeat these steps for each other process that you created in
        the mpirun command.
        */
        #include <unistd.h>
        int z = 0;
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        printf("PID %d on %s ready for attach\n", getpid(), hostname);
        fflush(stdout);
        while (0 == z)
        sleep(5);
    #endif 

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (id == ROOT && argc < 2) {
        print_error(USAGE_ERROR);
        // Change this to MPI Finalize, and have all p's hit this
        // As in assignment 4
        MPI_Abort(MPI_COMM_WORLD, USAGE_ERROR);
    }

    //printf("%s\t%d\n", argv[1], id); // Test

    /* Read the matrix from the file named on the command line and
    distribute to each process. Check error on return */
    read_and_distribute_matrix(argv[1],
        (void*) &matrix_pointers,
        (void*) &matrix_storage,
        MPI_TYPE, &nrows, &ncols, &error,
        MPI_COMM_WORLD);

    if (id == ROOT && error != SUCCESS) {
        print_error(error);
        MPI_Abort(MPI_COMM_WORLD, error);
    }
    // if (id == ROOT) { // Testing
    //     printf("\n%d:\n", id); // Testing
    //     for (int a = 0; a < nrows; a++) { // Testing
    //         for (int b = 0; b < ncols; b++) {
    //             printf("%f\t", matrix_pointers[a][b]);
    //         }
    //         printf("\n");
    //     }
    // }

    for (i = id; i < nrows && !saddle_point_flag; i += p) {

        // Find the max
        for (j = 0; j < ncols; j++) {
            if (j == 0 || matrix_pointers[i][j] > extremum) {
                extremum = matrix_pointers[i][j];
            }
        }

        // Check if any of the maxes (incl. duplicates) are saddle points
        for (j = 0; j < ncols && !saddle_point_flag; j++) {
            if (matrix_pointers[i][j] == extremum) {
                // Find the min in the column of the row's max
                for (k = 0; k < nrows; k++) {
                    // Elements within the col can be equal to the extremum
                    if (matrix_pointers[k][j] < extremum) {
                        break; // Break from inner for loop, not a saddle pt
                    }
                    else if (k == nrows - 1 && 
                            !(matrix_pointers[k][j] < extremum)) {
                        // This would be a saddle point
                        saddle_point[0] = i; // Store 
                        saddle_point[1] = j;
                        saddle_point_flag = 1; // Found one, skip the rest
                    }
                }
            }
        }
    }
    //MPI_Barrier(MPI_COMM_WORLD); // Testing ?
    //printf("%d\t%d\t%d\n", saddle_point[0], saddle_point[1], id); // Testing
    collect_and_print_saddle_point(
        saddle_point, 
        MPI_TYPE, 
        MPI_COMM_WORLD
    );

    free(matrix_pointers);
    free(matrix_storage);
    MPI_Finalize();
    return 0;
}



void print_error(int errvalue) {
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (! mpi_initialized) {
        errvalue = MPI_INIT_ERROR;
    }
    //printf("%d\n", errvalue); // Test
    switch(errvalue) {
        case MPI_INIT_ERROR :
            fprintf(stderr, "Error: MPI was not initialized.\n");
            break; // Need break statements so it doesn't cont to next case
        case MALLOC_ERROR :
            fprintf(stderr, "Error: Could not allocate data.\n");
            break;
        case OPEN_FILE_ERROR :
            fprintf(stderr, "Error: Could not open file.\n");
            break;
        case FILE_READ_ERROR :
            fprintf(stderr, "Error: Could not read file "
                     "contents properly.\n");
            break;
        case BAD_MATRIX_SIZE : 
            fprintf(stderr, "Error: Cannot construct matrix "
                     "based on dimensions given.\n");
            break;
        case BAD_ELEMENT_SIZE : // Won't be used - const type double
            fprintf(stderr, "Error: Invalid matrix element "
                     "type given.\n");
            break;
        case USAGE_ERROR :
            fprintf(stderr, "Error: Usage: saddlepoint <matrixfile>, "
                     "where matrixfile contains a binary matrix.\n");
            break;
        default: 
            break;
    }
}

void collect_and_print_saddle_point(
        int *saddle_point,     /* [ IN ] saddle point info to be sent */
        MPI_Datatype dtype,    /* [ IN ] matrix element type          */
        MPI_Comm comm)         /* [ IN ] communicator handle          */
{
    int id; /* process rank process */
    int p; /* number of processes in communicator group */
    int prompt; /* synchronizing variable */
    int i;
    void *sp = saddle_point;

    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &p);

    if (ROOT == id) {
        int saddle_point_flag = 0; // Flag to check if output already
        /* Check if P0 found the saddle point */
        if (saddle_point[0] != -1 && saddle_point[1] != -1 &&
             saddle_point_flag == 0) {
            saddle_point_flag = 1;
            printf("%d\t%d\n", 
            saddle_point[0], saddle_point[1]); // Print saddle point
        }
        if (p > 1) {
            for (i = 1; i < p; i++) {

                /* Send a message to process i telling it to send data */
                MPI_Send(&prompt, 1, MPI_INT, i, PROMPT_MSG, comm);
                /* Wait for data to arrive from process i */
                /* The data it gets is coordinates, so always 2 ints */
                /* Use *saddle_point to store data, can be overwritten */
                /* No status check is needed - use IGNORE */
                MPI_Recv(sp, 2, MPI_INT,
                            i, RESPONSE_MSG, comm, MPI_STATUS_IGNORE);

                if (saddle_point[0] != -1 && saddle_point[1] != -1 && 
                     saddle_point_flag == 0) {
                    saddle_point_flag = 1; // A saddle point was found
                    printf("%d\t%d\n", 
                    saddle_point[0], saddle_point[1]); // Print saddle point
                }
            }
        }

        if (saddle_point_flag == 0) { // Still no saddle point found
            printf("No saddle point\n"); // The matrix doesn't have one.
        }
    }

    else {  // What all other processes do if not aborted
            /* Wait for prompt message from process 0, dummy msg */
            /* No status check is needed - use IGNORE */
            MPI_Recv(&prompt, 1, MPI_INT, ROOT, 
                        PROMPT_MSG, comm, MPI_STATUS_IGNORE);
            /* On receiving it, send its own saddle point coordinates */
            MPI_Send(sp, 2, MPI_INT, ROOT, RESPONSE_MSG, comm);
    }
}

void alloc_matrix(
        int     nrows,          /* number of rows in matrix                   */
        int     ncols,          /* number of columns in matrix                */
        size_t  element_size,   /* number of bytes per matrix element         */
        void  **matrix_storage, /* address of linear storage array for matrix */
        void ***matrix,         /* address of start of matrix                 */
        int    *errvalue)       /* return code for error, if any              */
{
    int   i;
    void *ptr_to_row_in_storage; /* pointer to a place in linear storage array
                                    where a row begins                        */
    void **matrix_row_start;     /* address of a 2D matrix row start pointer
                                    e.g., address of (*matrix)[row]           */
    size_t total_bytes;          /* amount of memory to allocate              */

    //printf("alloc_matrix called with r=%d,c=%d,e=%d\n",
    //nrows, ncols, element_size); // Output dimensions for testing

    total_bytes = nrows * ncols * element_size;

    /* Step 1: Allocate an array of nrows * ncols * element_size bytes  */
    *matrix_storage = malloc(total_bytes);
    if (NULL == *matrix_storage) {
        /* malloc failed, so set error code and quit */
        *errvalue = MALLOC_ERROR; // Error sent back to calling function
        return;
    }

    memset(*matrix_storage, 0, total_bytes);

    /* Step 2: To create the 2D matrix, first allocate an array of nrows
       void* pointers */
    *matrix = malloc(nrows * sizeof(void*));
    if (NULL == *matrix) {
        /* malloc failed, so set error code and quit */
        *errvalue = MALLOC_ERROR; // Error sent back to calling funct
        return;
    }


    /* Step 3: (The hard part) We need to put the addresses into the
       pointers of the 2D matrix that correspond to the starts of rows
       in the linear storage array. The offset of each row in linear storage
       is a multiple of (ncols * element_size) bytes.  So we initialize
       ptr_to_row_in_storage to the start of the linear storage array and
       add (ncols * element_size) for each new row start.
       The pointers in the array of pointers to rows are of type void*
       so an increment operation on one of them advances it to the next ptr.
       Therefore, we can initialize matrix_row_start to the start of the
       array of pointers, and auto-increment it to advance it.
    */

    /* Get address of start of array of pointers to linear storage,
       which is the address of first pointer, (*matrix)[0]   */
    matrix_row_start = (void*) &(*matrix[0]);

    /* Get address of start of linear storage array */
    ptr_to_row_in_storage = (void*) *matrix_storage;

    /* For each matrix pointer, *matrix[i], i = 0... nrows-1,
       set it to the start of the ith row in linear storage */
    for (i = 0; i < nrows; i++) {
        /* matrix_row_start is the address of (*matrix)[i] and
           ptr_to_row_in_storage is the address of the start of the
           ith row in linear storage.
           Therefore, the following assignment changes the contents of
           (*matrix)[i]  to store the start of the ith row in linear storage
        */
        *matrix_row_start = (void*) ptr_to_row_in_storage;

        /* advance both pointers */
        matrix_row_start++;     /* next pointer in 2d array */
        ptr_to_row_in_storage +=  ncols * element_size; /* next row */
    }
    *errvalue = SUCCESS;
}

void read_and_distribute_matrix(
        char *filename, /* [ IN ] name of file to read */
        void ***matrix, /* [ OUT ] matrix to fill with data */
        void **matrix_storage, /* [ OUT ] linear storage for the matrix */
        MPI_Datatype dtype, /* [ IN ] matrix element type */
        int *nrows, /* [ OUT ] number of rows in matrix */
        int *ncols, /* [ OUT ] number of columns in matrix */
        int *errval, /* [ OUT ] sucess / error code on return */
        MPI_Comm comm) /* [ IN ] communicator handle */
{

    int id; /* process rank process */
    int p; /* number of processes in communicator group */
    size_t element_size; /* number of bytes in matrix element type */
    int mpi_initialized; /* flag to check if MPI_Init was called already */
    FILE * file; /* input file stream pointer */
    int error_values_or; /* result of reduce to find alloc error */

    /* Make sure we are being called by a program that init - ed MPI */
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        *errval = MPI_INIT_ERROR; // HANDLE ERROR
        return;
    }

    /* Get process rank and the number of processes in group */
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &id);

    /* Get the number of bytes in a matrix element */
    element_size = sizeof (dtype); // Changed from getsize()
    if (element_size <= 0) {
        *errval = BAD_ELEMENT_SIZE;
        return;
    }

    if (ROOT == id) {
    /* Process 0 opens the binary file containing the matrix and
    reads the first two numbers, which are the number of rows and
    columns respectively. */
        file = fopen(filename, "rb"); // DO NOT ADD SPACE AROUND rb
        if (NULL == file) {
            *errval = OPEN_FILE_ERROR;
            return;
        }
        else {
            fread(nrows, sizeof (int), 1, file);
            fread(ncols, sizeof (int), 1, file);
        }
    }

    /* Process 0 broadcasts the numbers of rows to all other processes */
    MPI_Bcast(nrows, 1, MPI_INT, ROOT, comm);

    if (0 == *nrows) {
        *errval = BAD_MATRIX_SIZE; // ROOT will check in main
        return; // for error value, then output msg and MPI_Abort
    }

    /* Process 0 broadcasts the numbers of columns to all other processes */
    MPI_Bcast(ncols, 1, MPI_INT, ROOT, comm);

    if (0 == *ncols) {
        *errval = BAD_MATRIX_SIZE;
        return;
    }

    // printf("%d\t%d\t%d\n", *nrows, *ncols, id); // Testing
    // Check if rows/cols is > 100,000,000, then term if it is???

    /* Each process creates its storage for accessing the 2D matrix */
    alloc_matrix(*nrows, *ncols, element_size,
                    matrix_storage, matrix, errval);

    /* Check for errors in all processes after allocating, use MPI_Reduce */
    /* Performing the logical-or reduction of these error values will ensure
    that if any single process had trouble allocating data, ROOT will know 
    through this Reduce, as MALLOC_ERROR == 1, so if error_values_or 
    is 1, then some process had an issue allocating */
    MPI_Reduce(errval, &error_values_or, 1, MPI_INT, 
                MPI_LOR, ROOT, MPI_COMM_WORLD);

    if (id == ROOT && error_values_or == MALLOC_ERROR) {
        *errval = error_values_or; // Send back malloc error
        return; // return to main and check for error
    }

    /* total number of matrix elements to send */
    int num_elements = (*nrows) * (*ncols);

    if (ROOT == id) {
        size_t nelements_read; /* result of read operation */
        /* Process 0 reads the file into its own linear storage. */
        nelements_read = fread(*matrix_storage, element_size,
                                num_elements, file);
        // printf("After read: read %d, should have read %d, end flag is %d, "
        //        "error flag is %d", nelements_read, num_elements, 
        //        feof(file), ferror(file)); // Testing
        /* Check that the number of items read matches the number requested */
        if (nelements_read != num_elements ||
                feof(file) != 0 || // Says if file has too few elements.
                ferror(file) != 0) { // Check for any other error
            *errval = FILE_READ_ERROR; // Set error
            fclose(file); // Clean up and return
            return;
        }
        /* Process 0 closes the file */
        fclose(file);
    }

    /* Process 0 broadcasts the matrix to all other processes . */
    MPI_Bcast(*matrix_storage, num_elements, dtype, ROOT, comm);
    *errval = SUCCESS; // Operation was a success
}