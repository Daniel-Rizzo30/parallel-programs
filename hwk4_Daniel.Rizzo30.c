/*******************************************************************************
Title : hwk4_Daniel.Rizzo30.c
Author : Daniel Rizzo
Class : CSCI 49365 Parallel Computing, Stewart Weiss
Created on : November 19th, 2021
Description : Parallel program using MPI to output the temperature of the 
middle of a plane of material when the WHOLE plate is in steady state. 
Purpose : Second MPI assignment in CSCI 49365.
Usage : steady_state <txtfile> (implied usage of mpirun -np N, mpirun -H...)
Build with : mpicc -Wall -g -o steady_state hwk4_Daniel.Rizzo30.c
Debug on : mpicc -g -DDEBUG_ON -Wall -o steady_state hwk4_Daniel.Rizzo30.c
Modifications : November 21st, 2021: Finished outputting result and fixed 
formatting of the program. Began Testing.
November 22nd: Fixed seeding in init_random function after rereading some of 
the chapter 8 lecture notes. C already uses an LFG and creates the lags 
using an LCG, so a first seed is just needed when initializing state. 
Began testing. 
November 24th: Cont. testing - changed convergence threshold and tablesize to 
better fit the parallel aspect of this program. Also added fix for the case of
p > width - 2 by including an extra if statement when initializing local array
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h> // For absolute value
#include <time.h> // For seeding with time()
#include "mpi.h"

#define ROOT                0
#define RESPONSE_MSG        1

#define MPI_INIT_ERROR     -1 // MPI not init'd by program
#define SUCCESS             0 // No error
#define MALLOC_ERROR        1 // Error allocating storage
#define OPEN_FILE_ERROR     2 // Error with OPENING given file
#define FILE_READ_ERROR     3 // Error with contents of file
#define BAD_MATRIX_SIZE     4 // Error constructing matrix based on size
#define MALLOC_TABLE_ERROR  5 // Error allocating for lag table
#define USAGE_ERROR         6 // Error with usage

#define NORTH   1 // Cardinal directions
#define EAST    2
#define SOUTH   3
#define WEST    4

#define CONVERGENCE_THRESHOLD  0.04

/******************************************************************************/
// These two functions taken from utilities_basic.c from chapter08 folder
// These functions were taken/modified from free software under copyright of 
// Stewart Weiss (found in chapter08/ folder).
/** uniform_random()  returns a uniformly distributed random number in [0,1]
 *  @return double  a pointer to the state array allocated for random()
 *  @pre           Either init_random() should have been called or srandom()
 */
double uniform_random() {
    return (double) (random()) / RAND_MAX;
}

/** init_random()  initializes the state for the C random() function
 *  @param  int    state_size [IN] Size of state array for random to use
 *  @param  int    id [IN] Used as unique seeding for this process
 *  @return char*  a pointer to the state array allocated for random()
 *  @post          After this call, an array of size state_size*sizeof(char) has
 *                 been allocated and initialized by C initstate(). It must be
 *                 freed by calling free()
 */
char* init_random(int state_size, int id) { // use id to compute seed for each p
    char *state;
    state  = (char*) malloc(state_size * sizeof(char));
    if (NULL != state)
        // First parameter is seed, previously time(NULL), now the seed is 
        // based off of the id + 1 (so ROOT isn't always seeded with 0), 
        // multiplied by the time (in seconds since 1970)
        initstate((id + 1) * time(NULL), state, state_size);
    return state;
}
// End block
/******************************************************************************/

/******************************************************************************/
// This block of code was taken from randomwalk_dirichlet.c 
// in chapter 08 folder. 
// These functions were taken/modified from free software under copyright of 
// Stewart Weiss (found in chapter08/ folder).

/* A 2D Point */
typedef struct {
    int x;
    int y;
} point2d;

/* The four possible directions to go */
const point2d East  = {1, 0};
const point2d West  = {-1,0};
const point2d North = {0, 1};
const point2d South = {0,-1};

/* Randomly generate a new direction to walk */
point2d next_dir()
{
    double u = uniform_random(); // From "utilities_basic.c" - RNG
    if (u < 0.25)
        return North;
    else if (u < 0.50)
        return East;
    else if (u < 0.75)
        return South;
    else
        return West;
} 

/* Generate next point from current point and direction */
point2d next_point(point2d oldpoint, point2d direction)
{
    point2d temp;
    temp.x = oldpoint.x + direction.x;
    temp.y = oldpoint.y + direction.y;
    return temp;
}

/* Test if given point is on a boundary */
int on_boundary(point2d point, int width, int height) 
{
    if ( 0 == point.x )
        return WEST;
    else if ( width -1 == point.x )
        return EAST;
    else if ( 0 == point.y )
        return NORTH;
    else if ( height - 1 == point.y )
        return SOUTH;
    else
        return 0;
}
// End block
/******************************************************************************/

/* A point on the plate */
typedef struct {
    point2d location; // Its location
    double temperature; // Its temperature
} PlatePoint;

/**
 * @brief print_error(errvalue) will print out an error statement
 * based on the errvalue given.
 */
void print_error( 
        int errvalue); /* Is the value that determines the err statement */



/*
    This main function was based off of randomwalk_dirichlet.c 
    in chapter 08 folder, which is a sequential program. 
    This function was taken/modified from free software under copyright of 
    Stewart Weiss (found in /chapter08 folder).
*/
int main(int argc, char * argv[])
{
    point2d current, next;
    int     i, j, k, count = 0, error = 0, error_values_or = 0;
    int     width = 0, height = 0; /* Rows/cols in plate plane */
    double  oldvalue, diff;
    double  maxdiff;
    double  tolerance = CONVERGENCE_THRESHOLD;
    char    *randtable;
    int     tablesize = 20000;
    FILE    *inputfile;
    int     location;
    double  boundary_temperature[4];
    int own_length; /* Length of array of points owned */
    PlatePoint *local_points; /* Store info about points process owns */
    PlatePoint *middle_point; /* Save location of midpt if owned */
    int owns_middle_point = 0; /* Check if this process owns the midpt */
    int assigned_points_per_row = 0; /* How many points given per row */
    int progress = 0; /* Track how far into the local array */
    int offset = 0; /* Track where to start claiming points in a row */
    int below_tolerance = 0; /* Flag to see if this process is below */
    int global_tolerance = 0; /* Store the result of Reduction */

    int id; /* Process rank */
    int p; /* Total num of processes */
    
    MPI_Init(&argc, &argv);

    // Debug code here

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc < 2) { // All p's have argc
        if (id == ROOT) {
            error = USAGE_ERROR;
            print_error(error);
        }
        MPI_Finalize(); // Finalize for all instead of Abort
        return 0; // Close program
    }
    if (id == ROOT) { // Scan for all the necessary values
        inputfile = fopen(argv[1], "r");
        if (NULL == inputfile) {
            error = OPEN_FILE_ERROR;
            print_error(error);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
        fscanf(inputfile, "%d ", &height);
        if (0 >= height) {
            error = BAD_MATRIX_SIZE; 
            print_error(error);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
        fscanf(inputfile, "%d ", &width);
        if (0 == width) {
            error = BAD_MATRIX_SIZE; 
            print_error(error);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
        for( i = 0; i < 4; i++ ) {
            fscanf(inputfile, " %lf ", &(boundary_temperature[i]));
        }
        fclose(inputfile);
    }

    /* Process 0 broadcasts the numbers of rows to all other processes */
    MPI_Bcast(&height, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    /* Process 0 broadcasts the numbers of columns to all other processes */
    MPI_Bcast(&width, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    /* Process 0 broadcasts the boundary temps to all other processes */
    MPI_Bcast(boundary_temperature, 4, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    //printf("Inputs: %d\t%d\t%f\t%f\t%f\t%f\t%d\n", height, width, // Testing
    //       boundary_temperature[0], boundary_temperature[1], 
    //       boundary_temperature[2], boundary_temperature[3], id);

    /* Compute size of array needed to store the points this process owns */
    if (((height - 2) * (width - 2)) % p > id) { // Gets +1 points
        own_length = ((height - 2) * (width - 2) / p) + 1;
    }
    else {
        own_length = ((height - 2) * (width - 2) / p);
    }
    //printf("Length: %d\t%d\n", own_length, id); // Testing
    // Store location and temperature of points owned
    if (own_length > 0) {
        local_points = malloc(own_length * sizeof(PlatePoint));
    }

    // Check that all processes could allocate data
    if (NULL == local_points && own_length > 0) {
        /* malloc failed, so set error code */
        error = MALLOC_ERROR; // Error sent back to ROOT
    }
    /* Check for errors in all processes after allocating, use MPI_Reduce */
    /* Performing the logical-or reduction of these error values will ensure
    that if any single process had trouble allocating data, ROOT will know 
    through this Reduce, as MALLOC_ERROR == 1, so if error_values_or 
    is 1, then some process had an issue allocating */
    MPI_Reduce(&error, &error_values_or, 1, MPI_INT, 
                MPI_LOR, ROOT, MPI_COMM_WORLD);
    if (id == ROOT && error_values_or == MALLOC_ERROR) {
        error = error_values_or; // Send back malloc error
        print_error(error);
        MPI_Abort(MPI_COMM_WORLD, error);
    }

    // Fill array with point data
    if ((width - 2) % p != 0) { // p does not divide evenly to the row
        //printf("Odd\n"); // Testing
        offset = id; // Start offset with just the id 
        if (own_length > 0) { // if (p < width - 2) {
            for (j = 1; j < height - 1; j++) { // Cyclically obtain points owned
                // The if statement below was added for the case of p > w - 2
                // If initially going into next row
                if (1 + offset >= width - 1) {
                        // Then carry over 'remainder' to new row
                        offset = (1 + offset) - (width - 1);
                        // Skip the below for loop and go to next row
                        continue;
                }
                for (i = 1 + offset; i < width - 1; i += p) {
                    local_points[progress].location.x = i; // Set location
                    local_points[progress].location.y = j;
                    local_points[progress].temperature = 0.0; // Set temp
                    if (i == ((width - 1) / 2) && j == ((height - 1) / 2)) {
                        // Save location of midpt
                        middle_point = &local_points[progress]; 
                        // Flag to say that this process owns the midpt
                        owns_middle_point = 1; 
                    }
                    progress++; // Go to next element
                    if (i + p >= width - 1) { // If next going into next row
                        // Then carry over 'remainder' to new row
                        offset = (i + p) - (width - 1); 
                    }
                }
            }
        }
    }
    else { // p is a factor of width - 2
           // each process gets same number of points per row
        //printf("Even\n"); // Testing
        assigned_points_per_row = (width - 2) / p;
        for (j = 1; j < height - 1; j++) {
            for (k = 0; k < assigned_points_per_row; k++) { // For each section
                // id % p is position in each section
                // Use j as a new offset for each row, multiply by some const
                // Mult. j - 1 by p - 1 to properly switch position in each row
                // k * p sets up which section of the row we're in
                i = k * p + ((id + (p - 1)*(j - 1)) % p);
                i++; // Starts with 1, not 0
                local_points[progress].location.x = i;
                local_points[progress].location.y = j; // Set location
                local_points[progress].temperature = 0.0; // Set temp
                if (i == ((width - 1) / 2) && j == ((height - 1) / 2)) {
                    // Save location of midpt
                    middle_point = &local_points[progress]; 
                    // Flag to say that this process owns the midpt
                    owns_middle_point = 1; 
                }
                progress++; // Go to next element
            }
        }
    }
    //for(i = 0; i < own_length; i++) { // Testing
    //    printf("Initial: %d\t%d\t%f\t%d\n", 
    //           local_points[i].location.x, local_points[i].location.y, 
    //           local_points[i].temperature, id);
    //}

    // Seed RNG
    randtable = init_random(tablesize, id);
    // Check that all processes could allocate data
    if (NULL == randtable) {
        /* malloc failed, so set error code and quit */
        error = MALLOC_ERROR; // Error sent back to ROOT
    }
    /* Check for errors in all processes after allocating, use MPI_Reduce */
    MPI_Reduce(&error, &error_values_or, 1, MPI_INT, 
                MPI_LOR, ROOT, MPI_COMM_WORLD);
    if (id == ROOT && error_values_or == MALLOC_ERROR) {
        error = MALLOC_TABLE_ERROR; // Send back malloc error
        print_error(error);
        MPI_Abort(MPI_COMM_WORLD, error);
    }

    while (count < 40000) { // Max of 40,000 loops
        maxdiff = 0;
        for (i = 0; i < own_length; i++) { // For each point owned
                current.x = local_points[i].location.x; // Get location
                current.y = local_points[i].location.y;
                while (0 == (location = on_boundary(current, width, height))) {
                    // Get next point while boundary not hit
                    next = next_point(current, next_dir());
                    //printf("%d\t%d\n", next.x, next.y); // Testing
                    current = next;
                }
                oldvalue = local_points[i].temperature; // Save old temp
                // Compute new temperature
                local_points[i].temperature = (oldvalue * count + 
                                boundary_temperature[location-1]) / (count + 1);
                diff = fabs(local_points[i].temperature - oldvalue);
                if (diff > maxdiff)
                    maxdiff = diff;
                /* maxdiff is largest difference in current iteration */
        }
        // Check all processes must have maxdiff < tolerance to be steady state
        if (maxdiff < tolerance) {
            below_tolerance = 1;
        }
        else { // Not below tolerance
            below_tolerance = 0;
        }
        /* Just as with malloc error handling, communicate tolerance */
        MPI_Reduce(&below_tolerance, &global_tolerance, 1, MPI_INT, 
                MPI_LAND, ROOT, MPI_COMM_WORLD);
        /* Process 0 broadcasts the global tolerance to all other processes */
        MPI_Bcast(&global_tolerance, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

        if (global_tolerance) { // If all processes are below tolerance
            break; // Each process breaks the outer while loop
        }
        else { // Otherwise continue while loop
            count++;
        }
    }
    //for (i = 0; i < own_length; i++) { // Testing
    //    printf("Post: %d\t%d\t%f\t%d\t%d\n", 
    //           local_points[i].location.x, local_points[i].location.y, 
    //           local_points[i].temperature, count, id);
    //}

    // Obtain the middle point temperature in steady state
    if (ROOT == id && !owns_middle_point) { 
        // If root doesn't own midpt
        // Initialize midptr to store temperature
        middle_point = malloc(sizeof(PlatePoint));
        // Recv midpt temp from owner
        MPI_Recv(&(middle_point->temperature), 1, MPI_DOUBLE,
                    MPI_ANY_SOURCE, RESPONSE_MSG, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // If it does own midpt, the temperature will
        // already be stored in the middle_point ptr
    }
    else if (ROOT != id && owns_middle_point) {
        // If this process owns the midpt, send to ROOT
        MPI_Send(&(middle_point->temperature), 1, MPI_DOUBLE, 
                    ROOT, RESPONSE_MSG, MPI_COMM_WORLD);
    }

    // Output the middle point temperature
    if (ROOT == id) {
        printf("%0.2f\n", middle_point->temperature);
        //printf("Midpt Location: %d\t%d\n", // Testing
        //       (width - 1) / 2, (height - 1) / 2);
        if (!owns_middle_point) {
            free(middle_point);
        }
    }

    free(local_points);
    free(randtable);
    MPI_Finalize();
    return 0;
}



void print_error(int errvalue) {
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        errvalue = MPI_INIT_ERROR;
    }
    //printf("%d\n", errvalue); // Testing
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
            fprintf(stderr, "Error: Cannot construct plate "
                     "based on dimensions given.\n");
            break;
        case MALLOC_TABLE_ERROR :
            fprintf(stderr, "Error: Could not allocate data for "
                     "random number generation.\n");
            break;
        case USAGE_ERROR :
            fprintf(stderr, "Error: Usage: steady_state <txtfile>, "
                     "where txtfile contains plate dimensions, followed "
                     "by temperature at each border.\n");
            break;
        default: 
            break;
    }
}