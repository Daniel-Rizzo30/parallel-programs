/*******************************************************************************
Title : hwk2_Daniel.Rizzo30.c
Author : Daniel Rizzo
Class : CSCI 49365 Parallel Computing, Stewart Weiss
Created on : September 27, 2021
Description : Prints out the path that simulates the sequence of messages that 
would be passed in a reduction operation that uses the binomial tree model of 
communication. 
Purpose : Demonstrates how a binomial tree helps with a reduction operation 
efficiently.
Usage : reduce <n>, where <n> is a positive amount of nodes, eg. reduce 4
Build with : gcc -o reduce hwk2_Daniel.Rizzo30.c
Modifications: September 28, 2021: Made program fit more accordingly to the 
programming rules
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

/** @brief binomial_tree_route(const int number_of_nodes) prints out to the 
 * console the order in which nodes of a binomial tree will communicate in a 
 * reduction operation, it works for both powers of two and non-powers of two
 * @param  number_of_nodes is the amount of nodes in the binomial tree
 * @pre    number_of_nodes is a positive integer
 * @post   the order of communication in the tree is printed
 */
void binomial_tree_route(const int number_of_nodes);

int main(int argc, char* argv[]) {
    if (argc != 2) { // check if usage is correct
        printf("Usage: %s <n>\n", argv[0]);
        return 0; // exit
    }
    int n = atoi(argv[1]); // convert to int
    if (n <= 0) { // check if |nodes| is positive
        printf("Error: Input <n> must be a positive integer. ");
        printf("Received %d.\n", n);
        return 0;
    }
    binomial_tree_route(n); // run
    return 0;
}

void binomial_tree_route(const int number_of_nodes) {
    // bit length to store for later for loop
    // power of two flag to determine if an extra step is needed
    int bit_length = 0, power_of_two = 0, closest_power_of_two = 1; 
    double number_of_nodes_copy =  (double) number_of_nodes; // copy value
    while (1) {
        // divide if it will not turn into a fraction
        if (number_of_nodes_copy / 2 >= 1) { 
            // add 1 to bit length so number of loops needed can be tracked
            bit_length++; 
            number_of_nodes_copy = number_of_nodes_copy / 2;
            // Calculate closest power of two (<= n) while in this loop
            closest_power_of_two = closest_power_of_two * 2;
        }
        // for when the copy is between 2 and 1 - [1,2)
        else {
            // if number of nodes is an exact power of 2
            if (number_of_nodes_copy == 1) { 
                power_of_two = 1; // set flag
            }
            break; // break while loop
        }
    }

    // if not a power of two, then an extra step is needed
    // print out the extra pass that is needed
    if (power_of_two == 0) {
        for (int i = number_of_nodes - 1; i >= closest_power_of_two; i--) {
            printf("task %d sends a value to task %d\n", 
                    i, (i - closest_power_of_two));
        }
    }

    for (int i = bit_length; i > 0; i--) { // for each bit
        // for half the amount that can be represented by the num of bits
        for (int j = closest_power_of_two - 1; 
                j >= closest_power_of_two / 2; j--) {
            // pair each node with another by flipping ith bit
            printf("task %d sends a value to task %d\n", 
                    j, (j - (closest_power_of_two / 2)));
        }
        closest_power_of_two = closest_power_of_two / 2; // decrement
    }
}