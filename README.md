# parallel_programming_lab_4_tree_search
Lab 4 for OC parallel programming class

 * Parallel Programming lab 4 - Tree Search
 * Implementation of a parallel tree search to solve a traveling salesperson problem using OpenMP
 * reads a matrix of travel costs from a file to construct graph
 * first line of the file is number of cities
 * matrix elements separated by whitespace, rows separated by line
 * element [i][j] is cost to travel from city i to city j
 * city 0 is always starting point
 * example:
 * 4
 * 0   1   2   3
 * 4   0   5   6
 * 7   8   0   9
 * 1   2   3   0
 * Program Arguments: <thread_count> <digraph_file>
 * Note: won't work if number of cities < number of threads
