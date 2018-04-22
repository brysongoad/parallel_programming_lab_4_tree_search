/* Bryson Goad
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
 * 10  11  12  0
 *
 * Program Arguments: <thread_count> <digraph_file>
 * Note: won't work if number of cities < number of threads
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

const int NO_CITY = -1;

typedef struct Tour Tour;
struct Tour {
    int* cities;    // array of cities in the tour
    int count;      // number of cities in the tour
    int cost;       // total travel cost of the tour
    Tour *next;     // next tour in list if this tour is in a list
};

int num_cities;     // number of cities in search
int* digraph;       // 2d array of travel costs between cities
Tour* best_tour;    // lowest cost tour that has been found

Tour* pop(Tour** stack);

void initTour(Tour* tour);

void push(Tour** stack, Tour* tour);

void pushCopy(Tour** stack, Tour* tour, Tour** available_stack);

int travelCost(int city1, int city2);

bool isNewBest(Tour *tour);

void addCity(Tour* tour, int new_city);

void updateBest(Tour* newBest);

bool feasible(Tour* tour, int newCity);

bool visited(Tour* tour, int city);

void copyTour(Tour* from, Tour* to);

Tour* allocTour (Tour** available_stack);

void removeLastCity(Tour* tour);

void readDigraph(FILE *digraph_file);

int main(int argc, char* argv[]) {

    FILE* digraph_file;             // file containing digraph
    double time_start, time_end;    // start and end times of parallel section
    int thread_count;               // number of threads to use
    Tour* initial_partition_stack;  // stack for initial partition of tree
    int initial_partition_size;     // number of branches in initial partition

    // get command line args
    thread_count = strtol(argv[1], NULL, 10);
    digraph_file = fopen(argv[2], "r");
    if (digraph_file == NULL) {
        fprintf(stderr, "Can't open %s\n", argv[1]);
    }

    // read digraph file
    readDigraph(digraph_file);
    fclose(digraph_file);

    //initialize best tour
    best_tour = allocTour(NULL);
    initTour(best_tour);
    best_tour->cost = INT_MAX;

    // start timer
    time_start = omp_get_wtime();

    // initialize partition stack
    initial_partition_stack = NULL;

#   pragma omp parallel num_threads(thread_count)
    {
        int thread_num = omp_get_thread_num();  // current thread's number
        Tour *stack = NULL;                     // current thread's stack
        Tour *available_stack = NULL;           // current thread's stack of available tour struct allocations
        Tour *curr_tour;                        // current thread's current tour

        // do initial partition of tree on a single thread so branches can be assigned to threads
#       pragma omp master
        {
            // current size of initial partition stack
            int current_size = 0;

            // create initial tour with just hometown
            Tour *initial_tour = allocTour(NULL);
            initTour(initial_tour);
            // push initial tour on to initial stack
            push(&initial_partition_stack, initial_tour);
            current_size++;

            // expand tree until there are at least as many branches as threads
            while (current_size < thread_count) {
                curr_tour = pop(&initial_partition_stack);
                current_size--;
                // expand to each possible city from current tour
                for (int city = 1; city < num_cities; city++)
                    if (!visited(curr_tour, city)) {
                        // push copy of tour with new city added
                        addCity(curr_tour, city);
                        pushCopy(&initial_partition_stack, curr_tour, &available_stack);
                        removeLastCity(curr_tour);

                        current_size++;
                    }
                // done expanding current tour, make its already allocated memory available for next tour
                push(&available_stack, curr_tour);
            }
            initial_partition_size = current_size;
        }
#       pragma omp barrier

        // determine how many branches to assign to which threads
        int first_tour, last_tour;
        int quotient, remainder, tour_count;
        quotient = initial_partition_size/thread_count;
        remainder = initial_partition_size % thread_count;
        if (thread_num < remainder) {
            tour_count = quotient+1;
            first_tour = thread_num*tour_count;
        } else {
            tour_count = quotient;
            first_tour = thread_num*tour_count + remainder;
        }
        last_tour = first_tour + tour_count -1;

        // push assigned number of branches onto local thread stack
#       pragma omp critical
        for (last_tour; last_tour >= first_tour; last_tour--) {
                Tour *temp = pop(&initial_partition_stack);
                push(&stack, temp);
        }

        // do a tree search
        while (stack != NULL) {
            curr_tour = pop(&stack);
            // check if current tour has visited all cities
            if (curr_tour->count == num_cities) {
                // if current tour has new lowest cost, update the best_tour
                if (isNewBest(curr_tour)) {
#                   pragma omp critical
                    updateBest(curr_tour);
                }
            } else {
                // expand to each possible city from current tour
                for (int city = num_cities - 1; city >= 1; city--) {
                    if (feasible(curr_tour, city)) {
                        // push copy of tour with new city added
                        addCity(curr_tour, city);
                        pushCopy(&stack, curr_tour, &available_stack);
                        removeLastCity(curr_tour);
                    }
                }
            }
            // done expanding current tour,
            // make its already allocated memory available for next tour to decrease memory allocation operations
            push(&available_stack, curr_tour);
        }

        // free memory in available stack
        curr_tour = pop(&available_stack);
        while (curr_tour != NULL) {
            free(curr_tour->cities);
            free(curr_tour);
            curr_tour = pop(&available_stack);
        }
    }

    // stop timer
    time_end = omp_get_wtime();

    // display best tour found
    printf("Best Tour: \n");
    for (int i = 0; i < num_cities; i++) {
        printf("%d -> ", best_tour->cities[i]);
    }
    printf("%d", best_tour->cities[num_cities]);
    printf("\nCost: %d\n", best_tour->cost);

    printf("Elapsed time: %f\n", time_end-time_start);

    // free memory for best tour and digraph
    free(best_tour->cities);
    free(best_tour);
    free(digraph);

    return 0;
}

// returns the top element of a stack
Tour* pop(Tour** stack) {
    if (*stack == NULL) return NULL;
    Tour* temp = *stack;
    *stack = (*stack)->next;
    temp->next = NULL;
    return temp;
}

// initializes values for a tour
void initTour(Tour* tour) {
    tour->cities[0] = 0;
    for (int i = 1; i < num_cities; i++) {
        tour->cities[i] = NO_CITY;
    }
    tour->cost = 0;
    tour->count = 1;
    tour->next = NULL;
}

// pushes a tour onto a stack
void push(Tour** stack, Tour* tour) {
    if (*stack != NULL)
        tour->next = *stack;
    else
        tour->next = NULL;
    *stack = tour;
}

// pushes a copy of a tour onto a stack
void pushCopy(Tour** stack, Tour* tour, Tour** available_stack) {

    Tour* temp = allocTour(available_stack);
    copyTour(tour, temp);

    push(stack, temp);
}

// returns travel cost from city1 to city2
int travelCost(int city1, int city2) {
    return digraph[city1*num_cities + city2];
}

// returns true if tour has lower cost than previous best
bool isNewBest(Tour* tour){
    int cost = tour->cost;
    cost += travelCost(tour->cities[tour->count-1], 0);

    return cost < best_tour->cost;
}

// adds a city to a tour
void addCity(Tour* tour, int new_city) {
    tour->cities[tour->count] = new_city;
    tour->cost += travelCost(tour->cities[tour->count - 1], new_city);
    tour->count++;
}

// updates the best_tour
void updateBest(Tour* newBest) {
    if (isNewBest(newBest)) {
        copyTour(newBest, best_tour);
        addCity(best_tour, 0);
    }
}

// returns true if the city has not been visited yet in the tour
// and it would not make the tour's total cost more than the best tour's cost
bool feasible(Tour* tour, int newCity) {
    if (tour->cost + travelCost(tour->cities[tour->count - 1], newCity) > best_tour->cost)
        return false;
    if (visited(tour, newCity))
        return false;
    return true;
}

// returns true if city has already been visited on tour
bool visited(Tour* tour, int city) {
    for (int i = 1; i < tour->count; i++) {
        if (tour->cities[i] == city)
            return true;
    }
    return false;
}

// copies a tour from one mem location to the other
void copyTour(Tour* from, Tour* to) {
    memcpy(to->cities, from->cities, (num_cities)*sizeof(int));
    to->count = from->count;
    to->cost = from->cost;
}

// returns pointer to memory allocated for a tour
Tour* allocTour (Tour** available_stack) {
    Tour* temp;

    if (available_stack == NULL || *available_stack == NULL) {
        temp = malloc(sizeof(Tour));
        temp->cities = malloc(num_cities * sizeof(int));
        return temp;
    }
    else
        return pop(available_stack);
}

// removes the last city from a tour
void removeLastCity(Tour* tour) {
    tour->cost -= travelCost(tour->cities[tour->count - 2], tour->cities[tour->count - 1]);
    tour->cities[tour->count - 1] = NO_CITY;
    tour->count--;
}

// reads digraph file
void readDigraph(FILE *digraph_file) {
    fscanf(digraph_file, "%d", &num_cities);
    if (num_cities <= 0) {
        fprintf(stderr, "Number of vertices in digraph must be positive\n");
        exit(-1);
    }
    digraph = malloc(num_cities*num_cities*sizeof(int));

    for (int i = 0; i < num_cities; i++)
        for (int j = 0; j < num_cities; j++) {
            fscanf(digraph_file, "%d", &digraph[i*num_cities + j]);
            if (i == j && digraph[i*num_cities + j] != 0) {
                fprintf(stderr, "Diagonal entries must be zero\n");
                exit(-1);
            } else if (i != j && digraph[i*num_cities + j] <= 0) {
                fprintf(stderr, "Off-diagonal entries must be positive\n");
                fprintf(stderr, "digraph[%d,%d] = %d\n", i, j, digraph[i*num_cities+j]);
                exit(-1);
            }
        }
}
