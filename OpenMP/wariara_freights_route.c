// =========================================================================
// Practical 3: Minimum Energy Consumption Freight Route Optimization
// =========================================================================
//
// GROUP NUMBER:
//
// MEMBERS:
//   - Muhammed Zaakir Vahed, VHDMUH004
//   - Kamil Singh, Student SNGKAM012

// ========================================================================
//  PART 1: Minimum Energy Consumption Freight Route Optimization using OpenMP
// =========================================================================


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <omp.h>

#define MAX_N 10

// ============================================================================
// Global variables
// ============================================================================

int procs = 1;

int n;
int adj[MAX_N][MAX_N];

// ============================================================================
// Branch and Bound variables
// ============================================================================

int best_cost = 1e9;
int best_path[MAX_N];

// Swap helper
void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

// Recursive Branch & Bound
void permute(int *path, int level, int current_cost) {

    if (current_cost >= best_cost)
        return; // prune

    if (level == n) {
        #pragma omp critical
        {
            if (current_cost < best_cost) {
                best_cost = current_cost;
                for (int i = 0; i < n; i++)
                    best_path[i] = path[i];
            }
        }
        return;
    }

    for (int i = level; i < n; i++) {

        swap(&path[level], &path[i]);

        int new_cost = current_cost;
        if (level > 0)
            new_cost += adj[path[level-1]][path[level]];

        permute(path, level + 1, new_cost);

        swap(&path[level], &path[i]);
    }
}

// ============================================================================
// Timer: returns time in seconds
// ============================================================================

double gettime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// ============================================================================
// Usage function
// ============================================================================

void Usage(char *program) {
  printf("Usage: %s [options]\n", program);
  printf("-p <num>\tNumber of processors/threads to use\n");
  printf("-i <file>\tInput file name\n");
  printf("-o <file>\tOutput file name\n");
  printf("-h \t\tDisplay this help\n");
}


int main(int argc, char **argv)
{
    
    int opt;
    int i, j;
    char *input_file = NULL;
    char *output_file = NULL;
    FILE *infile = NULL;
    FILE *outfile = NULL;
    int success_flag = 1; // 1 = good, 0 = error/help encountered
    
    

    while ((opt = getopt(argc, argv, "p:i:o:h")) != -1)
    {
        switch (opt)
        {
            case 'p':
            {
                procs = atoi(optarg);
                break;
            }

            case 'i':
            {
                input_file = optarg;
                break;
            }

            case 'o':
            {
                output_file = optarg;
                break;
            }

            case 'h':
            {
                Usage(argv[0]);
                success_flag = 0; 
                break;
            }

        default:
            Usage(argv[0]);
            success_flag = 0; 
        }
    }


    if (success_flag) {
        infile = fopen(input_file, "r");
        if (infile == NULL) {
            fprintf(stderr, "Error: Cannot open input file '%s'\n", input_file);
            perror("");
            success_flag = 0;
        } else {
            fscanf(infile, "%d", &n);

            for (i = 1; i < n; i++)
            {
                for (j = 0; j < i; j++)
                {
                    fscanf(infile, "%d", &adj[i][j]);
                    adj[j][i] = adj[i][j];
                }
            }
        }
    }

    if (success_flag) {
        outfile = fopen(output_file, "w");
        if (outfile == NULL) {
            fprintf(stderr, "Error: Cannot open output file '%s'\n", output_file);
            perror("");
            success_flag = 0;
        }
    }

    if (!success_flag) return 1;

    

    printf("Running with %d processes/threads on a graph with %d nodes\n", procs, n);

    
    // TODO: compute solution to minimum energy consumption problem here and write to outfile
    // TIMING 
    double t_init_start = gettime();

    omp_set_num_threads(procs);

    int path[MAX_N];
    for (i = 0; i < n; i++)
        path[i] = i;

    double t_init_end = gettime();

    double t_comp_start = gettime();

    // PARALLEL REGION 
    #pragma omp parallel
{
    #pragma omp single
    {
        for (i = 1; i < n; i++) {

            int local_path[MAX_N];
            for (j = 0; j < n; j++)
                local_path[j] = path[j];

            swap(&local_path[1], &local_path[i]);

            #pragma omp task
            permute(local_path, 2, adj[local_path[0]][local_path[1]]);
        }

        #pragma omp taskwait  
    }
}

    double t_comp_end = gettime();

    printf("Best cost found: %d\n", best_cost);
    // OUTPUT 
    fprintf(outfile, "Best cost: %d\n", best_cost);
    fprintf(outfile, "Best path: ");
    for (i = 0; i < n; i++)
        fprintf(outfile, "%d ", best_path[i] + 1);
    fprintf(outfile, "\n");

    double Tinit = t_init_end - t_init_start;
    double Tcomp = t_comp_end - t_comp_start;
    double Ttotal = Tinit + Tcomp;

    fprintf(outfile, "T_init: %f\n", Tinit);
    fprintf(outfile, "T_comp: %f\n", Tcomp);
    fprintf(outfile, "T_total: %f\n", Ttotal);

    fclose(infile);
    fclose(outfile);
    

    return 0;
}
