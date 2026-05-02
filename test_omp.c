#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) printf("Threads: %d\n", omp_get_num_threads());
    }
    return 0;
}
