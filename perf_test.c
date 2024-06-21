#include <stdio.h>
#include "perf_counter.h"

int main() {
    PerfCounter *pc = PerfCounter_init();
    if (pc == NULL) {
        fprintf(stderr, "Failed to initialize PerfCounter\n");
        return 1;
    }

    printf("Starting counters...\n");
    PerfCounter_startCounters(pc);

    // Simulate some workload
    for (volatile int i = 0; i < 100000000; i++);

    PerfCounter_stopCounters(pc);
    printf("Stopped counters.\n");

    printf("Performance counters report:\n");
    PerfCounter_printReport(pc, stdout, 1);

    PerfCounter_cleanup(pc);
    return 0;
}
