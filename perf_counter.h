#ifndef PERF_COUNTER_H
#define PERF_COUNTER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <perfmon/pfmlib_perf_event.h>
#include <time.h>

typedef struct {
    uint64_t value;
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t id;
} read_format_t;

typedef struct {
    struct perf_event_attr pe;
    int fd;
    read_format_t prev;
    read_format_t data;
} event_t;

typedef enum {
    USER = 0b1,
    KERNEL = 0b10,
    HYPERVISOR = 0b100,
    ALL = 0b111
} EventDomain;

typedef struct {
    event_t *events;
    char **names;
    size_t event_count;
    struct timespec startTime;
    struct timespec stopTime;
} PerfCounter;

uint64_t readCounter(event_t *event) {
    uint64_t count = 0, values[3];
    int ret = read(event->fd, values, sizeof(values));
    if (ret != sizeof(values)) {
        fprintf(stderr, "cannot read results: %s\n", strerror(errno));
    }
    if (values[2]) {
        count = (uint64_t)((double)values[0] * values[1] / values[2]);
    }
    return count;
}

PerfCounter* PerfCounter_init() {
    if (pfm_initialize() != PFM_SUCCESS) {
        fprintf(stderr, "libpfm initialization failed\n");
        return NULL;
    }

    PerfCounter* pc = (PerfCounter*)malloc(sizeof(PerfCounter));
    if (!pc) {
        return NULL;
    }

    pc->events = NULL;
    pc->names = NULL;
    pc->event_count = 0;

    // Registering Counters
    // // -------- OK ------------
    PerfCounter_registerCounter(pc, "ANY_DATA_CACHE_FILLS_FROM_SYSTEM:INT_CACHE", ALL);
    PerfCounter_registerCounter(pc, "ANY_DATA_CACHE_FILLS_FROM_SYSTEM:EXT_CACHE_LCL", ALL);
    PerfCounter_registerCounter(pc, "ANY_DATA_CACHE_FILLS_FROM_SYSTEM:EXT_CACHE_RMT", ALL);
    PerfCounter_registerCounter(pc, "ANY_DATA_CACHE_FILLS_FROM_SYSTEM:MEM_IO_LCL", ALL);
    PerfCounter_registerCounter(pc, "ANY_DATA_CACHE_FILLS_FROM_SYSTEM:MEM_IO_RMT", ALL);
    // // -------------

    PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_MISSES", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_REFERENCES", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_LL:READ", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_LL:WRITE", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_LL:PREFETCH", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_LL:ACCESS", ALL);
    // PerfCounter_registerCounter(pc, "LLC-LOADS", ALL);


    // PerfCounter_registerCounter(pc, "HARDWARE_PREFETCH_DATA_CACHE_FILLS:INT_CACHE", ALL);
    // PerfCounter_registerCounter(pc, "HARDWARE_PREFETCH_DATA_CACHE_FILLS:LCL_L2", ALL);
    // PerfCounter_registerCounter(pc, "HARDWARE_PREFETCH_DATA_CACHE_FILLS:EXT_CACHE_LCL", ALL);
    // PerfCounter_registerCounter(pc, "HARDWARE_PREFETCH_DATA_CACHE_FILLS:EXT_CACHE_RMT", ALL);

    // PerfCounter_registerCounter(pc, "RETIRED_BRANCH_INSTRUCTIONS_MISPREDICTED", ALL);

    // // PerfCounter_registerCounter(pc, "SOFTWARE_PREFETCH_DATA_CACHE_FILLS:INT_CACHE", ALL);
    // PerfCounter_registerCounter(pc, "SOFTWARE_PREFETCH_DATA_CACHE_FILLS:LCL_L2", ALL);
    // // PerfCounter_registerCounter(pc, "SOFTWARE_PREFETCH_DATA_CACHE_FILLS:EXT_CACHE_LCL", ALL);
    // // PerfCounter_registerCounter(pc, "SOFTWARE_PREFETCH_DATA_CACHE_FILLS:EXT_CACHE_RMT", ALL);

    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_DTLB:READ", ALL);
    // // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_DTLB:WRITE", ALL);
    // // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_DTLB:PREFETCH", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_DTLB:ACCESS", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_DTLB:MISS", ALL);

    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_ITLB:READ", ALL);
    // // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_ITLB:WRITE", ALL);
    // // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_ITLB:PREFETCH", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_ITLB:ACCESS", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_ITLB:MISS", ALL);



    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_BRANCH_INSTRUCTIONS", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_BRANCH_MISSES", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_STALLED_CYCLES_BACKEND", ALL);
    // // PerfCounter_registerCounter(pc, "PERF_COUNT_SW_PAGE_FAULTS", ALL);
    // PerfCounter_registerCounter(pc, "L1-DCACHE-LOAD-MISSES", ALL);
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_L1I:MISS", ALL);
    // PerfCounter_registerCounter(pc, "L2_PREFETCH_HIT_L2:L2_HW_PREFETCHER", ALL);
    // PerfCounter_registerCounter(pc, "L2_PREFETCH_HIT_L2:L1_HW_PREFETCHER", ALL);
    // PerfCounter_registerCounter(pc, "L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER", ALL);
    // PerfCounter_registerCounter(pc, "L2_PREFETCH_HIT_L3:L1_HW_PREFETCHER", ALL);
    // PerfCounter_registerCounter(pc, "L2_PREFETCH_MISS_L3:L2_HW_PREFETCHER", ALL);
    // PerfCounter_registerCounter(pc, "L2_PREFETCH_MISS_L3:L1_HW_PREFETCHER", ALL);
    // PerfCounter_registerCounter(pc, "LLC-STORE-MISSES", ALL);
    // PerfCounter_registerCounter(pc, "LLC-PREFETCHES", ALL);
    // PerfCounter_registerCounter(pc, "LLC-PREFETCH-MISSES", ALL);
    // PerfCounter_registerCounter(pc, "DTLB-LOADS", ALL);
    // PerfCounter_registerCounter(pc, "DTLB-STORES", ALL);
    // PerfCounter_registerCounter(pc, "BRANCH-LOADS", ALL);

    // PerfCounter_registerCounter(pc, "BRANCH-LOAD-MISSES", ALL);
    // PerfCounter_registerCounter(pc, "DEMAND_DATA_CACHE_FILLS_FROM_SYSTEM:INT_CACHE", ALL);
    // PerfCounter_registerCounter(pc, "DEMAND_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2", ALL);
    // PerfCounter_registerCounter(pc, "DEMAND_DATA_CACHE_FILLS_FROM_SYSTEM:EXT_CACHE_LCL", ALL);
    // PerfCounter_registerCounter(pc, "DEMAND_DATA_CACHE_FILLS_FROM_SYSTEM:EXT_CACHE_RMT", ALL);
    // PerfCounter_registerCounter(pc, "MISALIGNED_LOADS", ALL);
    
    // PerfCounter_registerCounter(pc, "SOFTWARE_PREFETCH_DATA_CACHE_FILLS:LCL_L2", ALL);
    // // PerfCounter_registerCounter(pc, "SOFTWARE_PREFETCH_DATA_CACHE_FILLS:INT_CACHE", ALL);
    // // PerfCounter_registerCounter(pc, "SOFTWARE_PREFETCH_DATA_CACHE_FILLS:EXT_CACHE_LCL", ALL);
    // // PerfCounter_registerCounter(pc, "SOFTWARE_PREFETCH_DATA_CACHE_FILLS:EXT_CACHE_RMT", ALL);

    PerfCounter_registerCounter(pc, "INSTRUCTION_CACHE_REFILLS_FROM_L2", ALL);
    PerfCounter_registerCounter(pc, "INSTRUCTION_CACHE_REFILLS_FROM_SYSTEM", ALL);
    // PerfCounter_registerCounter(pc, "L2_PREFETCH_HIT_L3:L2_HW_PREFETCHER", ALL);
    // PerfCounter_registerCounter(pc, "L2_PREFETCH_HIT_L3:L1_HW_PREFETCHER", ALL);
    
    // PerfCounter_registerCounter(pc, "PERF_COUNT_HW_CACHE_DTLB:PREFETCH", ALL);


    for (size_t i = 0; i < pc->event_count; i++) {
        event_t* event = &pc->events[i];
        event->fd = syscall(__NR_perf_event_open, &event->pe, 0, -1, -1, 0);
        if (event->fd < 0) {
            fprintf(stderr, "Error opening counter %s\n", pc->names[i]);
            free(pc->events);
            free(pc->names);
            free(pc);
            return NULL;
        }
    }

    return pc;
}

void PerfCounter_registerCounter(PerfCounter* pc, const char* name, EventDomain domain) {
    pc->event_count++;
    pc->events = (event_t*)realloc(pc->events, pc->event_count * sizeof(event_t));
    pc->names = (char**)realloc(pc->names, pc->event_count * sizeof(char*));
    pc->names[pc->event_count - 1] = strdup(name);

    event_t* event = &pc->events[pc->event_count - 1];
    memset(&event->pe, 0, sizeof(struct perf_event_attr));

    int ret = pfm_get_perf_event_encoding(name, PFM_PLM0 | PFM_PLM3, &event->pe, NULL, NULL);
    if (ret != PFM_SUCCESS) {
        fprintf(stderr, "Cannot find encoding: %s\n", pfm_strerror(ret));
        free(pc->events);
        free(pc->names);
        free(pc);
        return;
    }

    event->pe.disabled = 1;
    event->pe.inherit = 1;
    event->pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    event->fd = syscall(__NR_perf_event_open, &event->pe, 0, -1, -1, 0);

    if (event->fd < 0) {
        fprintf(stderr, "Error opening counter %s\n", name);
        free(pc->events);
        free(pc->names);
        free(pc);
        return;
    }
}

void PerfCounter_startCounters(PerfCounter* pc) {
    for (size_t i = 0; i < pc->event_count; i++) {
        event_t* event = &pc->events[i];
        ioctl(event->fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(event->fd, PERF_EVENT_IOC_ENABLE, 0);
        if (read(event->fd, &event->prev, sizeof(uint64_t) * 3) != sizeof(uint64_t) * 3) {
            fprintf(stderr, "Error reading counter %s\n", pc->names[i]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &pc->startTime);
}

void PerfCounter_resetCounter(PerfCounter* pc, const char* name) {
    for (size_t i = 0; i < pc->event_count; i++) {
        if (strcmp(pc->names[i], name) == 0) {
            event_t* event = &pc->events[i];
            ioctl(event->fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(event->fd, PERF_EVENT_IOC_ENABLE, 0);
            if (read(event->fd, &event->prev, sizeof(uint64_t) * 3) != sizeof(uint64_t) * 3) {
                fprintf(stderr, "Error reading counter %s\n", pc->names[i]);
            }
            break;
        }
    }
}

void PerfCounter_stopCounters(PerfCounter* pc) {
    clock_gettime(CLOCK_MONOTONIC, &pc->stopTime);
    for (size_t i = 0; i < pc->event_count; i++) {
        event_t* event = &pc->events[i];
        if (read(event->fd, &event->data, sizeof(uint64_t) * 3) != sizeof(uint64_t) * 3) {
            fprintf(stderr, "Error reading counter %s\n", pc->names[i]);
        }
        ioctl(event->fd, PERF_EVENT_IOC_DISABLE, 0);
    }
}

uint64_t PerfCounter_getCounter(PerfCounter* pc, const char* name) {
    for (size_t i = 0; i < pc->event_count; i++) {
        if (strcmp(pc->names[i], name) == 0) {
            return readCounter(&pc->events[i]);
        }
    }
    return (uint64_t)-1;
}

void PerfCounter_printReport(PerfCounter* pc, FILE* out, uint64_t normalizationConstant) {
    if (pc->event_count == 0) return;

    for (size_t i = 0; i < pc->event_count; i++) {
        fprintf(out, "%s: %lf\n", pc->names[i], (double)readCounter(&pc->events[i]) / normalizationConstant);
    }

    fprintf(out, "scale: %lu\n", normalizationConstant);
}

void PerfCounter_cleanup(PerfCounter* pc) {
    for (size_t i = 0; i < pc->event_count; i++) {
        close(pc->events[i].fd);
        free(pc->names[i]);
    }
    pfm_terminate();
    free(pc->events);
    free(pc->names);
    free(pc);
}

#endif // PERF_COUNTER_H
