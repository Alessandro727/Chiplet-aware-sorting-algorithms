/* Copyright (c) 2013
 * The Trustees of Columbia University in the City of New York
 * All rights reserved.
 *
 * Author:  Orestis Polychroniou  (orestis@cs.columbia.edu)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <smmintrin.h>
#include <sched.h>
#include <numa.h>
#include <numaif.h>
#undef _GNU_SOURCE

#ifdef AVX
#include <immintrin.h>
#else
#include <smmintrin.h>
#endif

#include "rand.h"
#include "util.h"

#include "perf_counter.h"

#include <sys/mman.h>
#include <errno.h>

int global_tuples;

uint64_t micro_time(void)
{
	struct timeval t;
	struct timezone z;
	gettimeofday(&t, &z);
	return t.tv_sec * 1000000 + t.tv_usec;
}

int hardware_threads(void)
{
	char name[40];
	struct stat st;
	int cpus = -1;
	do {
		sprintf(name, "/sys/devices/system/cpu/cpu%d", ++cpus);
	} while (stat(name, &st) == 0);
	return cpus;
}

#define CHIPLETS 16
#define CORES_PER_CHIPLET 8
#define CORES_PER_NUMA_NODE 64

int calculateCore(int core) {
    int base = (core % CHIPLETS) * CORES_PER_CHIPLET;  // This calculates the multiple of 8 part
    int cycle = core / CHIPLETS;       // This determines whether to add 0 or 1
    
    if (cycle % 2 == 0) {
        return base;
    } else {
        return base + 1;
    }
}

int calculatePattern(int n) {
	int blockSize = 128;
	int groupSize = 16;

	// if (global_tuples >= 100) {
	// 	blockSize = 128;
	// 	groupSize = 16;
	// }
    int base = (n / blockSize) * blockSize;  // Calculate the base value for the current block
    int offset = n % blockSize;              // Calculate the offset within the current block

    // Calculate the output based on the offset
    return base + (offset / groupSize) + (offset % groupSize) * (blockSize / groupSize);
}

void cpu_bind(int cpu_id)
{
	//Chiplet Local
	// int newCore = cpu_id;
	// Chiplet Mixed
	int newCore = calculatePattern(cpu_id);
	// printf("%d -> %d\n", cpu_id, newCore);
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(newCore, &cpu_set);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);

	int numa_node = newCore / CORES_PER_NUMA_NODE;
	numa_node = numa_node % 2;
    unsigned long nodemask = 1UL << numa_node;
    if (set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8) == -1) {
        printf("set_mempolicy error");
    }  
}

void memory_bind(int cpu_id)
{
	char numa_id_str[12];
	struct bitmask *numa_node;
	int numa_id = numa_node_of_cpu(cpu_id);
	sprintf(numa_id_str, "%d", numa_id);
	numa_node = numa_parse_nodestring(numa_id_str);
	numa_set_membind(numa_node);
	numa_free_nodemask(numa_node);
}

size_t align_to_huge_page_size(size_t size) {
    long huge_page_size = 2 * 1024 * 1024;  // 2MB huge pages
    return (size + huge_page_size - 1) & ~(huge_page_size - 1);
}

void *mamalloc(size_t size) {
    void *ptr = NULL;
    size = align_to_huge_page_size(size);  // Align size to huge page boundary

    // Use mmap to allocate memory with huge pages
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }
    return ptr;
}

void *mamalloc_2(size_t size)
{
	void *ptr = NULL;
	return posix_memalign(&ptr, 64, size) ? NULL : ptr;
}

void swap_i(int *x, int *y)
{
	int t = *x; *x = *y; *y = t;
}

void schedule_threads(int *cpu, int *numa_node, int threads, int numa)
{
	int max_numa = numa_max_node() + 1;
	int max_threads = hardware_threads();
	int max_threads_per_numa = max_threads / max_numa;
	int t, threads_per_numa = threads / numa;
	assert(numa > 0 && threads >= numa && threads % numa == 0);
	if (numa > max_numa ||
	    threads > max_threads ||
	    threads_per_numa > max_threads_per_numa)
		for (t = 0 ; t != threads ; ++t) {
			cpu[t] = t;
			numa_node[t] = t / threads_per_numa;
		}
	else {
		int *thread_numa = malloc(max_threads * sizeof(int));
		for (t = 0 ; t != max_threads ; ++t)
			thread_numa[t] = numa_node_of_cpu(t);
		for (t = 0 ; t != threads ; ++t) {
			int i, n = t % numa;
			for (i = 0 ; i != max_threads ; ++i)
				if (thread_numa[i] == n) break;
			assert(i != max_threads);
			thread_numa[i] = -1;
			cpu[t] = i;
			if (numa_node != NULL)
				numa_node[t] = n;
			assert(numa_node_of_cpu(i) == n);
		}
		free(thread_numa);
	}
}

inline uint64_t mulhi(uint64_t x, uint64_t y)
{
	uint64_t l, h;
	asm("mulq	%2"
	: "=a"(l), "=d"(h)
	: "r"(y), "a"(x)
	: "cc");
	return h;
}

int ceil_div(int x, int y)
{
	return (x + y - 1) / y;
}

int ceil_log_2(uint64_t x)
{
	uint8_t power = 0;
	uint64_t one = 1;
	while ((one << power) < x && power != 64)
		power++;
	return power;
}

int distribute_bits(int bits, int numa, int pass[], int print)
{
	int numa_bits = ceil_log_2(numa);
	int end_bits = numa_bits > 0 ? 1 : 0;
	int total_bits = bits + numa_bits;
	int limit[] = {12, 23, 34, 45, 56, 67};

	if (global_tuples >= 100) {
		limit[0] = 14;
		limit[1] = 27;
		limit[2] = 40; 
		limit[3] = 53;
		limit[4] = 66;
	}
	// int limit[] = {12, 23, 34, 45, 56, 67};

	// int limit[] = {13, 25, 37, 49, 61, 73};

	

    // int limit[] = {15, 29, 43, 57, 71};

	// int limit[] = {16, 31, 46, 61, 76};

	// int limit[] = {17, 33, 49, 66};

    // int limit[] = {9, 17, 25, 33, 41, 49, 57, 65};

	// determine how many passes to do
	int p, passes = 0;
	while (limit[passes++] < total_bits);
	// first pass gets numa split
	pass[0] = ceil_div(total_bits - end_bits, passes) - numa_bits;
	int rem_bits = bits - pass[0];
	for (p = 1 ; p != passes ; ++p) {
		pass[p] = ceil_div(rem_bits - end_bits, passes - p);
		rem_bits -= pass[p];
	}
	pass[passes - 1] += end_bits;
	assert(rem_bits == end_bits);
	// print bit passes
	if (print) {
		fprintf(stderr, "Passes:  %d", pass[0]);
		if (numa_bits) fprintf(stderr, "(+%d)", numa_bits);
		for (p = 1 ; p != passes ; ++p)
			fprintf(stderr, " -> %d", pass[p]);
		fprintf(stderr, "\n");
	}
	pass[passes] = 0;
	return passes;
}

void partition_keys(uint64_t *keys, uint64_t *keys_out, uint64_t size,
                    uint64_t **hist, uint8_t shift_bits, uint8_t radix_bits,
                    int thread_id, int threads, pthread_barrier_t *barrier)
{
	// inputs and outputs must be aligned
	assert(0 == (15 & (size_t) keys));
	assert(0 == (63 & (size_t) keys_out));
	// set sub-array for current thread
	uint64_t local_size = (size / threads) & ~15;
	uint64_t *local_keys = &keys[local_size * thread_id];
	if (thread_id + 1 == threads)
		local_size = size - local_size * thread_id;
	// initialize histogram
	uint64_t i, j, p; int t;
	uint64_t partitions = 1 << radix_bits;
	uint64_t *local_hist = hist[thread_id];
	for (p = 0 ; p != partitions ; ++p)
		local_hist[p] = 0;
	// main histogram loop
	__m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
	__m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
	for (i = p = 0 ; i != local_size ; i += 4) {
		__m128i k12 = _mm_load_si128((__m128i*) &local_keys[i]);
		__m128i k34 = _mm_load_si128((__m128i*) &local_keys[i + 2]);
		__m128i h12 = _mm_srl_epi64(k12, s);
		__m128i h34 = _mm_srl_epi64(k34, s);
		h12 = _mm_and_si128(h12, m);
		h34 = _mm_and_si128(h34, m);
		__m128i h = _mm_packus_epi32(h12, h34);
		for (j = 0 ; j != 4 ; ++j) {
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			local_hist[p]++;
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
		}
	}
	// wait all threads to complete histogram generation
	pthread_barrier_wait(&barrier[0]);
	// initialize buffer
	uint64_t *buf = mamalloc((partitions << 3) * sizeof(uint64_t));
	for (i = p = 0 ; p != partitions ; ++p) {
		for (t = 0 ; t != thread_id ; ++t)
			i += hist[t][p];
		buf[(p << 3) | 7] = i;
		for (; t != threads ; ++t)
			i += hist[t][p];
	}
	assert(i == size);
	// main partitioning loop
	for (i = p = 0 ; i != local_size ; i += 4) {
		__m128i k12 = _mm_load_si128((__m128i*) &local_keys[i]);
		__m128i k34 = _mm_load_si128((__m128i*) &local_keys[i + 2]);
		__m128i h12 = _mm_srl_epi64(k12, s);
		__m128i h34 = _mm_srl_epi64(k34, s);
		h12 = _mm_and_si128(h12, m);
		h34 = _mm_and_si128(h34, m);
		k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
		k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
		__m128i h = _mm_packus_epi32(h12, h34);
		__m128i k_L = _mm_unpacklo_epi64(k12, k34);
		__m128i k_H = _mm_unpackhi_epi64(k12, k34);
		h = _mm_slli_epi32(h, 3);
		for (j = 0 ; j != 4 ; ++j) {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			// offset in the cache line pair
			uint64_t *src = &buf[p];
			uint64_t index = src[7]++;
			uint64_t offset = index & 7;
			__m128i k = _mm_unpacklo_epi32(k_L, k_H);
			_mm_storel_epi64((__m128i*) &src[offset], k);
			if (offset == 7) {
				uint64_t *dst = &keys_out[index - 7];
				__m128i r0 = _mm_load_si128((__m128i*) &src[0]);
				__m128i r1 = _mm_load_si128((__m128i*) &src[2]);
				__m128i r2 = _mm_load_si128((__m128i*) &src[4]);
				__m128i r3 = _mm_load_si128((__m128i*) &src[6]);
				_mm_stream_si128((__m128i*) &dst[0], r0);
				_mm_stream_si128((__m128i*) &dst[2], r1);
				_mm_stream_si128((__m128i*) &dst[4], r2);
				_mm_stream_si128((__m128i*) &dst[6], r3);
				src[7] = index + 1;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
			k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
		}
	}
	// wait all threads to complete main partition part
	pthread_barrier_wait(&barrier[1]);
	// flush remaining items from buffers to output
	for (p = 0 ; p != partitions ; ++p) {
		uint64_t *src = &buf[p << 3];
		uint64_t index = src[7];
		uint64_t remain = index & 7;
		uint64_t offset = 0;
		if (remain > local_hist[p])
			offset = remain - local_hist[p];
		index -= remain - offset;
		while (offset != remain)
			_mm_stream_si64((long long *) &keys_out[index++], src[offset++]);
	}
	// wait all threads to complete last partition part
	pthread_barrier_wait(&barrier[2]);
	// free(buf);

}

void free_with_hugepages(uint64_t *ptr, uint64_t size) {
    if (munmap(ptr, size * sizeof(uint64_t)) == -1) {
        perror("munmap");
    }
}

void histogram_numa_2(uint64_t *keys, uint64_t size, uint64_t *count,
                      uint8_t radix_bits, uint64_t delim[])
{
	assert(radix_bits <= 16);
	uint64_t convert = ((uint64_t) 1) << 63;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
	__m128i d = _mm_set1_epi64x(delim[0] - convert);
	__m128i c = _mm_set1_epi64x(convert);
	__m128i k12, k34; int i;
	// check for unaligned keys
	uint64_t p = 0; i = 0;
	uint64_t unaligned_keys[4];
	while ((15 & ((uint64_t) keys)) && i != size)
		unaligned_keys[i++] = *keys++;
	uint64_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint64_t *keys_end = &keys[size - i];
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k12 = _mm_load_si128((__m128i*) &keys[0]);
		k34 = _mm_load_si128((__m128i*) &keys[2]);
		keys += 4; i = 4;
		unaligned_intro:;
		__m128i h12 = _mm_and_si128(k12, m);
		__m128i h34 = _mm_and_si128(k34, m);
		__m128i h = _mm_packus_epi32(h12, h34);
		__m128i k12_c = _mm_sub_epi64(k12, c);
		__m128i k34_c = _mm_sub_epi64(k34, c);
		__m128i e_L = _mm_cmpgt_epi64(k12_c, d);
		__m128i e_H = _mm_cmpgt_epi64(k34_c, d);
		__m128i e = _mm_packs_epi32(e_L, e_H);
		__m128i r = _mm_setzero_si128();
		r = _mm_sub_epi32(r, e);
		r = _mm_sll_epi32(r, s);
		h = _mm_or_si128(h, r);
		do {
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			count[p]++;
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end)
		unaligned_keys[i++] = *keys++;
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
}

void partition_numa_2(uint64_t *keys, uint64_t *rids, uint64_t size,
                      uint64_t *offsets, uint64_t *sizes, uint64_t *buf,
                      uint64_t *keys_out, uint64_t *rids_out,
                      uint8_t radix_bits, uint64_t delim[])
{
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	assert(radix_bits <= 16);
	int i, partitions = 1 << (radix_bits + 1);
	// initialize buffers
	for (i = 0 ; i != partitions ; ++i)
		buf[(i << 4) | 14] = offsets[i] << 1;
	// main partition loop
	uint64_t convert = ((uint64_t) 1) << 63;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
	__m128i d = _mm_set1_epi64x(delim[0] - convert);
	__m128i c = _mm_set1_epi64x(convert);
	__m128i k12, k34, v12, v34;
	// check for unaligned items
	uint64_t unaligned_keys[4];
	uint64_t unaligned_rids[4];
	uint64_t p = 0; i = 0;
	while ((15 & ((uint64_t) keys)) && i != size) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	assert(i == size || (15 & (uint64_t) rids) == 0);
	uint32_t *keys_32 = (uint32_t*) keys_out;
	uint32_t *rids_32 = (uint32_t*) rids_out;
	uint64_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint64_t *keys_end = &keys[size - i];
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		v12 = _mm_loadu_si128((__m128i*) &unaligned_rids[0]);
		v34 = _mm_loadu_si128((__m128i*) &unaligned_rids[2]);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k12 = _mm_load_si128((__m128i*) &keys[0]);
		k34 = _mm_load_si128((__m128i*) &keys[2]);
		v12 = _mm_load_si128((__m128i*) &rids[0]);
		v34 = _mm_load_si128((__m128i*) &rids[2]);
		keys += 4; rids += 4; i = 4;
		unaligned_intro:;
		__m128i k12_c = _mm_sub_epi64(k12, c);
		__m128i k34_c = _mm_sub_epi64(k34, c);
		__m128i e_L = _mm_cmpgt_epi64(k12_c, d);
		__m128i e_H = _mm_cmpgt_epi64(k34_c, d);
		__m128i e = _mm_packs_epi32(e_L, e_H);
		__m128i h12 = _mm_and_si128(k12, m);
		__m128i h34 = _mm_and_si128(k34, m);
		k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
		k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
		v12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
		v34 = _mm_shuffle_epi32(v34, _MM_SHUFFLE(3, 1, 2, 0));
		__m128i k_L = _mm_unpacklo_epi64(k12, k34);
		__m128i k_H = _mm_unpackhi_epi64(k12, k34);
		__m128i v_L = _mm_unpacklo_epi64(v12, v34);
		__m128i v_H = _mm_unpackhi_epi64(v12, v34);
		__m128i h = _mm_packus_epi32(h12, h34);
		__m128i r = _mm_setzero_si128();
		r = _mm_sub_epi32(r, e);
		r = _mm_sll_epi32(r, s);
		h = _mm_or_si128(h, r);
		h = _mm_slli_epi32(h, 4);
		do {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			// offset in the cache line pair
			uint64_t *src = &buf[p];
			uint64_t index = src[14];
			src[14] = index + 2;
			uint64_t offset = index & 15;
			// pack and store
			__m128i kkxx = _mm_unpacklo_epi32(k_L, k_H);
			__m128i vvxx = _mm_unpacklo_epi32(v_L, v_H);
			__m128i kkvv = _mm_unpacklo_epi64(kkxx, vvxx);
			_mm_store_si128((__m128i*) &src[offset], kkvv);
			if (offset == 14) {
				uint32_t *dest_x = &keys_32[index - 14];
				uint32_t *dest_y = &rids_32[index - 14];
				// load cache line from cache to 8 128-bit registers
				__m128i r0 = _mm_load_si128((__m128i*) &src[0]);
				__m128i r1 = _mm_load_si128((__m128i*) &src[2]);
				__m128i r2 = _mm_load_si128((__m128i*) &src[4]);
				__m128i r3 = _mm_load_si128((__m128i*) &src[6]);
				__m128i r4 = _mm_load_si128((__m128i*) &src[8]);
				__m128i r5 = _mm_load_si128((__m128i*) &src[10]);
				__m128i r6 = _mm_load_si128((__m128i*) &src[12]);
				__m128i r7 = _mm_load_si128((__m128i*) &src[14]);
				// split first column
				__m128i x0 = _mm_unpacklo_epi64(r0, r1);
				__m128i x1 = _mm_unpacklo_epi64(r2, r3);
				__m128i x2 = _mm_unpacklo_epi64(r4, r5);
				__m128i x3 = _mm_unpacklo_epi64(r6, r7);
				// stream first column
				_mm_stream_si128((__m128i*) &dest_x[0], x0);
				_mm_stream_si128((__m128i*) &dest_x[4], x1);
				_mm_stream_si128((__m128i*) &dest_x[8], x2);
				_mm_stream_si128((__m128i*) &dest_x[12],x3);
				// split second column
				__m128i y0 = _mm_unpackhi_epi64(r0, r1);
				__m128i y1 = _mm_unpackhi_epi64(r2, r3);
				__m128i y2 = _mm_unpackhi_epi64(r4, r5);
				__m128i y3 = _mm_unpackhi_epi64(r6, r7);
				// stream second column
				_mm_stream_si128((__m128i*) &dest_y[0], y0);
				_mm_stream_si128((__m128i*) &dest_y[4], y1);
				_mm_stream_si128((__m128i*) &dest_y[8], y2);
				_mm_stream_si128((__m128i*) &dest_y[12],y3);
				// restore overwritten pointer
				src[14] = index + 2;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
			k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
			v_L = _mm_shuffle_epi32(v_L, _MM_SHUFFLE(0, 3, 2, 1));
			v_H = _mm_shuffle_epi32(v_H, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		v12 = _mm_loadu_si128((__m128i*) &unaligned_rids[0]);
		v34 = _mm_loadu_si128((__m128i*) &unaligned_rids[2]);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
#ifdef BG
	// check partition sanity
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t index = buf[(i << 4) | 14] >> 1;
		assert(index - offsets[i] == sizes[i]);
	}
#endif
}

void histogram_numa_4(uint64_t *keys, uint64_t size, uint64_t *count,
                      uint8_t radix_bits, uint64_t delim[])
{
	assert(radix_bits <= 16);
	uint64_t convert = ((uint64_t) 1) << 63;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
	__m128i d1 = _mm_set1_epi64x(delim[0] - convert);
	__m128i d2 = _mm_set1_epi64x(delim[1] - convert);
	__m128i d3 = _mm_set1_epi64x(delim[2] - convert);
	__m128i c = _mm_set1_epi64x(convert);
	__m128i k12, k34; int i;
	// check for unaligned keys
	uint64_t p = 0; i = 0;
	uint64_t unaligned_keys[4];
	while ((15 & ((uint64_t) keys)) && i != size)
		unaligned_keys[i++] = *keys++;
	uint64_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint64_t *keys_end = &keys[size - i];
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k12 = _mm_load_si128((__m128i*) &keys[0]);
		k34 = _mm_load_si128((__m128i*) &keys[2]);
		keys += 4; i = 4;
		unaligned_intro:;
		__m128i h12 = _mm_and_si128(k12, m);
		__m128i h34 = _mm_and_si128(k34, m);
		__m128i h = _mm_packus_epi32(h12, h34);
		__m128i k12_c = _mm_sub_epi64(k12, c);
		__m128i k34_c = _mm_sub_epi64(k34, c);
		__m128i e1_L = _mm_cmpgt_epi64(k12_c, d2);
		__m128i e1_H = _mm_cmpgt_epi64(k34_c, d2);
		__m128i d13_L = _mm_blendv_epi8(d1, d3, e1_L);
		__m128i d13_H = _mm_blendv_epi8(d1, d3, e1_H);
		__m128i e1 = _mm_packs_epi32(e1_L, e1_H);
		__m128i e2_L = _mm_cmpgt_epi64(k12_c, d13_L);
		__m128i e2_H = _mm_cmpgt_epi64(k34_c, d13_H);
		__m128i e2 = _mm_packs_epi32(e2_L, e2_H);
		__m128i r = _mm_setzero_si128();
		r = _mm_sub_epi32(r, e1);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e2);
		r = _mm_sll_epi32(r, s);
		h = _mm_or_si128(h, r);
		do {
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			count[p]++;
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end)
		unaligned_keys[i++] = *keys++;
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
}

void partition_numa_4(uint64_t *keys, uint64_t *rids, uint64_t size,
                      uint64_t *offsets, uint64_t *sizes, uint64_t *buf,
                      uint64_t *keys_out, uint64_t *rids_out,
                      uint8_t radix_bits, uint64_t delim[])
{
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	assert(radix_bits <= 16);
	int i, partitions = 1 << (radix_bits + 2);
	// initialize buffers
	for (i = 0 ; i != partitions ; ++i)
		buf[(i << 4) | 14] = offsets[i] << 1;
	// main partition loop
	uint64_t convert = ((uint64_t) 1) << 63;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
	__m128i d1 = _mm_set1_epi64x(delim[0] - convert);
	__m128i d2 = _mm_set1_epi64x(delim[1] - convert);
	__m128i d3 = _mm_set1_epi64x(delim[2] - convert);
	__m128i c = _mm_set1_epi64x(convert);
	__m128i k12, k34, v12, v34;
	// check for unaligned items
	uint64_t unaligned_keys[4];
	uint64_t unaligned_rids[4];
	uint64_t p = 0; i = 0;
	while ((15 & ((uint64_t) keys)) && i != size) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	assert(i == size || (15 & (uint64_t) rids) == 0);
	uint32_t *keys_32 = (uint32_t*) keys_out;
	uint32_t *rids_32 = (uint32_t*) rids_out;
	uint64_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint64_t *keys_end = &keys[size - i];
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		v12 = _mm_loadu_si128((__m128i*) &unaligned_rids[0]);
		v34 = _mm_loadu_si128((__m128i*) &unaligned_rids[2]);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k12 = _mm_load_si128((__m128i*) &keys[0]);
		k34 = _mm_load_si128((__m128i*) &keys[2]);
		v12 = _mm_load_si128((__m128i*) &rids[0]);
		v34 = _mm_load_si128((__m128i*) &rids[2]);
		keys += 4; rids += 4; i = 4;
		unaligned_intro:;
		__m128i k12_c = _mm_sub_epi64(k12, c);
		__m128i k34_c = _mm_sub_epi64(k34, c);
		__m128i e1_L = _mm_cmpgt_epi64(k12_c, d2);
		__m128i e1_H = _mm_cmpgt_epi64(k34_c, d2);
		__m128i d13_L = _mm_blendv_epi8(d1, d3, e1_L);
		__m128i d13_H = _mm_blendv_epi8(d1, d3, e1_H);
		__m128i e1 = _mm_packs_epi32(e1_L, e1_H);
		__m128i e2_L = _mm_cmpgt_epi64(k12_c, d13_L);
		__m128i e2_H = _mm_cmpgt_epi64(k34_c, d13_H);
		__m128i e2 = _mm_packs_epi32(e2_L, e2_H);
		__m128i h12 = _mm_and_si128(k12, m);
		__m128i h34 = _mm_and_si128(k34, m);
		k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
		k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
		v12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
		v34 = _mm_shuffle_epi32(v34, _MM_SHUFFLE(3, 1, 2, 0));
		__m128i k_L = _mm_unpacklo_epi64(k12, k34);
		__m128i k_H = _mm_unpackhi_epi64(k12, k34);
		__m128i v_L = _mm_unpacklo_epi64(v12, v34);
		__m128i v_H = _mm_unpackhi_epi64(v12, v34);
		__m128i h = _mm_packus_epi32(h12, h34);
		__m128i r = _mm_setzero_si128();
		r = _mm_sub_epi32(r, e1);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e2);
		r = _mm_sll_epi32(r, s);
		h = _mm_or_si128(h, r);
		h = _mm_slli_epi32(h, 4);
		do {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			// offset in the cache line pair
			uint64_t *src = &buf[p];
			uint64_t index = src[14];
			src[14] = index + 2;
			uint64_t offset = index & 15;
			// pack and store
			__m128i kkxx = _mm_unpacklo_epi32(k_L, k_H);
			__m128i vvxx = _mm_unpacklo_epi32(v_L, v_H);
			__m128i kkvv = _mm_unpacklo_epi64(kkxx, vvxx);
			_mm_store_si128((__m128i*) &src[offset], kkvv);
			if (offset == 14) {
				uint32_t *dest_x = &keys_32[index - 14];
				uint32_t *dest_y = &rids_32[index - 14];
				// load cache line from cache to 8 128-bit registers
				__m128i r0 = _mm_load_si128((__m128i*) &src[0]);
				__m128i r1 = _mm_load_si128((__m128i*) &src[2]);
				__m128i r2 = _mm_load_si128((__m128i*) &src[4]);
				__m128i r3 = _mm_load_si128((__m128i*) &src[6]);
				__m128i r4 = _mm_load_si128((__m128i*) &src[8]);
				__m128i r5 = _mm_load_si128((__m128i*) &src[10]);
				__m128i r6 = _mm_load_si128((__m128i*) &src[12]);
				__m128i r7 = _mm_load_si128((__m128i*) &src[14]);
				// split first column
				__m128i x0 = _mm_unpacklo_epi64(r0, r1);
				__m128i x1 = _mm_unpacklo_epi64(r2, r3);
				__m128i x2 = _mm_unpacklo_epi64(r4, r5);
				__m128i x3 = _mm_unpacklo_epi64(r6, r7);
				// stream first column
				_mm_stream_si128((__m128i*) &dest_x[0], x0);
				_mm_stream_si128((__m128i*) &dest_x[4], x1);
				_mm_stream_si128((__m128i*) &dest_x[8], x2);
				_mm_stream_si128((__m128i*) &dest_x[12],x3);
				// split second column
				__m128i y0 = _mm_unpackhi_epi64(r0, r1);
				__m128i y1 = _mm_unpackhi_epi64(r2, r3);
				__m128i y2 = _mm_unpackhi_epi64(r4, r5);
				__m128i y3 = _mm_unpackhi_epi64(r6, r7);
				// stream second column
				_mm_stream_si128((__m128i*) &dest_y[0], y0);
				_mm_stream_si128((__m128i*) &dest_y[4], y1);
				_mm_stream_si128((__m128i*) &dest_y[8], y2);
				_mm_stream_si128((__m128i*) &dest_y[12],y3);
				// restore overwritten pointer
				src[14] = index + 2;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
			k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
			v_L = _mm_shuffle_epi32(v_L, _MM_SHUFFLE(0, 3, 2, 1));
			v_H = _mm_shuffle_epi32(v_H, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		v12 = _mm_loadu_si128((__m128i*) &unaligned_rids[0]);
		v34 = _mm_loadu_si128((__m128i*) &unaligned_rids[2]);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
#ifdef BG
	// check partition sanity
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t index = buf[(i << 4) | 14] >> 1;
		assert(index - offsets[i] == sizes[i]);
	}
#endif
}

void histogram_numa_8(uint64_t *keys, uint64_t size, uint64_t *count,
                      uint8_t radix_bits, uint64_t delim[])
{
	assert(radix_bits <= 16);
	uint64_t convert = ((uint64_t) 1) << 63;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
	__m128i d1 = _mm_set1_epi64x(delim[0] - convert);
	__m128i d2 = _mm_set1_epi64x(delim[1] - convert);
	__m128i d3 = _mm_set1_epi64x(delim[2] - convert);
	__m128i d4 = _mm_set1_epi64x(delim[3] - convert);
	__m128i d5 = _mm_set1_epi64x(delim[4] - convert);
	__m128i d6 = _mm_set1_epi64x(delim[5] - convert);
	__m128i d7 = _mm_set1_epi64x(delim[6] - convert);
	__m128i c = _mm_set1_epi64x(convert);
	__m128i k12, k34; int i;
	// check for unaligned keys
	uint64_t p = 0; i = 0;
	uint64_t unaligned_keys[4];
	while ((15 & ((uint64_t) keys)) && i != size)
		unaligned_keys[i++] = *keys++;
	uint64_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint64_t *keys_end = &keys[size - i];
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k12 = _mm_load_si128((__m128i*) &keys[0]);
		k34 = _mm_load_si128((__m128i*) &keys[2]);
		keys += 4; i = 4;
		unaligned_intro:;
		__m128i h12 = _mm_and_si128(k12, m);
		__m128i h34 = _mm_and_si128(k34, m);
		__m128i h = _mm_packus_epi32(h12, h34);
		__m128i k12_c = _mm_sub_epi64(k12, c);
		__m128i k34_c = _mm_sub_epi64(k34, c);
		__m128i e1_L = _mm_cmpgt_epi64(k12_c, d4);
		__m128i e1_H = _mm_cmpgt_epi64(k34_c, d4);
		__m128i d26_L = _mm_blendv_epi8(d2, d6, e1_L);
		__m128i d26_H = _mm_blendv_epi8(d2, d6, e1_H);
		__m128i d15_L = _mm_blendv_epi8(d1, d5, e1_L);
		__m128i d15_H = _mm_blendv_epi8(d1, d5, e1_H);
		__m128i d37_L = _mm_blendv_epi8(d3, d7, e1_L);
		__m128i d37_H = _mm_blendv_epi8(d3, d7, e1_H);
		__m128i e1 = _mm_packs_epi32(e1_L, e1_H);
		__m128i e2_L = _mm_cmpgt_epi64(k12_c, d26_L);
		__m128i e2_H = _mm_cmpgt_epi64(k34_c, d26_H);
		__m128i d1357_L = _mm_blendv_epi8(d15_L, d37_L, e2_L);
		__m128i d1357_H = _mm_blendv_epi8(d15_H, d37_H, e2_H);
		__m128i e2 = _mm_packs_epi32(e2_L, e2_H);
		__m128i e3_L = _mm_cmpgt_epi64(k12_c, d1357_L);
		__m128i e3_H = _mm_cmpgt_epi64(k34_c, d1357_H);
		__m128i e3 = _mm_packs_epi32(e3_L, e3_H);
		__m128i r = _mm_setzero_si128();
		r = _mm_sub_epi32(r, e1);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e2);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e3);
		r = _mm_sll_epi32(r, s);
		h = _mm_or_si128(h, r);
		do {
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			count[p]++;
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end)
		unaligned_keys[i++] = *keys++;
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
}

void partition_numa_8(uint64_t *keys, uint64_t *rids, uint64_t size,
                      uint64_t *offsets, uint64_t *sizes, uint64_t *buf,
                      uint64_t *keys_out, uint64_t *rids_out,
                      uint8_t radix_bits, uint64_t delim[])
{
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	assert(radix_bits <= 16);
	int i, partitions = 1 << (radix_bits + 3);
	// initialize buffers
	for (i = 0 ; i != partitions ; ++i)
		buf[(i << 4) | 14] = offsets[i] << 1;
	// main partition loop
	uint64_t convert = ((uint64_t) 1) << 63;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
	__m128i d1 = _mm_set1_epi64x(delim[0] - convert);
	__m128i d2 = _mm_set1_epi64x(delim[1] - convert);
	__m128i d3 = _mm_set1_epi64x(delim[2] - convert);
	__m128i d4 = _mm_set1_epi64x(delim[3] - convert);
	__m128i d5 = _mm_set1_epi64x(delim[4] - convert);
	__m128i d6 = _mm_set1_epi64x(delim[5] - convert);
	__m128i d7 = _mm_set1_epi64x(delim[6] - convert);
	__m128i c = _mm_set1_epi64x(convert);
	__m128i k12, k34, v12, v34;
	// check for unaligned items
	uint64_t unaligned_keys[4];
	uint64_t unaligned_rids[4];
	uint64_t p = 0; i = 0;
	while ((15 & ((uint64_t) keys)) && i != size) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	assert(i == size || (15 & (uint64_t) rids) == 0);
	uint32_t *keys_32 = (uint32_t*) keys_out;
	uint32_t *rids_32 = (uint32_t*) rids_out;
	uint64_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint64_t *keys_end = &keys[size - i];
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		v12 = _mm_loadu_si128((__m128i*) &unaligned_rids[0]);
		v34 = _mm_loadu_si128((__m128i*) &unaligned_rids[2]);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k12 = _mm_load_si128((__m128i*) &keys[0]);
		k34 = _mm_load_si128((__m128i*) &keys[2]);
		v12 = _mm_load_si128((__m128i*) &rids[0]);
		v34 = _mm_load_si128((__m128i*) &rids[2]);
		keys += 4; rids += 4; i = 4;
		unaligned_intro:;
		__m128i k12_c = _mm_sub_epi64(k12, c);
		__m128i k34_c = _mm_sub_epi64(k34, c);
		__m128i e1_L = _mm_cmpgt_epi64(k12_c, d4);
		__m128i e1_H = _mm_cmpgt_epi64(k34_c, d4);
		__m128i d26_L = _mm_blendv_epi8(d2, d6, e1_L);
		__m128i d26_H = _mm_blendv_epi8(d2, d6, e1_H);
		__m128i d15_L = _mm_blendv_epi8(d1, d5, e1_L);
		__m128i d15_H = _mm_blendv_epi8(d1, d5, e1_H);
		__m128i d37_L = _mm_blendv_epi8(d3, d7, e1_L);
		__m128i d37_H = _mm_blendv_epi8(d3, d7, e1_H);
		__m128i e1 = _mm_packs_epi32(e1_L, e1_H);
		__m128i e2_L = _mm_cmpgt_epi64(k12_c, d26_L);
		__m128i e2_H = _mm_cmpgt_epi64(k34_c, d26_H);
		__m128i d1357_L = _mm_blendv_epi8(d15_L, d37_L, e2_L);
		__m128i d1357_H = _mm_blendv_epi8(d15_H, d37_H, e2_H);
		__m128i e2 = _mm_packs_epi32(e2_L, e2_H);
		__m128i e3_L = _mm_cmpgt_epi64(k12_c, d1357_L);
		__m128i e3_H = _mm_cmpgt_epi64(k34_c, d1357_H);
		__m128i e3 = _mm_packs_epi32(e3_L, e3_H);
		__m128i r = _mm_setzero_si128();
		r = _mm_sub_epi32(r, e1);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e2);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e3);
		__m128i h12 = _mm_and_si128(k12, m);
		__m128i h34 = _mm_and_si128(k34, m);
		k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
		k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
		v12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
		v34 = _mm_shuffle_epi32(v34, _MM_SHUFFLE(3, 1, 2, 0));
		__m128i k_L = _mm_unpacklo_epi64(k12, k34);
		__m128i k_H = _mm_unpackhi_epi64(k12, k34);
		__m128i v_L = _mm_unpacklo_epi64(v12, v34);
		__m128i v_H = _mm_unpackhi_epi64(v12, v34);
		__m128i h = _mm_packus_epi32(h12, h34);
		r = _mm_sll_epi32(r, s);
		h = _mm_or_si128(h, r);
		h = _mm_slli_epi32(h, 4);
		do {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			// offset in the cache line pair
			uint64_t *src = &buf[p];
			uint64_t index = src[14];
			src[14] = index + 2;
			uint64_t offset = index & 15;
			// pack and store
			__m128i kkxx = _mm_unpacklo_epi32(k_L, k_H);
			__m128i vvxx = _mm_unpacklo_epi32(v_L, v_H);
			__m128i kkvv = _mm_unpacklo_epi64(kkxx, vvxx);
			_mm_store_si128((__m128i*) &src[offset], kkvv);
			if (offset == 14) {
				uint32_t *dest_x = &keys_32[index - 14];
				uint32_t *dest_y = &rids_32[index - 14];
				// load cache line from cache to 8 128-bit registers
				__m128i r0 = _mm_load_si128((__m128i*) &src[0]);
				__m128i r1 = _mm_load_si128((__m128i*) &src[2]);
				__m128i r2 = _mm_load_si128((__m128i*) &src[4]);
				__m128i r3 = _mm_load_si128((__m128i*) &src[6]);
				__m128i r4 = _mm_load_si128((__m128i*) &src[8]);
				__m128i r5 = _mm_load_si128((__m128i*) &src[10]);
				__m128i r6 = _mm_load_si128((__m128i*) &src[12]);
				__m128i r7 = _mm_load_si128((__m128i*) &src[14]);
				// split first column
				__m128i x0 = _mm_unpacklo_epi64(r0, r1);
				__m128i x1 = _mm_unpacklo_epi64(r2, r3);
				__m128i x2 = _mm_unpacklo_epi64(r4, r5);
				__m128i x3 = _mm_unpacklo_epi64(r6, r7);
				// stream first column
				_mm_stream_si128((__m128i*) &dest_x[0], x0);
				_mm_stream_si128((__m128i*) &dest_x[4], x1);
				_mm_stream_si128((__m128i*) &dest_x[8], x2);
				_mm_stream_si128((__m128i*) &dest_x[12],x3);
				// split second column
				__m128i y0 = _mm_unpackhi_epi64(r0, r1);
				__m128i y1 = _mm_unpackhi_epi64(r2, r3);
				__m128i y2 = _mm_unpackhi_epi64(r4, r5);
				__m128i y3 = _mm_unpackhi_epi64(r6, r7);
				// stream second column
				_mm_stream_si128((__m128i*) &dest_y[0], y0);
				_mm_stream_si128((__m128i*) &dest_y[4], y1);
				_mm_stream_si128((__m128i*) &dest_y[8], y2);
				_mm_stream_si128((__m128i*) &dest_y[12],y3);
				// restore overwritten pointer
				src[14] = index + 2;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
			k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
			v_L = _mm_shuffle_epi32(v_L, _MM_SHUFFLE(0, 3, 2, 1));
			v_H = _mm_shuffle_epi32(v_H, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		v12 = _mm_loadu_si128((__m128i*) &unaligned_rids[0]);
		v34 = _mm_loadu_si128((__m128i*) &unaligned_rids[2]);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
#ifdef BG
	// check partition sanity
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t index = buf[(i << 4) | 14] >> 1;
		assert(index - offsets[i] == sizes[i]);
	}
#endif
}


void histogram(uint64_t *keys, uint64_t size, uint64_t *count,
               uint8_t shift_bits, uint8_t radix_bits)
{
    assert(radix_bits <= 16);
    __m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
    __m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
    __m128i k12, k34; 
    int i;
    uint64_t p = 0;

    // Align the keys pointer
    uint64_t unaligned_keys[4];
    int unaligned_count = 0;
    while (((uintptr_t)keys & 15) && size > 0) {
        unaligned_keys[unaligned_count++] = *keys++;
        size--;
    }

    uint64_t *keys_aligned_end = &keys[size & ~3];
    uint64_t *keys_end = &keys[size];

    while (keys < keys_aligned_end) {
        // Prefetch the next cache line
        _mm_prefetch((char *)(keys + 128), _MM_HINT_T0);

        k12 = _mm_load_si128((__m128i*) &keys[0]);
        k34 = _mm_load_si128((__m128i*) &keys[2]);
        keys += 4;

        __m128i h12 = _mm_srl_epi64(k12, s);
        __m128i h34 = _mm_srl_epi64(k34, s);
        h12 = _mm_and_si128(h12, m);
        h34 = _mm_and_si128(h34, m);
        __m128i h = _mm_packus_epi32(h12, h34);

        for (i = 0; i < 4; i++) {
            asm("movd %1, %%eax" : "=a"(p) : "x"(h), "0"(p));
            count[p]++;
            h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
        }
    }

    // Handle remaining unaligned keys
    if (unaligned_count > 0) {
        k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
        if (unaligned_count > 2) {
            k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
        } else {
            k34 = _mm_setzero_si128();
        }

        __m128i h12 = _mm_srl_epi64(k12, s);
        __m128i h34 = _mm_srl_epi64(k34, s);
        h12 = _mm_and_si128(h12, m);
        h34 = _mm_and_si128(h34, m);
        __m128i h = _mm_packus_epi32(h12, h34);

        for (i = 0; i < unaligned_count; i++) {
            asm("movd %1, %%eax" : "=a"(p) : "x"(h), "0"(p));
            count[p]++;
            h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
        }
    }

    // Handle the final unaligned keys
    unaligned_count = 0;
    while (keys < keys_end) {
        unaligned_keys[unaligned_count++] = *keys++;
    }
    if (unaligned_count > 0) {
        k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
        if (unaligned_count > 2) {
            k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
        } else {
            k34 = _mm_setzero_si128();
        }

        __m128i h12 = _mm_srl_epi64(k12, s);
        __m128i h34 = _mm_srl_epi64(k34, s);
        h12 = _mm_and_si128(h12, m);
        h34 = _mm_and_si128(h34, m);
        __m128i h = _mm_packus_epi32(h12, h34);

        for (i = 0; i < unaligned_count; i++) {
            asm("movd %1, %%eax" : "=a"(p) : "x"(h), "0"(p));
            count[p]++;
            h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
        }
    }
}

void histogram_old(uint64_t *keys, uint64_t size, uint64_t *count,
	       uint8_t shift_bits, uint8_t radix_bits)
{
	assert(radix_bits <= 16);
	__m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
	__m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
	__m128i k12, k34; int i;
	// check for unaligned keys
	uint64_t p = 0; i = 0;
	uint64_t unaligned_keys[4];
	while ((15 & ((uint64_t) keys)) && i != size)
		unaligned_keys[i++] = *keys++;
	uint64_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint64_t *keys_end = &keys[size - i];
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k12 = _mm_load_si128((__m128i*) &keys[0]);
		k34 = _mm_load_si128((__m128i*) &keys[2]);
		keys += 4; i = 4;
		unaligned_intro:;
		__m128i h12 = _mm_srl_epi64(k12, s);
		__m128i h34 = _mm_srl_epi64(k34, s);
		h12 = _mm_and_si128(h12, m);
		h34 = _mm_and_si128(h34, m);
		__m128i h = _mm_packus_epi32(h12, h34);
		do {
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			count[p]++;
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end)
		unaligned_keys[i++] = *keys++;
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
}

void partition_offsets(uint64_t **count, int partitions, int id,
		       int threads, uint64_t *offsets)
{	int i, t;
	uint64_t p = 0;
	for (i = 0 ; i != partitions ; ++i) {
		for (t = 0 ; t != id ; ++t)
			p += count[t][i];
		offsets[i] = p;
		for (; t != threads ; ++t)
			p += count[t][i];
	}
}

// Function to prefetch a memory address
static inline void prefetch(const void *ptr) {
    _mm_prefetch(ptr, _MM_HINT_T0);
}


void partition_2_chunks(uint64_t* keys, uint64_t* rids, uint64_t size, uint64_t* offsets, uint64_t* sizes, uint64_t* buf, uint64_t* keys_out, uint64_t* rids_out, uint8_t shift_bits, uint8_t radix_bits) {
    assert((63 & (uint64_t)keys_out) == 0);
    assert((63 & (uint64_t)rids_out) == 0);
    assert(radix_bits <= 16);

    int i, partitions = 1 << radix_bits;

    // Initialize buffers
    for (i = 0; i != partitions; ++i) {
        buf[(i << 4) | 14] = offsets[i] << 1;
    }

    __m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
    __m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
    __m128i k12, k34, v12, v34;

    uint64_t unaligned_keys[4];
    uint64_t unaligned_rids[4];
    uint64_t p = 0;
    i = 0;

    // Process unaligned keys and rids at the beginning
    while ((15 & ((uint64_t)keys)) && i != size) {
        unaligned_keys[i] = *keys++;
        unaligned_rids[i++] = *rids++;
    }
    assert(i == size || (15 & (uint64_t)rids) == 0);

    uint32_t* keys_32 = (uint32_t*)keys_out;
    uint32_t* rids_32 = (uint32_t*)rids_out;
    uint64_t* keys_aligned_end = &keys[(size - i) & ~3];
    uint64_t* keys_end = &keys[size - i];

    if (i) {
        k12 = _mm_loadu_si128((__m128i*)&unaligned_keys[0]);
        k34 = _mm_loadu_si128((__m128i*)&unaligned_keys[2]);
        v12 = _mm_loadu_si128((__m128i*)&unaligned_rids[0]);
        v34 = _mm_loadu_si128((__m128i*)&unaligned_rids[2]);
        goto unaligned_intro;
    }

    uint64_t chunk_size = (keys_aligned_end - keys) / 2;

    // First chunk
    uint64_t* chunk_end = keys + chunk_size;
    while (keys != chunk_end) {
        prefetch(&keys[32]);
        prefetch(&rids[32]);

        k12 = _mm_load_si128((__m128i*)&keys[0]);
        k34 = _mm_load_si128((__m128i*)&keys[2]);
        v12 = _mm_load_si128((__m128i*)&rids[0]);
        v34 = _mm_load_si128((__m128i*)&rids[2]);
        keys += 4;
        rids += 4;
        i = 4;

    unaligned_intro:;
        __m128i h12 = _mm_srl_epi64(k12, s);
        __m128i h34 = _mm_srl_epi64(k34, s);
        h12 = _mm_and_si128(h12, m);
        h34 = _mm_and_si128(h34, m);
        __m128i h = _mm_packus_epi32(h12, h34);
        h = _mm_slli_epi32(h, 4);

        k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
        k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
        v12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
        v34 = _mm_shuffle_epi32(v34, _MM_SHUFFLE(3, 1, 2, 0));

        __m128i k_L = _mm_unpacklo_epi64(k12, k34);
        __m128i k_H = _mm_unpackhi_epi64(k12, k34);
        __m128i v_L = _mm_unpacklo_epi64(v12, v34);
        __m128i v_H = _mm_unpackhi_epi64(v12, v34);

        for (int j = 0; j < 4; j++) {
            asm("movd %1, %%eax" : "=a"(p) : "x"(h), "0"(p));
            uint64_t* src = &buf[p];
            uint64_t index = src[14];
            src[14] = index + 2;
            uint64_t offset = index & 15;

            __m128i kkxx = _mm_unpacklo_epi32(k_L, k_H);
            __m128i vvxx = _mm_unpacklo_epi32(v_L, v_H);
            __m128i kkvv = _mm_unpacklo_epi64(kkxx, vvxx);
            _mm_store_si128((__m128i*)&src[offset], kkvv);

            if (offset == 14) {
                uint32_t* dest_x = &keys_32[index - 14];
                uint32_t* dest_y = &rids_32[index - 14];

                __m128i r0 = _mm_load_si128((__m128i*)&src[0]);
                __m128i r1 = _mm_load_si128((__m128i*)&src[2]);
                __m128i r2 = _mm_load_si128((__m128i*)&src[4]);
                __m128i r3 = _mm_load_si128((__m128i*)&src[6]);
                __m128i r4 = _mm_load_si128((__m128i*)&src[8]);
                __m128i r5 = _mm_load_si128((__m128i*)&src[10]);
                __m128i r6 = _mm_load_si128((__m128i*)&src[12]);
                __m128i r7 = _mm_load_si128((__m128i*)&src[14]);

                __m128i x0 = _mm_unpacklo_epi64(r0, r1);
                __m128i x1 = _mm_unpacklo_epi64(r2, r3);
                __m128i x2 = _mm_unpacklo_epi64(r4, r5);
                __m128i x3 = _mm_unpacklo_epi64(r6, r7);

                _mm_stream_si128((__m128i*)&dest_x[0], x0);
                _mm_stream_si128((__m128i*)&dest_x[4], x1);
                _mm_stream_si128((__m128i*)&dest_x[8], x2);
                _mm_stream_si128((__m128i*)&dest_x[12], x3);

                __m128i y0 = _mm_unpackhi_epi64(r0, r1);
                __m128i y1 = _mm_unpackhi_epi64(r2, r3);
                __m128i y2 = _mm_unpackhi_epi64(r4, r5);
                __m128i y3 = _mm_unpackhi_epi64(r6, r7);

                _mm_stream_si128((__m128i*)&dest_y[0], y0);
                _mm_stream_si128((__m128i*)&dest_y[4], y1);
                _mm_stream_si128((__m128i*)&dest_y[8], y2);
                _mm_stream_si128((__m128i*)&dest_y[12], y3);

                src[14] = index + 2;
            }

            h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
            k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
            k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
            v_L = _mm_shuffle_epi32(v_L, _MM_SHUFFLE(0, 3, 2, 1));
            v_H = _mm_shuffle_epi32(v_H, _MM_SHUFFLE(0, 3, 2, 1));
        }
    }

    // Second chunk
    while (keys != keys_aligned_end) {
        prefetch(&keys[32]);
        prefetch(&rids[32]);

        k12 = _mm_load_si128((__m128i*)&keys[0]);
        k34 = _mm_load_si128((__m128i*)&keys[2]);
        v12 = _mm_load_si128((__m128i*)&rids[0]);
        v34 = _mm_load_si128((__m128i*)&rids[2]);
        keys += 4;
        rids += 4;
        i = 4;

    unaligned_intro_second:;
        __m128i h12 = _mm_srl_epi64(k12, s);
        __m128i h34 = _mm_srl_epi64(k34, s);
        h12 = _mm_and_si128(h12, m);
        h34 = _mm_and_si128(h34, m);
        __m128i h = _mm_packus_epi32(h12, h34);
        h = _mm_slli_epi32(h, 4);

        k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
        k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
        v12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
        v34 = _mm_shuffle_epi32(v34, _MM_SHUFFLE(3, 1, 2, 0));

        __m128i k_L = _mm_unpacklo_epi64(k12, k34);
        __m128i k_H = _mm_unpackhi_epi64(k12, k34);
        __m128i v_L = _mm_unpacklo_epi64(v12, v34);
        __m128i v_H = _mm_unpackhi_epi64(v12, v34);

        for (int j = 0; j < 4; j++) {
            asm("movd %1, %%eax" : "=a"(p) : "x"(h), "0"(p));
            uint64_t* src = &buf[p];
            uint64_t index = src[14];
            src[14] = index + 2;
            uint64_t offset = index & 15;

            __m128i kkxx = _mm_unpacklo_epi32(k_L, k_H);
            __m128i vvxx = _mm_unpacklo_epi32(v_L, v_H);
            __m128i kkvv = _mm_unpacklo_epi64(kkxx, vvxx);
            _mm_store_si128((__m128i*)&src[offset], kkvv);

            if (offset == 14) {
                uint32_t* dest_x = &keys_32[index - 14];
                uint32_t* dest_y = &rids_32[index - 14];

                __m128i r0 = _mm_load_si128((__m128i*)&src[0]);
                __m128i r1 = _mm_load_si128((__m128i*)&src[2]);
                __m128i r2 = _mm_load_si128((__m128i*)&src[4]);
                __m128i r3 = _mm_load_si128((__m128i*)&src[6]);
                __m128i r4 = _mm_load_si128((__m128i*)&src[8]);
                __m128i r5 = _mm_load_si128((__m128i*)&src[10]);
                __m128i r6 = _mm_load_si128((__m128i*)&src[12]);
                __m128i r7 = _mm_load_si128((__m128i*)&src[14]);

                __m128i x0 = _mm_unpacklo_epi64(r0, r1);
                __m128i x1 = _mm_unpacklo_epi64(r2, r3);
                __m128i x2 = _mm_unpacklo_epi64(r4, r5);
                __m128i x3 = _mm_unpacklo_epi64(r6, r7);

                _mm_stream_si128((__m128i*)&dest_x[0], x0);
                _mm_stream_si128((__m128i*)&dest_x[4], x1);
                _mm_stream_si128((__m128i*)&dest_x[8], x2);
                _mm_stream_si128((__m128i*)&dest_x[12], x3);

                __m128i y0 = _mm_unpackhi_epi64(r0, r1);
                __m128i y1 = _mm_unpackhi_epi64(r2, r3);
                __m128i y2 = _mm_unpackhi_epi64(r4, r5);
                __m128i y3 = _mm_unpackhi_epi64(r6, r7);

                _mm_stream_si128((__m128i*)&dest_y[0], y0);
                _mm_stream_si128((__m128i*)&dest_y[4], y1);
                _mm_stream_si128((__m128i*)&dest_y[8], y2);
                _mm_stream_si128((__m128i*)&dest_y[12], y3);

                src[14] = index + 2;
            }

            h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
            k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
            k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
            v_L = _mm_shuffle_epi32(v_L, _MM_SHUFFLE(0, 3, 2, 1));
            v_H = _mm_shuffle_epi32(v_H, _MM_SHUFFLE(0, 3, 2, 1));
        }
    }

    // Histogram last 0-3 unaligned items
    i = 0;
    p = 0;
    while (keys != keys_end) {
        unaligned_keys[i] = *keys++;
        unaligned_rids[i++] = *rids++;
    }
    if (i) {
        k12 = _mm_loadu_si128((__m128i*)&unaligned_keys[0]);
        k34 = _mm_loadu_si128((__m128i*)&unaligned_keys[2]);
        v12 = _mm_loadu_si128((__m128i*)&unaligned_rids[0]);
        v34 = _mm_loadu_si128((__m128i*)&unaligned_rids[2]);
        keys_aligned_end = keys_end;
        goto unaligned_intro_second;
    }

#ifdef BG
    for (i = 0; i != partitions; ++i) {
        uint64_t index = buf[(i << 4) | 14] >> 1;
        assert(index - offsets[i] == sizes[i]);
    }
#endif
}

void partition_bultin_expect(uint64_t* keys, uint64_t* rids, uint64_t size, uint64_t* offsets, uint64_t* sizes, uint64_t* buf, uint64_t* keys_out, uint64_t* rids_out, uint8_t shift_bits, uint8_t radix_bits) {
    assert((63 & (uint64_t)keys_out) == 0);
    assert((63 & (uint64_t)rids_out) == 0);
    assert(radix_bits <= 16);

    int i, partitions = 1 << radix_bits;

    // Initialize buffers
    for (i = 0; i != partitions; ++i) {
        buf[(i << 4) | 14] = offsets[i] << 1;
    }

    // Main partition loop
    __m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
    __m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
    __m128i k12, k34, v12, v34;

    // Check for unaligned items
    uint64_t unaligned_keys[4];
    uint64_t unaligned_rids[4];
    uint64_t p = 0;
    i = 0;
    while (__builtin_expect((15 & ((uint64_t)keys)) && i != size, 0)) {
        unaligned_keys[i] = *keys++;
        unaligned_rids[i++] = *rids++;
    }
    assert(i == size || (15 & (uint64_t)rids) == 0);

    uint32_t* keys_32 = (uint32_t*)keys_out;
    uint32_t* rids_32 = (uint32_t*)rids_out;
    uint64_t* keys_aligned_end = &keys[(size - i) & ~3];
    uint64_t* keys_end = &keys[size - i];

    if (__builtin_expect(i != 0, 0)) {
        k12 = _mm_loadu_si128((__m128i*)&unaligned_keys[0]);
        k34 = _mm_loadu_si128((__m128i*)&unaligned_keys[2]);
        v12 = _mm_loadu_si128((__m128i*)&unaligned_rids[0]);
        v34 = _mm_loadu_si128((__m128i*)&unaligned_rids[2]);
        goto unaligned_intro;
    }

    while (__builtin_expect(keys != keys_aligned_end, 1)) {
        _mm_prefetch((const char*)&keys[128], _MM_HINT_T0);
        _mm_prefetch((const char*)&rids[128], _MM_HINT_T0);

        k12 = _mm_load_si128((__m128i*)&keys[0]);
        k34 = _mm_load_si128((__m128i*)&keys[2]);
        v12 = _mm_load_si128((__m128i*)&rids[0]);
        v34 = _mm_load_si128((__m128i*)&rids[2]);
        keys += 4;
        rids += 4;
        i = 4;

    unaligned_intro:;
        __m128i h12 = _mm_srl_epi64(k12, s);
        __m128i h34 = _mm_srl_epi64(k34, s);
        h12 = _mm_and_si128(h12, m);
        h34 = _mm_and_si128(h34, m);
        __m128i h = _mm_packus_epi32(h12, h34);
        h = _mm_slli_epi32(h, 4);

        k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
        k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
        v12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
        v34 = _mm_shuffle_epi32(v34, _MM_SHUFFLE(3, 1, 2, 0));

        __m128i k_L = _mm_unpacklo_epi64(k12, k34);
        __m128i k_H = _mm_unpackhi_epi64(k12, k34);
        __m128i v_L = _mm_unpacklo_epi64(v12, v34);
        __m128i v_H = _mm_unpackhi_epi64(v12, v34);

        for (int j = 0; j < 4; j++) {
            // Extract partition
            asm("movd %1, %%eax" : "=a"(p) : "x"(h), "0"(p));
            // Offset in the cache line pair
            uint64_t* src = &buf[p];
            uint64_t index = src[14];
            src[14] = index + 2;
            uint64_t offset = index & 15;

            // Pack and store
            __m128i kkxx = _mm_unpacklo_epi32(k_L, k_H);
            __m128i vvxx = _mm_unpacklo_epi32(v_L, v_H);
            __m128i kkvv = _mm_unpacklo_epi64(kkxx, vvxx);
            _mm_store_si128((__m128i*)&src[offset], kkvv);

            if (__builtin_expect(offset == 14, 0)) {
                uint32_t* dest_x = &keys_32[index - 14];
                uint32_t* dest_y = &rids_32[index - 14];

                // Load cache line from cache to 8 128-bit registers
                __m128i r0 = _mm_load_si128((__m128i*)&src[0]);
                __m128i r1 = _mm_load_si128((__m128i*)&src[2]);
                __m128i r2 = _mm_load_si128((__m128i*)&src[4]);
                __m128i r3 = _mm_load_si128((__m128i*)&src[6]);
                __m128i r4 = _mm_load_si128((__m128i*)&src[8]);
                __m128i r5 = _mm_load_si128((__m128i*)&src[10]);
                __m128i r6 = _mm_load_si128((__m128i*)&src[12]);
                __m128i r7 = _mm_load_si128((__m128i*)&src[14]);

                // Split first column
                __m128i x0 = _mm_unpacklo_epi64(r0, r1);
                __m128i x1 = _mm_unpacklo_epi64(r2, r3);
                __m128i x2 = _mm_unpacklo_epi64(r4, r5);
                __m128i x3 = _mm_unpacklo_epi64(r6, r7);

                // Stream first column
                _mm_stream_si128((__m128i*)&dest_x[0], x0);
                _mm_stream_si128((__m128i*)&dest_x[4], x1);
                _mm_stream_si128((__m128i*)&dest_x[8], x2);
                _mm_stream_si128((__m128i*)&dest_x[12], x3);

                // Split second column
                __m128i y0 = _mm_unpackhi_epi64(r0, r1);
                __m128i y1 = _mm_unpackhi_epi64(r2, r3);
                __m128i y2 = _mm_unpackhi_epi64(r4, r5);
                __m128i y3 = _mm_unpackhi_epi64(r6, r7);

                // Stream second column
                _mm_stream_si128((__m128i*)&dest_y[0], y0);
                _mm_stream_si128((__m128i*)&dest_y[4], y1);
                _mm_stream_si128((__m128i*)&dest_y[8], y2);
                _mm_stream_si128((__m128i*)&dest_y[12], y3);

                // Restore overwritten pointer
                src[14] = index + 2;
            }

            // Rotate
            h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
            k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
            k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
            v_L = _mm_shuffle_epi32(v_L, _MM_SHUFFLE(0, 3, 2, 1));
            v_H = _mm_shuffle_epi32(v_H, _MM_SHUFFLE(0, 3, 2, 1));
        }
    }

    // Histogram last 0-3 unaligned items
    i = 0;
    p = 0;
    while (__builtin_expect(keys != keys_end, 0)) {
        unaligned_keys[i] = *keys++;
        unaligned_rids[i++] = *rids++;
    }
    if (__builtin_expect(i != 0, 0)) {
        k12 = _mm_loadu_si128((__m128i*)&unaligned_keys[0]);
        k34 = _mm_loadu_si128((__m128i*)&unaligned_keys[2]);
        v12 = _mm_loadu_si128((__m128i*)&unaligned_rids[0]);
        v34 = _mm_loadu_si128((__m128i*)&unaligned_rids[2]);
        keys_aligned_end = keys_end;
        goto unaligned_intro;
    }

#ifdef BG
    // Check partition sanity
    for (i = 0; i != partitions; ++i) {
        uint64_t index = buf[(i << 4) | 14] >> 1;
        assert(index - offsets[i] == sizes[i]);
    }
#endif
}

void partition_wow(uint64_t* keys, uint64_t* rids, uint64_t size, uint64_t* offsets, uint64_t* sizes, uint64_t* buf, uint64_t* keys_out, uint64_t* rids_out, uint8_t shift_bits, uint8_t radix_bits) {
    assert((63 & (uint64_t)keys_out) == 0);
    assert((63 & (uint64_t)rids_out) == 0);
    assert(radix_bits <= 16);

    int i, partitions = 1 << radix_bits;

    // Initialize buffers
    for (i = 0; i != partitions; ++i) {
        buf[(i << 4) | 14] = offsets[i] << 1;
    }

    // Main partition loop
    __m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
    __m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
    __m128i k12, k34, v12, v34;

    // Check for unaligned items
    uint64_t unaligned_keys[4];
    uint64_t unaligned_rids[4];
    uint64_t p = 0;
    i = 0;
    while ((15 & ((uint64_t)keys)) && i != size) {
        unaligned_keys[i] = *keys++;
        unaligned_rids[i++] = *rids++;
    }
    assert(i == size || (15 & (uint64_t)rids) == 0);

    uint32_t* keys_32 = (uint32_t*)keys_out;
    uint32_t* rids_32 = (uint32_t*)rids_out;
    uint64_t* keys_aligned_end = &keys[(size - i) & ~3];
    uint64_t* keys_end = &keys[size - i];

    if (i) {
        k12 = _mm_loadu_si128((__m128i*)&unaligned_keys[0]);
        k34 = _mm_loadu_si128((__m128i*)&unaligned_keys[2]);
        v12 = _mm_loadu_si128((__m128i*)&unaligned_rids[0]);
        v34 = _mm_loadu_si128((__m128i*)&unaligned_rids[2]);
        goto unaligned_intro;
    }

    while (keys != keys_aligned_end) {
        _mm_prefetch((const char*)&keys[128], _MM_HINT_T0);
        _mm_prefetch((const char*)&rids[128], _MM_HINT_T0);

        k12 = _mm_load_si128((__m128i*)&keys[0]);
        k34 = _mm_load_si128((__m128i*)&keys[2]);
        v12 = _mm_load_si128((__m128i*)&rids[0]);
        v34 = _mm_load_si128((__m128i*)&rids[2]);
        keys += 4;
        rids += 4;
        i = 4;

    unaligned_intro:;
        __m128i h12 = _mm_srl_epi64(k12, s);
        __m128i h34 = _mm_srl_epi64(k34, s);
        h12 = _mm_and_si128(h12, m);
        h34 = _mm_and_si128(h34, m);
        __m128i h = _mm_packus_epi32(h12, h34);
        h = _mm_slli_epi32(h, 4);

        k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
        k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
        v12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
        v34 = _mm_shuffle_epi32(v34, _MM_SHUFFLE(3, 1, 2, 0));

        __m128i k_L = _mm_unpacklo_epi64(k12, k34);
        __m128i k_H = _mm_unpackhi_epi64(k12, k34);
        __m128i v_L = _mm_unpacklo_epi64(v12, v34);
        __m128i v_H = _mm_unpackhi_epi64(v12, v34);

        for (int j = 0; j < 4; j++) {
            // Extract partition
            asm("movd %1, %%eax" : "=a"(p) : "x"(h), "0"(p));
            // Offset in the cache line pair
            uint64_t* src = &buf[p];
            uint64_t index = src[14];
            src[14] = index + 2;
            uint64_t offset = index & 15;

            // Pack and store
            __m128i kkxx = _mm_unpacklo_epi32(k_L, k_H);
            __m128i vvxx = _mm_unpacklo_epi32(v_L, v_H);
            __m128i kkvv = _mm_unpacklo_epi64(kkxx, vvxx);
            _mm_store_si128((__m128i*)&src[offset], kkvv);

            if (offset == 14) {
                uint32_t* dest_x = &keys_32[index - 14];
                uint32_t* dest_y = &rids_32[index - 14];

                // Load cache line from cache to 8 128-bit registers
                __m128i r0 = _mm_load_si128((__m128i*)&src[0]);
                __m128i r1 = _mm_load_si128((__m128i*)&src[2]);
                __m128i r2 = _mm_load_si128((__m128i*)&src[4]);
                __m128i r3 = _mm_load_si128((__m128i*)&src[6]);
                __m128i r4 = _mm_load_si128((__m128i*)&src[8]);
                __m128i r5 = _mm_load_si128((__m128i*)&src[10]);
                __m128i r6 = _mm_load_si128((__m128i*)&src[12]);
                __m128i r7 = _mm_load_si128((__m128i*)&src[14]);

                // Split first column
                __m128i x0 = _mm_unpacklo_epi64(r0, r1);
                __m128i x1 = _mm_unpacklo_epi64(r2, r3);
                __m128i x2 = _mm_unpacklo_epi64(r4, r5);
                __m128i x3 = _mm_unpacklo_epi64(r6, r7);

                // Stream first column
                _mm_stream_si128((__m128i*)&dest_x[0], x0);
                _mm_stream_si128((__m128i*)&dest_x[4], x1);
                _mm_stream_si128((__m128i*)&dest_x[8], x2);
                _mm_stream_si128((__m128i*)&dest_x[12], x3);

                // Split second column
                __m128i y0 = _mm_unpackhi_epi64(r0, r1);
                __m128i y1 = _mm_unpackhi_epi64(r2, r3);
                __m128i y2 = _mm_unpackhi_epi64(r4, r5);
                __m128i y3 = _mm_unpackhi_epi64(r6, r7);

                // Stream second column
                _mm_stream_si128((__m128i*)&dest_y[0], y0);
                _mm_stream_si128((__m128i*)&dest_y[4], y1);
                _mm_stream_si128((__m128i*)&dest_y[8], y2);
                _mm_stream_si128((__m128i*)&dest_y[12], y3);

                // Restore overwritten pointer
                src[14] = index + 2;
            }

            // Rotate
            h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
            k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
            k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
            v_L = _mm_shuffle_epi32(v_L, _MM_SHUFFLE(0, 3, 2, 1));
            v_H = _mm_shuffle_epi32(v_H, _MM_SHUFFLE(0, 3, 2, 1));
        }
    }

    // Histogram last 0-3 unaligned items
    i = 0;
    p = 0;
    while (keys != keys_end) {
        unaligned_keys[i] = *keys++;
        unaligned_rids[i++] = *rids++;
    }
    if (i) {
        k12 = _mm_loadu_si128((__m128i*)&unaligned_keys[0]);
        k34 = _mm_loadu_si128((__m128i*)&unaligned_keys[2]);
        v12 = _mm_loadu_si128((__m128i*)&unaligned_rids[0]);
        v34 = _mm_loadu_si128((__m128i*)&unaligned_rids[2]);
        keys_aligned_end = keys_end;
        goto unaligned_intro;
    }

#ifdef BG
    // Check partition sanity
    for (i = 0; i != partitions; ++i) {
        uint64_t index = buf[(i << 4) | 14] >> 1;
        assert(index - offsets[i] == sizes[i]);
    }
#endif
}

void partition(uint64_t *keys, uint64_t *rids, uint64_t size,
	       uint64_t *offsets, uint64_t *sizes, uint64_t *buf,
	       uint64_t *keys_out, uint64_t *rids_out,
	       uint8_t shift_bits, uint8_t radix_bits)
{
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	assert(radix_bits <= 16);
	int i, partitions = 1 << radix_bits;
	// initialize buffers
	for (i = 0 ; i != partitions ; ++i)
		buf[(i << 4) | 14] = offsets[i] << 1;
	// main partition loop
	__m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
	__m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
	__m128i k12, k34, v12, v34;
	// check for unaligned items
	uint64_t unaligned_keys[4];
	uint64_t unaligned_rids[4];
	uint64_t p = 0; i = 0;
	while ((15 & ((uint64_t) keys)) && i != size) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	assert(i == size || (15 & (uint64_t) rids) == 0);
	uint32_t *keys_32 = (uint32_t*) keys_out;
	uint32_t *rids_32 = (uint32_t*) rids_out;
	uint64_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint64_t *keys_end = &keys[size - i];
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		v12 = _mm_loadu_si128((__m128i*) &unaligned_rids[0]);
		v34 = _mm_loadu_si128((__m128i*) &unaligned_rids[2]);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k12 = _mm_load_si128((__m128i*) &keys[0]);
		k34 = _mm_load_si128((__m128i*) &keys[2]);
		v12 = _mm_load_si128((__m128i*) &rids[0]);
		v34 = _mm_load_si128((__m128i*) &rids[2]);
		keys += 4; rids += 4; i = 4;
		unaligned_intro:;
		__m128i h12 = _mm_srl_epi64(k12, s);
		__m128i h34 = _mm_srl_epi64(k34, s);
		h12 = _mm_and_si128(h12, m);
		h34 = _mm_and_si128(h34, m);
		__m128i h = _mm_packus_epi32(h12, h34);
		h = _mm_slli_epi32(h, 4);
		k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
		k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
		v12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
		v34 = _mm_shuffle_epi32(v34, _MM_SHUFFLE(3, 1, 2, 0));
		__m128i k_L = _mm_unpacklo_epi64(k12, k34);
		__m128i k_H = _mm_unpackhi_epi64(k12, k34);
		__m128i v_L = _mm_unpacklo_epi64(v12, v34);
		__m128i v_H = _mm_unpackhi_epi64(v12, v34);
		do {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			// offset in the cache line pair
			uint64_t *src = &buf[p];
			uint64_t index = src[14];
			src[14] = index + 2;
			uint64_t offset = index & 15;
			// pack and store
			__m128i kkxx = _mm_unpacklo_epi32(k_L, k_H);
			__m128i vvxx = _mm_unpacklo_epi32(v_L, v_H);
			__m128i kkvv = _mm_unpacklo_epi64(kkxx, vvxx);
			_mm_store_si128((__m128i*) &src[offset], kkvv);
			if (offset == 14) {
				uint32_t *dest_x = &keys_32[index - 14];
				uint32_t *dest_y = &rids_32[index - 14];
				// load cache line from cache to 8 128-bit registers
				__m128i r0 = _mm_load_si128((__m128i*) &src[0]);
				__m128i r1 = _mm_load_si128((__m128i*) &src[2]);
				__m128i r2 = _mm_load_si128((__m128i*) &src[4]);
				__m128i r3 = _mm_load_si128((__m128i*) &src[6]);
				__m128i r4 = _mm_load_si128((__m128i*) &src[8]);
				__m128i r5 = _mm_load_si128((__m128i*) &src[10]);
				__m128i r6 = _mm_load_si128((__m128i*) &src[12]);
				__m128i r7 = _mm_load_si128((__m128i*) &src[14]);
				// split first column
				__m128i x0 = _mm_unpacklo_epi64(r0, r1);
				__m128i x1 = _mm_unpacklo_epi64(r2, r3);
				__m128i x2 = _mm_unpacklo_epi64(r4, r5);
				__m128i x3 = _mm_unpacklo_epi64(r6, r7);
				// stream first column
				_mm_stream_si128((__m128i*) &dest_x[0], x0);
				_mm_stream_si128((__m128i*) &dest_x[4], x1);
				_mm_stream_si128((__m128i*) &dest_x[8], x2);
				_mm_stream_si128((__m128i*) &dest_x[12],x3);
				// split second column
				__m128i y0 = _mm_unpackhi_epi64(r0, r1);
				__m128i y1 = _mm_unpackhi_epi64(r2, r3);
				__m128i y2 = _mm_unpackhi_epi64(r4, r5);
				__m128i y3 = _mm_unpackhi_epi64(r6, r7);
				// stream second column
				_mm_stream_si128((__m128i*) &dest_y[0], y0);
				_mm_stream_si128((__m128i*) &dest_y[4], y1);
				_mm_stream_si128((__m128i*) &dest_y[8], y2);
				_mm_stream_si128((__m128i*) &dest_y[12],y3);
				// restore overwritten pointer
				src[14] = index + 2;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
			k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
			v_L = _mm_shuffle_epi32(v_L, _MM_SHUFFLE(0, 3, 2, 1));
			v_H = _mm_shuffle_epi32(v_H, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	if (i) {
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		v12 = _mm_loadu_si128((__m128i*) &unaligned_rids[0]);
		v34 = _mm_loadu_si128((__m128i*) &unaligned_rids[2]);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
#ifdef BG
	// check partition sanity
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t index = buf[(i << 4) | 14] >> 1;
		assert(index - offsets[i] == sizes[i]);
	}
#endif
}

void partition_2(uint64_t* keys, uint64_t* rids, uint64_t size, uint64_t* offsets, uint64_t* sizes, uint64_t* buf, uint64_t* keys_out, uint64_t* rids_out, uint8_t shift_bits, uint8_t radix_bits) {
    assert((63 & (uint64_t)keys_out) == 0);
    assert((63 & (uint64_t)rids_out) == 0);
    assert(radix_bits <= 16);

    int i, partitions = 1 << radix_bits;

    // Initialize buffers
    for (i = 0; i != partitions; ++i) {
        buf[(i << 4) | 14] = offsets[i] << 1;
    }

    // Main partition loop
    __m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
    __m128i m = _mm_set1_epi64x((1 << radix_bits) - 1);
    __m128i k12, k34, v12, v34;

    // Check for unaligned items
    uint64_t unaligned_keys[4];
    uint64_t unaligned_rids[4];
    uint64_t p = 0;
    i = 0;
    while ((15 & ((uint64_t)keys)) && i != size) {
        unaligned_keys[i] = *keys++;
        unaligned_rids[i++] = *rids++;
    }
    assert(i == size || (15 & (uint64_t)rids) == 0);

    uint32_t* keys_32 = (uint32_t*)keys_out;
    uint32_t* rids_32 = (uint32_t*)rids_out;
    uint64_t* keys_aligned_end = &keys[(size - i) & ~3];
    uint64_t* keys_end = &keys[size - i];

    if (i) {
        k12 = _mm_loadu_si128((__m128i*)&unaligned_keys[0]);
        k34 = _mm_loadu_si128((__m128i*)&unaligned_keys[2]);
        v12 = _mm_loadu_si128((__m128i*)&unaligned_rids[0]);
        v34 = _mm_loadu_si128((__m128i*)&unaligned_rids[2]);
        goto unaligned_intro;
    }

    while (keys != keys_aligned_end) {

		prefetch(&keys[32]);
        prefetch(&rids[32]);

        k12 = _mm_load_si128((__m128i*)&keys[0]);
        k34 = _mm_load_si128((__m128i*)&keys[2]);
        v12 = _mm_load_si128((__m128i*)&rids[0]);
        v34 = _mm_load_si128((__m128i*)&rids[2]);
        keys += 4;
        rids += 4;
        i = 4;

    unaligned_intro:;
        __m128i h12 = _mm_srl_epi64(k12, s);
        __m128i h34 = _mm_srl_epi64(k34, s);
        h12 = _mm_and_si128(h12, m);
        h34 = _mm_and_si128(h34, m);
        __m128i h = _mm_packus_epi32(h12, h34);
        h = _mm_slli_epi32(h, 4);

        k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
        k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
        v12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
        v34 = _mm_shuffle_epi32(v34, _MM_SHUFFLE(3, 1, 2, 0));

        __m128i k_L = _mm_unpacklo_epi64(k12, k34);
        __m128i k_H = _mm_unpackhi_epi64(k12, k34);
        __m128i v_L = _mm_unpacklo_epi64(v12, v34);
        __m128i v_H = _mm_unpackhi_epi64(v12, v34);

        for (int j = 0; j < 4; j++) {
            // Extract partition
            asm("movd %1, %%eax" : "=a"(p) : "x"(h), "0"(p));
            // Offset in the cache line pair
            uint64_t* src = &buf[p];
            uint64_t index = src[14];
            src[14] = index + 2;
            uint64_t offset = index & 15;

            // Pack and store
            __m128i kkxx = _mm_unpacklo_epi32(k_L, k_H);
            __m128i vvxx = _mm_unpacklo_epi32(v_L, v_H);
            __m128i kkvv = _mm_unpacklo_epi64(kkxx, vvxx);
            _mm_store_si128((__m128i*)&src[offset], kkvv);

            if (offset == 14) {
                uint32_t* dest_x = &keys_32[index - 14];
                uint32_t* dest_y = &rids_32[index - 14];

                // Load cache line from cache to 8 128-bit registers
                __m128i r0 = _mm_load_si128((__m128i*)&src[0]);
                __m128i r1 = _mm_load_si128((__m128i*)&src[2]);
                __m128i r2 = _mm_load_si128((__m128i*)&src[4]);
                __m128i r3 = _mm_load_si128((__m128i*)&src[6]);
                __m128i r4 = _mm_load_si128((__m128i*)&src[8]);
                __m128i r5 = _mm_load_si128((__m128i*)&src[10]);
                __m128i r6 = _mm_load_si128((__m128i*)&src[12]);
                __m128i r7 = _mm_load_si128((__m128i*)&src[14]);

                // Split first column
                __m128i x0 = _mm_unpacklo_epi64(r0, r1);
                __m128i x1 = _mm_unpacklo_epi64(r2, r3);
                __m128i x2 = _mm_unpacklo_epi64(r4, r5);
                __m128i x3 = _mm_unpacklo_epi64(r6, r7);

                // Stream first column
                _mm_stream_si128((__m128i*)&dest_x[0], x0);
                _mm_stream_si128((__m128i*)&dest_x[4], x1);
                _mm_stream_si128((__m128i*)&dest_x[8], x2);
                _mm_stream_si128((__m128i*)&dest_x[12], x3);

                // Split second column
                __m128i y0 = _mm_unpackhi_epi64(r0, r1);
                __m128i y1 = _mm_unpackhi_epi64(r2, r3);
                __m128i y2 = _mm_unpackhi_epi64(r4, r5);
                __m128i y3 = _mm_unpackhi_epi64(r6, r7);

                // Stream second column
                _mm_stream_si128((__m128i*)&dest_y[0], y0);
                _mm_stream_si128((__m128i*)&dest_y[4], y1);
                _mm_stream_si128((__m128i*)&dest_y[8], y2);
                _mm_stream_si128((__m128i*)&dest_y[12], y3);

                // Restore overwritten pointer
                src[14] = index + 2;
            }

            // Rotate
            h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
            k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
            k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
            v_L = _mm_shuffle_epi32(v_L, _MM_SHUFFLE(0, 3, 2, 1));
            v_H = _mm_shuffle_epi32(v_H, _MM_SHUFFLE(0, 3, 2, 1));
        }
    }

    // Histogram last 0-3 unaligned items
    i = 0;
    p = 0;
    while (keys != keys_end) {
        unaligned_keys[i] = *keys++;
        unaligned_rids[i++] = *rids++;
    }
    if (i) {
        k12 = _mm_loadu_si128((__m128i*)&unaligned_keys[0]);
        k34 = _mm_loadu_si128((__m128i*)&unaligned_keys[2]);
        v12 = _mm_loadu_si128((__m128i*)&unaligned_rids[0]);
        v34 = _mm_loadu_si128((__m128i*)&unaligned_rids[2]);
        keys_aligned_end = keys_end;
        goto unaligned_intro;
    }

#ifdef BG
    // Check partition sanity
    for (i = 0; i != partitions; ++i) {
        uint64_t index = buf[(i << 4) | 14] >> 1;
        assert(index - offsets[i] == sizes[i]);
    }
#endif
}

void finalize(uint64_t *sizes, uint64_t *buf,
	      uint64_t *keys_out, uint64_t *rids_out, int partitions)
{	int i;
	// find offset to align output
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	// flush remaining items from buffers to output
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t *src = &buf[i << 4];
		uint64_t index = src[14] >> 1;
		uint64_t rem = index & 7;
		uint64_t off = 0;
		if (rem > sizes[i])
			off = rem - sizes[i];
		index -= rem - off;
		while (off != rem) {
			keys_out[index] = src[off + off];
			rids_out[index] = src[off + off + 1];
			off++; index++;
		}
	}
}

void swap_ppl(uint64_t ***a, uint64_t ***b)
{
	uint64_t **t = *a; *a = *b; *b = t;
}

void extract_delimiters(uint64_t *sample, uint64_t sample_size, uint64_t *delimiter)
{
	uint64_t i, parts = 0;
	while (delimiter[parts] != ~ (uint64_t) 0) parts++;
	double percentile = sample_size * 1.0 / (parts + 1);
	for (i = 0 ; i != parts ; ++i) {
		uint64_t index = percentile * (i + 1) - 0.001;
		delimiter[i] = sample[index];
		// search repetitions in sample
		uint64_t start, end;
		for (start = index ; start ; --start)
			if (sample[start] != delimiter[i]) break;
		for (end = index ; end != sample_size ; ++end)
			if (sample[end] != delimiter[i]) break;
		// if more repetitions after, don't include
		if (index - start < end - index && delimiter[i])
			delimiter[i]--;
	}
}

typedef struct {
	int *bits;
	double fudge;
	uint64_t **keys;
	uint64_t **rids;
	uint64_t *size;
	uint64_t **keys_buf;
	uint64_t **rids_buf;
	uint64_t ***count;
	uint64_t **numa_local_count;
	uint64_t *sample;
	uint64_t *sample_buf;
	uint64_t **sample_hist;
	uint64_t sample_size;
	int *numa_node;
	int *cpu;
	int threads;
	int numa;
	int max_threads;
	int max_numa;
	int allocated;
	int interleaved;
	pthread_barrier_t *global_barrier;
	pthread_barrier_t **local_barrier;
	pthread_barrier_t *sample_barrier;
} global_data_t;

typedef struct {
	int id;
	uint32_t seed;
	uint64_t checksum;
	uint64_t alloc_time;
	uint64_t sample_time;
	uint64_t numa_shuffle_time;
	uint64_t hist_time[8];
	uint64_t part_time[8];
	global_data_t *global;
} thread_data_t;

int uint64_compare(const void *x, const void *y)
{
	uint64_t a = *((uint64_t*) x);
	uint64_t b = *((uint64_t*) y);
	return a < b ? -1 : (a > b ? 1 : 0);
}



void *sort_thread(void *arg)
{

	thread_data_t *a = (thread_data_t*) arg;
	global_data_t *d = a->global;
	int i, j, k, t, n, lb = 0, gb = 0, id = a->id;
	int numa = d->numa;
	int numa_node = d->numa_node[id];
	int numa_dst, numa_src;
	int threads = d->threads;
	int threads_per_numa = threads / numa;
	uint32_t seed = a->seed;
	pthread_barrier_t *local_barrier = d->local_barrier[numa_node];
	pthread_barrier_t *global_barrier = d->global_barrier;
	// id in local numa threads
	int numa_local_id = 0;
	for (i = 0 ; i != id ; ++i)
		if (d->numa_node[i] == numa_node)
			numa_local_id++;
	// compute total size
	uint64_t total_size = 0;
	for (n = 0 ; n != numa ; ++n)
		total_size += d->size[n];
	// bind thread and its allocation
	if (threads <= d->max_threads)
		cpu_bind(id);
		// cpu_bind(d->cpu[id]);
	// if (numa <= d->max_numa)
	// 	memory_bind(d->numa_node[id]);
	// size for histograms
	int radix_bits = d->bits[0];
	int partitions = (1 << radix_bits) * (numa == 3 ? 4 : numa);
	int max_partitions = partitions;
	for (i = 1 ; d->bits[i] != 0 ; ++i) {
		int parts = 1 << d->bits[i];
		if (parts > max_partitions)
			max_partitions = parts;
	}
	// initial histogram and buffers
	uint64_t *offsets = malloc(max_partitions * sizeof(uint64_t));
	uint64_t *count = calloc(max_partitions, sizeof(uint64_t));
	uint64_t *buf = mamalloc((max_partitions << 4) * sizeof(uint64_t));
	d->count[numa_node][numa_local_id] = count;
	uint64_t numa_size = d->size[numa_node];
	uint64_t size = numa_size / threads_per_numa;
	uint64_t offset = size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		size = numa_size - size * numa_local_id;
	uint64_t tim = micro_time();
	if (!d->allocated) {
		if (!numa_local_id) {
			uint64_t cap = d->size[numa_node] * d->fudge;
			if (d->interleaved) {
				d->keys_buf[numa_node] = numa_alloc_interleaved(cap * sizeof(uint64_t));
				d->rids_buf[numa_node] = numa_alloc_interleaved(cap * sizeof(uint64_t));
			} else {
				d->keys_buf[numa_node] = mamalloc(cap * sizeof(uint64_t));
				d->rids_buf[numa_node] = mamalloc(cap * sizeof(uint64_t));
			}
		}
		pthread_barrier_wait(&local_barrier[lb++]);
	}
	uint64_t *keys = &d->keys[numa_node][offset];
	uint64_t *rids = &d->rids[numa_node][offset];
	uint64_t *keys_end = NULL, *rids_end = NULL;
	if (!d->allocated) {
		uint64_t *keys_buf = &d->keys_buf[numa_node][offset];
		uint64_t *rids_buf = &d->rids_buf[numa_node][offset];
		uint64_t p;
		for (p = 0 ; p != size ; ++p)
			_mm_stream_si64((long long int*) &keys_buf[p], 0);
		for (p = 0 ; p != size ; ++p)
			_mm_stream_si64((long long int*) &rids_buf[p], 0);
		pthread_barrier_wait(&local_barrier[lb++]);
	}
	tim = micro_time() - tim;
	a->alloc_time = tim;
	// sampling
	tim = micro_time();
	uint64_t *delimiter = calloc(numa, sizeof(uint64_t));
	delimiter[numa - 1] = ~0;
	if (numa > 1) {
		assert((d->sample_size & 3) == 0);
		uint64_t p, sample_size = (d->sample_size / threads) & ~15;
		uint64_t *sample = &d->sample[sample_size * id];
		if (id + 1 == threads)
			sample_size = d->sample_size - sample_size * id;
		rand64_t *gen = rand64_init(a->seed);
		for (p = 0 ; p != sample_size ; ++p)
			sample[p] = keys[mulhi(rand64_next(gen), size)];
		// (in-parallel) LSB radix-sort the sample
		partition_keys(d->sample, d->sample_buf, d->sample_size, d->sample_hist, 0, 8,
		               id, threads, &global_barrier[gb]);
		partition_keys(d->sample_buf, d->sample, d->sample_size, d->sample_hist, 8, 8,
		               id, threads, &global_barrier[gb + 3]);
		partition_keys(d->sample, d->sample_buf, d->sample_size, d->sample_hist, 16, 8,
		               id, threads, &global_barrier[gb + 6]);
		partition_keys(d->sample_buf, d->sample, d->sample_size, d->sample_hist, 24, 8,
		               id, threads, &global_barrier[gb + 9]);
		partition_keys(d->sample, d->sample_buf, d->sample_size, d->sample_hist, 32, 8,
		               id, threads, &global_barrier[gb + 12]);
		partition_keys(d->sample_buf, d->sample, d->sample_size, d->sample_hist, 40, 8,
		               id, threads, &global_barrier[gb + 15]);
		partition_keys(d->sample, d->sample_buf, d->sample_size, d->sample_hist, 48, 8,
		               id, threads, &global_barrier[gb + 18]);
		partition_keys(d->sample_buf, d->sample, d->sample_size, d->sample_hist, 56, 8,
		               id, threads, &global_barrier[gb + 21]);
		gb += 24;
		extract_delimiters(d->sample, d->sample_size, delimiter);
	}
	tim = micro_time() - tim;
	a->sample_time = tim;
	// histogram
	tim = micro_time();
	if (numa == 1)
		histogram(keys, size, count, 0, radix_bits);
	else if (numa == 2)
		histogram_numa_2(keys, size, count, radix_bits, delimiter);
	else if (numa <= 4)
		histogram_numa_4(keys, size, count, radix_bits, delimiter);
	else if (numa <= 8)
		histogram_numa_8(keys, size, count, radix_bits, delimiter);
	tim = micro_time() - tim;
	a->hist_time[0] = tim;
	// local counts for numa transfer
	uint64_t *numa_local_count = NULL;
	if (numa > 1) {
		numa_local_count = calloc(numa, sizeof(uint64_t));
		for (i = 0 ; i != partitions ; ++i)
			numa_local_count[i >> radix_bits] += count[i];
	}
	d->numa_local_count[id] = numa_local_count;
	// local sync and partition
	pthread_barrier_wait(&local_barrier[lb++]);
	// offsets of output partitions
	tim = micro_time();
	uint64_t **counts = d->count[numa_node];
	partition_offsets(counts, partitions, numa_local_id,
			  threads_per_numa, offsets);
	// partition range partitioned data in local nodes
	uint64_t *keys_out = d->keys_buf[numa_node];
	uint64_t *rids_out = d->rids_buf[numa_node];
	if (numa == 1)
		partition(keys, rids, size, offsets, count, buf,
		          keys_out, rids_out, 0, radix_bits);
	else if (numa == 2)
		partition_numa_2(keys, rids, size, offsets, count, buf,
		                 keys_out, rids_out, radix_bits, delimiter);
	else if (numa <= 4)
		partition_numa_4(keys, rids, size, offsets, count, buf,
		                 keys_out, rids_out, radix_bits, delimiter);
	else if (numa <= 8)
		partition_numa_8(keys, rids, size, offsets, count, buf,
		                 keys_out, rids_out, radix_bits, delimiter);
	// local sync and finalize
	pthread_barrier_wait(&local_barrier[lb++]);
	finalize(count, buf, keys_out, rids_out, partitions);
	tim = micro_time() - tim;
	a->part_time[0] = tim;
	// synchronize globally
	pthread_barrier_wait(d->sample_barrier);
	a->numa_shuffle_time = 0;
	// copy remote partitions
	if (numa > 1) {
		// check numa part sizes
		uint64_t **transfer = malloc(numa * sizeof(uint64_t*));
		for (n = 0 ; n != numa ; ++n)
			transfer[n] = calloc(numa, sizeof(uint64_t));
		// compute sizes of numa transfers
		for (i = 0 ; i != threads ; ++i) {
			int j = d->numa_node[i];
			for (n = 0 ; n != numa ; ++n)
				transfer[j][n] += d->numa_local_count[i][n];
		}
		// compute local numa size after move
		uint64_t max_size = numa_size * d->fudge;
		numa_size = 0;
		for (n = 0 ; n != numa ; ++n)
			numa_size += transfer[n][numa_node];
		if (numa_size > max_size)
			fprintf(stderr, "NUMA %d is %.2f%% of input\n", numa_node,
					 numa_size * 100.0 / total_size);
		assert(numa_size <= max_size);
		tim = micro_time();
		// compute starting numa offsets
		uint64_t *numa_offset = calloc(numa, sizeof(uint64_t));
		uint64_t *numa_part = malloc(numa * sizeof(uint64_t));
		for (numa_src = 0 ; numa_src != numa ; ++numa_src)
			for (numa_dst = 0 ; numa_dst != numa_node ; ++numa_dst)
				numa_offset[numa_src] += transfer[numa_src][numa_dst];
		// order of copies
		uint64_t *order_size = malloc(numa * sizeof(uint64_t));
		int *order = malloc(numa * sizeof(int));
		for (n = 0 ; n != numa ; ++n)
			order[n] = n;
		// copy one partition at a time
		uint64_t output_partition_offset = 0;
		for (i = 0 ; i != (1 << radix_bits) ; ++i) {
			j = i | (numa_node << radix_bits);
			// compute size per numa and size for thread to move
			uint64_t thread_part = 0;
			uint64_t part_size = 0;
			for (n = 0 ; n != numa ; ++n) {
				numa_part[n] = 0;
				for (t = 0 ; t != threads_per_numa ; ++t)
					numa_part[n] += d->count[n][t][j];
				part_size += numa_part[n];
				thread_part += numa_part[n] / threads_per_numa;
			}
			// compute output offset
			uint64_t output_offset = output_partition_offset +
						 thread_part * numa_local_id;
			// shuffle numa order
			rand64_t *gen = rand64_init(seed);
			for (n = 0 ; n != numa ; ++n) {
				swap_i(&order[n], &order[rand64_next(gen) % (numa - n) + n]);
				// compute size
				uint64_t input_size = numa_part[n] / threads_per_numa;
				uint64_t input_offset = input_size * numa_local_id;
				if (numa_local_id + 1 == threads_per_numa)
					input_size = numa_part[n] - input_offset;
				order_size[n] = input_size + (!n ? 0 : order_size[n - 1]);
			}
			free(gen);
			// random numa order
			for (k = 0 ; k != numa ; ++k) {
				n = order[k];
				// compute input offset and size of copy
				uint64_t input_size = numa_part[n] / threads_per_numa;
				uint64_t input_offset = input_size * numa_local_id;
				if (numa_local_id + 1 == threads_per_numa)
					input_size = numa_part[n] - input_offset;
				input_offset += numa_offset[n];
				// re-ordered output offset
				uint64_t order_output_offset = output_offset +
							       order_size[n] - input_size;
				// copy keys and rids
				keys = &d->keys_buf[n][input_offset];
				rids = &d->rids_buf[n][input_offset];
				keys_end = &keys[input_size];
				rids_end = &rids[input_size];
				keys_out = &d->keys[numa_node][order_output_offset];
				rids_out = &d->rids[numa_node][order_output_offset];
				while (keys != keys_end)
					_mm_stream_si64((long long int*) keys_out++, *keys++);
				while (rids != rids_end)
					_mm_stream_si64((long long int*) rids_out++, *rids++);
			}
			output_offset += order_size[numa - 1];
			// update numa offsets and output offset
			for (n = 0 ; n != numa ; ++n)
				numa_offset[n] += numa_part[n];
			output_partition_offset += part_size;
		}
		for (n = 0 ; n != numa ; ++n)
			free(transfer[n]);
		free(order);
		free(order_size);
		free(transfer);
		free(numa_part);
		free(numa_offset);
		tim = micro_time() - tim;
		a->numa_shuffle_time = tim;
		// sync globally
		pthread_barrier_wait(&global_barrier[gb++]);
	}
	// input and outputs
	uint64_t **keys_a = numa > 1 ? d->keys : d->keys_buf;
	uint64_t **rids_a = numa > 1 ? d->rids : d->rids_buf;
	uint64_t **keys_b = numa > 1 ? d->keys_buf : d->keys;
	uint64_t **rids_b = numa > 1 ? d->rids_buf : d->rids;
	// local partitioning phases
	size = (numa_size / threads_per_numa) & ~3;
	offset = size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		size = numa_size - size * numa_local_id;
	counts = d->count[numa_node];
	count = d->count[numa_node][numa_local_id];
	int pass = 0;
	int shift_bits = 0;
	while (d->bits[++pass]) {
		// sync transfer phase
		if (pass != 1)
			pthread_barrier_wait(&local_barrier[lb++]);
		// start phase
		keys = &keys_a[numa_node][offset];
		rids = &rids_a[numa_node][offset];
		keys_out = keys_b[numa_node];
		rids_out = rids_b[numa_node];
		shift_bits += radix_bits;
		radix_bits = d->bits[pass];
		partitions = 1 << radix_bits;
		memset(count, 0, partitions * sizeof(uint64_t));
		// histogram
		tim = micro_time();
		histogram(keys, size, count, shift_bits, radix_bits);
		tim = micro_time() - tim;
		a->hist_time[pass] = tim;
		// sync histogram result
		pthread_barrier_wait(&local_barrier[lb++]);
		// compute offsets and partition
		tim = micro_time();
		partition_offsets(counts, partitions, numa_local_id,
				  threads_per_numa, offsets);
		partition(keys, rids, size, offsets, count, buf,
			  keys_out, rids_out, shift_bits, radix_bits);
		tim = micro_time() - tim;
		a->part_time[pass] = tim;
		// sync partitioning across threads
		pthread_barrier_wait(&local_barrier[lb++]);
		// finalize partitions
		finalize(count, buf, keys_out, rids_out, partitions);
		swap_ppl(&keys_a, &keys_b);
		swap_ppl(&rids_a, &rids_b);
	}
	// free(buf);
	// free(offsets);
	if (numa > 1 && !numa_local_id)
		d->size[numa_node] = numa_size;
	pthread_exit(NULL);
}

int sort(uint64_t **keys, uint64_t **rids, uint64_t *size, int threads,
         int numa, int bits, double fudge, uint64_t **keys_buf, uint64_t **rids_buf,
         char **description, uint64_t *times, int interleaved)
{
	int i, j, p, t, n, bits_space[8];
	int bit_passes = distribute_bits(bits, numa, bits_space, 0);
	int threads_per_numa = threads / numa;
	pthread_t *id = mamalloc(threads * sizeof(pthread_t));
	thread_data_t *data = mamalloc(threads * sizeof(thread_data_t));
	// check aligned input
	for (i = 0 ; i != numa ; ++i) {
		assert((15 & (uint64_t) keys[i]) == 0);
		assert((15 & (uint64_t) rids[i]) == 0);
	}
	// initialize global barriers
	int local_barriers = 32;
	int global_barriers = 28;
	pthread_barrier_t sample_barrier;
	pthread_barrier_t *global_barrier = malloc(global_barriers * sizeof(pthread_barrier_t));
	pthread_barrier_t **local_barrier = malloc(numa * sizeof(pthread_barrier_t*));
	pthread_barrier_init(&sample_barrier, NULL, threads + 1);
	for (t = 0 ; t != global_barriers ; ++t)
		pthread_barrier_init(&global_barrier[t], NULL, threads);
	// initialize local barriers
	for (n = 0 ; n != numa ; ++n) {
		local_barrier[n] = malloc(local_barriers * sizeof(pthread_barrier_t));
		for (t = 0 ; t != local_barriers ; ++t)
			pthread_barrier_init(&local_barrier[n][t], NULL, threads_per_numa);
	}
	// universal meta data
	global_data_t global;
	global.numa = numa;
	global.bits = bits_space;
	global.threads = threads;
	global.max_threads = hardware_threads();
	global.max_numa = numa_max_node() + 1;
	global.fudge = fudge;
	global.keys = keys;
	global.rids = rids;
	global.size = size;
	global.keys_buf = keys_buf;
	global.rids_buf = rids_buf;
	global.interleaved = interleaved;
	global.global_barrier = global_barrier;
	global.local_barrier = local_barrier;
	global.sample_barrier = &sample_barrier;
	// total array size
	uint64_t total_size = 0;
	for (n = 0 ; n != numa ; ++n)
		total_size += size[n];
	// allocate the sample
	if (numa > 1) {
		global.sample_size = 0.001 * total_size;
		global.sample_size &= ~15;
		if (global.sample_size > 100000)
			global.sample_size = 100000;
		global.sample	  = numa_alloc_interleaved(global.sample_size * sizeof(uint64_t));
		global.sample_buf = numa_alloc_interleaved(global.sample_size * sizeof(uint64_t));
		global.sample_hist = malloc(threads * sizeof(uint64_t*));
		for (t = 0 ; t != threads ; ++t)
			global.sample_hist[t] = malloc(256 * sizeof(uint64_t));
	}
	// check if allocation needed
	if (keys_buf[0] == NULL)
		for (n = 0 ; n != numa ; ++n) {
			assert(keys_buf[n] == NULL);
			assert(rids_buf[n] == NULL);
		}
	else
		for (n = 0 ; n != numa ; ++n) {
			assert(keys_buf[n] != NULL);
			assert(rids_buf[n] != NULL);
		}
	global.allocated = keys_buf[0] != NULL;
	// counts
	global.count = malloc(numa * sizeof(uint64_t**));
	for (n = 0 ; n != numa ; ++n)
		global.count[n] = malloc(threads_per_numa * sizeof(uint64_t));
	global.cpu = malloc(threads * sizeof(int));
	global.numa_node = malloc(threads * sizeof(int));
	global.numa_local_count = malloc(threads * sizeof(uint64_t*));
	schedule_threads(global.cpu, global.numa_node, threads, numa);
	// spawn threads
	for (t = 0 ; t != threads ; ++t) {
		data[t].id = t;
		data[t].seed = rand();
		data[t].global = &global;
		pthread_create(&id[t], NULL, sort_thread, (void*) &data[t]);
	}
	// free sample data
	pthread_barrier_wait(&sample_barrier);
	pthread_barrier_destroy(&sample_barrier);
	numa_free(global.sample,     global.sample_size * sizeof(uint32_t));
	numa_free(global.sample_buf, global.sample_size * sizeof(uint32_t));
	// join threads
	for (t = 0 ; t != threads ; ++t)
		pthread_join(id[t], NULL);
	// assemble times
	uint64_t at = 0, dt = 0, st = 0;
	uint64_t ht[] = {0, 0, 0, 0, 0, 0, 0, 0};
	uint64_t pt[] = {0, 0, 0, 0, 0, 0, 0, 0};
	for (t = 0 ; t != threads ; ++t) {
		at += data[t].alloc_time;
		dt += data[t].sample_time;
		st += data[t].numa_shuffle_time;
		for (p = 0 ; bits_space[p] != 0 ; ++p) {
			ht[p] += data[t].hist_time[p];
			pt[p] += data[t].part_time[p];
		}
	}
	times[0] = at / threads;    description[0] = "Allocation time:		  ";
	times[1] = dt / threads;    description[1] = "Sampling time:		  ";
	times[2] = ht[0] / threads; description[2] = "Range-radix histogram time: ";
	times[3] = pt[0] / threads; description[3] = "Range-radix partition time: ";
	times[4] = st / threads;    description[4] = "Data shuffle time:	  ";
	times[5] = ht[1] / threads; description[5] = "2nd radix histogram time:   ";
	times[6] = pt[1] / threads; description[6] = "2nd radix partition time:   ";
	times[7] = ht[2] / threads; description[7] = "3rd radix histogram time:   ";
	times[8] = pt[2] / threads; description[8] = "3rd radix partition time:   ";
	times[9] = ht[3] / threads; description[9] = "4th radix histogram time:   ";
	times[10]= pt[3] / threads; description[10]= "4th radix partition time:   ";
	times[11]= ht[4] / threads; description[11]= "5th radix histogram time:   ";
	times[12]= pt[4] / threads; description[12]= "5th radix partition time:   ";
	times[13]= ht[5] / threads; description[13]= "6th radix histogram time:   ";
	times[14]= pt[5] / threads; description[14]= "6th radix partition time:   ";
	description[15] = NULL;
	// destroy barriers
	for (t = 0 ; t != global_barriers ; ++t)
		pthread_barrier_destroy(&global_barrier[t]);
	for (n = 0 ; n != numa ; ++n) {
		for (t = 0 ; t != local_barriers ; ++t)
			pthread_barrier_destroy(&local_barrier[n][t]);
		free(local_barrier[n]);
	}
	free(local_barrier);
	free(global_barrier);
	// release memory
	// free(id);
	free(global.numa_node);
	free(global.cpu);
	for (i = 0 ; i != numa ; ++i) {
		for (j = 0 ; j != threads_per_numa ; ++j)
			free(global.count[i][j]);
		free(global.count[i]);
	}
	if (numa > 1) {
		numa_free(global.sample,     global.sample_size * sizeof(uint64_t));
		numa_free(global.sample_buf, global.sample_size * sizeof(uint64_t));
	}
	for (i = 0 ; i != threads ; ++i)
		free(global.numa_local_count[i]);
	free(global.numa_local_count);
	free(global.count);
	// free(data);
	if (numa > 1) bit_passes++;
	return bit_passes & 1;
}

void *check_thread(void *arg)
{
	thread_data_t *a = (thread_data_t*) arg;
	global_data_t *d = a->global;
	int i, id = a->id;
	int numa = d->numa;
	int numa_node = d->numa_node[id];
	int threads = d->threads;
	int threads_per_numa = threads / numa;
	// id in local numa threads
	int numa_local_id = 0;
	for (i = 0 ; i != id ; ++i)
		if (d->numa_node[i] == numa_node)
			numa_local_id++;
	// compute checksum
	uint64_t numa_size = d->size[numa_node];
	uint64_t size = numa_size / threads_per_numa;
	uint64_t offset = size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		size = numa_size - size * numa_local_id;
	uint64_t *keys = &d->keys[numa_node][offset];
	uint64_t *rids = NULL;
	if (d->rids != NULL)
		rids =	&d->rids[numa_node][offset];
	uint64_t *keys_end = &keys[size];
	uint64_t sum = 0;
	uint64_t pkey = 0;
	while (keys != keys_end) {
		uint64_t key = *keys++;
		// if (rids) assert(key == *rids++);
		// assert(key >= pkey);
		sum += key;
		pkey = key;
	}
	a->checksum = sum;
	pthread_exit(NULL);
}

uint64_t check(uint64_t **keys, uint64_t **rids, uint64_t *size, int numa, int same)
{
	int max_threads = hardware_threads();
	int n, t, threads = 0;
	for (t = 0 ; t != max_threads ; ++t)
		if (numa_node_of_cpu(t) < numa)
			threads++;
	global_data_t global;
	global.threads = threads;
	global.numa = numa;
	global.keys = keys;
	global.rids = same ? rids : NULL;
	global.size = size;
	global.cpu = malloc(threads * sizeof(int));
	global.numa_node = malloc(threads * sizeof(int));
	schedule_threads(global.cpu, global.numa_node, threads, numa);
	thread_data_t *data = malloc(threads * sizeof(thread_data_t));
	pthread_t *id = malloc(threads * sizeof(pthread_t));
	for (t = 0 ; t != threads ; ++t) {
		data[t].id = t;
		data[t].global = &global;
		pthread_create(&id[t], NULL, check_thread, (void*) &data[t]);
	}
	for (n = 1 ; n != numa ; ++n)
		assert(keys[n][0] >= keys[n - 1][size[n - 1] - 1]);
	uint64_t checksum = 0;
	for (t = 0 ; t != threads ; ++t) {
		pthread_join(id[t], NULL);
		checksum += data[t].checksum;
	}
	free(global.numa_node);
	free(global.cpu);
	free(data);
	free(id);
	return checksum;
}

uint64_t read_from_file(uint64_t **keys, uint64_t *size, int numa, const char *name)
{
	FILE *fp = fopen(name, "r");
	assert(fp != NULL);
	fseek(fp, 0, SEEK_END);
	uint64_t total_size = 0; int n, c;
	for (n = 0 ; n != numa ; ++n)
		total_size += size[n];
	assert(total_size <= ftell(fp));
	fseek(fp, 0, SEEK_SET);
	assert(ftell(fp) == 0);
	uint64_t checksum = 0;
	uint64_t p, perc = total_size / 100;;
	uint64_t q = 1, done = 0;
	for (n = 0 ; n != numa ; ++n) {
		memory_bind(n);
		for (c = 0 ; numa_node_of_cpu(c) != n ; ++c);
		cpu_bind(c);
		uint64_t rem_size = size[n];
		uint64_t *key = keys[n];
		while (rem_size) {
			uint64_t size = 4096;
			if (size > rem_size) size = rem_size;
			size = fread(key, sizeof(uint64_t), size, fp);
			assert(size > 0);
			for (p = 0 ; p != size ; ++p)
				checksum += key[p];
			rem_size -= size;
			key += size;
			done += size;
			if (done > (perc * q))
				fprintf(stderr, "Finished %ld%%\n", q++);
		}
	}
	p = ftell(fp);
	assert(p == total_size * sizeof(uint64_t));
	fclose(fp);
	return checksum;
}

int main(int argc, char **argv)
{
	int r, i, max_threads = hardware_threads();
	int max_numa = numa_max_node() + 1;
	uint64_t tuples = argc > 1 ? atoi(argv[1]) : 1000;
	int threads = argc > 2 ? atoi(argv[2]) : max_threads;
	int numa = argc > 3 ? atoi(argv[3]) : max_numa;
	int bits = argc > 4 ? atoi(argv[4]) : 64;
	int interleaved = argc > 5 ? atoi(argv[5]) : 0;
	int allocated = argc > 6 ? atoi(argv[6]) : 1;
	char *name = NULL;
	double theta = 1.0;
	if (argc > 7) {
		assert(bits == 64);
		if (isdigit(argv[7][0]))
			theta = atof(argv[7]);
		else {
			name = argv[7];
			FILE *fp = fopen(name, "r");
			assert(fp != NULL);
			fclose(fp);
		}
	}
	int threads_per_numa = threads / numa;
	int same_key_payload = 1;
	global_tuples = tuples;
	tuples *= 1000000;
	assert(bits > 0 && bits <= 64);
	assert(numa > 0 && numa <= 8);
	assert(threads >= numa && threads % numa == 0);
	uint64_t tuples_per_numa = tuples / numa;
	double fudge = 1.1;
	uint64_t capacity_per_numa = tuples_per_numa * fudge;
	uint64_t *keys[numa], *keys_buf[numa];
	uint64_t *rids[numa], *rids_buf[numa];
	uint64_t size[numa], cap[numa];
	uint32_t seed = micro_time();
	srand(seed);
	fprintf(stderr, "Tuples: %.2f mil. (%.1f GB)\n", tuples / 1000000.0,
			(tuples * 16.0) / (1024 * 1024 * 1024));
	fprintf(stderr, "NUMA nodes: %d\n", numa);
	if (interleaved)
		fprintf(stderr, "Memory interleaved\n");
	else
		fprintf(stderr, "Memory bound\n");
	if (allocated)
		fprintf(stderr, "Buffers pre-allocated\n");
	else
		fprintf(stderr, "Buffers not pre-allocated\n");
	fprintf(stderr, "Hardware threads: %d (%d per NUMA)\n",
			max_threads, max_threads / max_numa);
	fprintf(stderr, "Threads: %d (%d per NUMA)\n", threads, threads / numa);
	fprintf(stderr, "Sorting bits: %d\n", bits);
	for (i = 0 ; i != numa ; ++i) {
		size[i] = tuples_per_numa;
		cap[i] = size[i] * fudge;
		keys_buf[i] = NULL;
		rids_buf[i] = NULL;
	}
	// initialize space
	uint64_t t = micro_time();
	uint64_t sum_k, sum_v;
	srand(t);
	if (argc <= 6) {
		sum_k = init_64(keys, size, cap, threads, numa, bits, 0.0, 0, interleaved);
		srand(t);
		sum_v = init_64(rids, size, cap, threads, numa, bits, 0.0, 0, interleaved);
		assert(sum_k == sum_v);
	} else {
		uint64_t ranks[33];
		init_64(keys, size, cap, threads, numa, 0, 0.0, 0, interleaved);
		if (name != NULL) {
			fprintf(stderr, "Opening file: %s\n", name);
			sum_k = read_from_file(keys, size, numa, name);
		} else {
			fprintf(stderr, "Generating zipfian with theta = %.2f\n", theta);
			abort();
		}
		same_key_payload = 0;
		sum_v = init_64(rids, size, cap, threads, numa, 64, 0.0, 0, interleaved);
	}
	if (allocated) {
		init_64(keys_buf, size, cap, threads, numa, 0, 0.0, 0, interleaved);
		init_64(rids_buf, size, cap, threads, numa, 0, 0.0, 0, interleaved);
	}
	t = micro_time() - t;
	fprintf(stderr, "Generation time: %ld us\n", t);
	fprintf(stderr, "Generation rate: %.1f mrps\n", tuples * 1.0 / t);
	// sort info
	char *desc[16];
	uint64_t times[16];
	// call parallel sort

	PerfCounter *pc = PerfCounter_init();
    if (pc == NULL) {
        fprintf(stderr, "Failed to initialize PerfCounter\n");
        return 1;
    }

    printf("Starting counters...\n");
    PerfCounter_startCounters(pc);
	
	t = micro_time();
	r = sort(keys, rids, size, threads, numa, bits, fudge,
	         keys_buf, rids_buf, desc, times, interleaved);
	t = micro_time() - t;

	PerfCounter_stopCounters(pc);
    printf("Stopped counters.\n");

    printf("Performance counters report:\n");
    PerfCounter_printReport(pc, stdout, 1);

    PerfCounter_cleanup(pc);
	
	// print bit passes
	int bits_space[8];
	distribute_bits(bits, numa, bits_space, 1);
	// print sort times
	fprintf(stderr, "Sort time: %ld us\n", t);
	double gigs = (tuples * 16.0) / (1024 * 1024 * 1024);
	fprintf(stderr, "Sort rate: %.1f mrps (%.2f GB / sec)\n",
		tuples * 1.0 / t, (gigs * 1000000) / t);
	// compute total time
	uint64_t total_time = 0;
	for (i = 0 ; desc[i] != NULL ; ++i)
		total_time += times[i];
	// show part times
	for (i = 0 ; desc[i] != NULL ; ++i)
		fprintf(stderr, "%s %10ld us (%5.2f%%)\n", desc[i],
				 times[i], times[i] * 100.0 / total_time);
	fprintf(stderr, "Noise time loss: %.2f%%\n", t * 100.0 / total_time - 100);
	// show numa allocation
	for (i = 0 ; i != numa ; ++i)
		fprintf(stderr, "Node %d:%6.2f%%\n", i, size[i] * 100.0 / tuples);
	// check sort order and sum
	if (r) fprintf(stderr, "Destination changed\n");
	else fprintf(stderr, "Destination remained the same\n");
	uint64_t **keys_out = r ? keys_buf : keys;
	uint64_t **rids_out = r ? rids_buf : rids;
	uint64_t checksum = check(keys_out, rids_out, size, numa, same_key_payload);
	// assert(checksum == sum_k);
	// free sort data
	for (i = 0 ; i != numa ; ++i)
		if (interleaved) {
			numa_free(keys_buf[i], cap[i] * sizeof(uint64_t));
			numa_free(rids_buf[i], cap[i] * sizeof(uint64_t));
			numa_free(keys[i], cap[i] * sizeof(uint64_t));
			numa_free(rids[i], cap[i] * sizeof(uint64_t));
		} else {
			free(keys_buf[i]);
			free(rids_buf[i]);
			free(keys[i]);
			free(rids[i]);
		}
	printf("%.1f mrps (%.2f GB / sec)\n",
		tuples * 1.0 / t, (gigs * 1000000) / t);
	return EXIT_SUCCESS;
}
