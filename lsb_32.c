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
#undef _GNU_SOURCE

#ifdef AVX
#include <immintrin.h>
#else
#include <smmintrin.h>
#endif

#include "rand.h"
#include "util.h"

#include "perf_counter.h"


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

void cpu_bind(int cpu_id)
{
	int cpus = hardware_threads();
	size_t size = CPU_ALLOC_SIZE(cpus);
	cpu_set_t *cpu_set = CPU_ALLOC(cpus);
	assert(cpu_set != NULL);
	CPU_ZERO_S(size, cpu_set);
	CPU_SET_S(cpu_id, size, cpu_set);
	assert(pthread_setaffinity_np(pthread_self(),
	       size, cpu_set) == 0);
	CPU_FREE(cpu_set);
}

void memory_bind(int numa_id)
{
	char numa_id_str[12];
	struct bitmask *numa_node;
	sprintf(numa_id_str, "%d", numa_id);
	numa_node = numa_parse_nodestring(numa_id_str);
	numa_set_membind(numa_node);
	numa_free_nodemask(numa_node);
}

void *mamalloc(size_t size)
{
	void *ptr = NULL;
	return posix_memalign(&ptr, 64, size) ? NULL : ptr;
}

void swap_i(int *x, int *y)
{
	int t = *x; *x = *y; *y = t;
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
	assert(total_bits <= 35);
	int limit[] = {12, 24, 35};
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

void partition_keys(uint32_t *keys, uint32_t *keys_out, uint64_t size,
                    uint64_t **hist, uint8_t shift_bits, uint8_t radix_bits,
                    int thread_id, int threads, pthread_barrier_t *barrier)
{
	// inputs and outputs must be aligned
	assert(0 == (15 & (size_t) keys));
	assert(0 == (63 & (size_t) keys_out));
	assert((size & 3) == 0);
	// set sub-array for current thread
	uint64_t local_size = (size / threads) & ~15;
	uint32_t *local_keys = &keys[local_size * thread_id];
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
	__m128i m = _mm_set1_epi32((1 << radix_bits) - 1);
	for (i = p = 0 ; i != local_size ; i += 4) {
		__m128i k = _mm_load_si128((__m128i*) &local_keys[i]);
		__m128i h = _mm_srl_epi32(k, s);
		h = _mm_and_si128(h, m);
		for (j = 0 ; j != 4 ; ++j) {
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			local_hist[p]++;
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
		}
	}
	// wait all threads to complete histogram generation
	pthread_barrier_wait(&barrier[0]);
	// initialize buffer
	uint64_t *index = malloc(partitions * sizeof(uint64_t));
	uint32_t *buf = mamalloc((partitions << 4) * sizeof(uint32_t));
	uint64_t *buf_64 = (uint64_t*) buf;
	for (i = p = 0 ; p != partitions ; ++p) {
		for (t = 0 ; t != thread_id ; ++t)
			i += hist[t][p];
		index[p] = i;
		for (; t != threads ; ++t)
			i += hist[t][p];
	}
	assert(i == size);
	// main partitioning loop
	for (i = p = 0 ; i != local_size ; i += 4) {
		__m128i k = _mm_load_si128((__m128i*) &local_keys[i]);
		__m128i h = _mm_srl_epi32(k, s);
		h = _mm_and_si128(h, m);
		for (j = 0 ; j != 4 ; ++j) {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			// offset in the cache line pair
			uint64_t *src_64 = &buf_64[p << 3];
			uint32_t *src = (uint32_t*) src_64;
			uint64_t indexl = index[p]++;
			uint64_t offset = indexl & 15;
			asm("movd	%0, (%1,%2,4)" :: "x"(k), "r"(src), "r"(offset) : "memory");
			if (offset == 15) {
				uint32_t *dst = &keys_out[indexl - 15];
				__m128i r0 = _mm_load_si128((__m128i*) &src[0]);
				__m128i r1 = _mm_load_si128((__m128i*) &src[4]);
				__m128i r2 = _mm_load_si128((__m128i*) &src[8]);
				__m128i r3 = _mm_load_si128((__m128i*) &src[12]);
				_mm_stream_si128((__m128i*) &dst[0], r0);
				_mm_stream_si128((__m128i*) &dst[4], r1);
				_mm_stream_si128((__m128i*) &dst[8], r2);
				_mm_stream_si128((__m128i*) &dst[12],r3);
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k = _mm_shuffle_epi32(k, _MM_SHUFFLE(0, 3, 2, 1));
		}
	}
	// wait all threads to complete main partition part
	pthread_barrier_wait(&barrier[1]);
	// flush remaining items from buffers to output
	for (p = 0 ; p != partitions ; ++p) {
		uint32_t *src = &buf[p << 4];
		uint64_t indexl = index[p];
		uint64_t remain = indexl & 15;
		uint64_t offset = 0;
		if (remain > local_hist[p])
			offset = remain - local_hist[p];
		indexl -= remain - offset;
		while (offset != remain)
			_mm_stream_si32(&keys_out[indexl++], src[offset++]);
	}
	// wait all threads to complete last partition part
	pthread_barrier_wait(&barrier[2]);
	free(index);
	free(buf);
}

void histogram_numa_2(uint32_t *keys, uint64_t size, uint64_t *count,
                      uint8_t radix_bits, uint32_t delim[])
{
	uint32_t convert = ((uint32_t) 1) << 31;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi32((1 << radix_bits) - 1);
	__m128i d = _mm_set1_epi32(delim[0] - convert);
	__m128i c = _mm_set1_epi32(convert);
	__m128i k; int i;
	// check for unaligned keys
	uint64_t p = 0; i = 0;
	uint32_t unaligned_keys[4];
	while ((15 & ((uint64_t) keys)) && i != size)
		unaligned_keys[i++] = *keys++;
	uint32_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint32_t *keys_end = &keys[size - i];
	if (i) {
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k = _mm_load_si128((__m128i*) keys);
		keys += 4; i = 4;
		unaligned_intro:;
		__m128i h = _mm_and_si128(k, m);
		__m128i k_c = _mm_sub_epi32(k, c);
		__m128i e = _mm_cmpgt_epi32(k_c, d);
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
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
}

void partition_numa_2(uint32_t *keys, uint32_t *rids, uint64_t size,
                      uint64_t *offsets, uint64_t *sizes, uint64_t *buf,
                      uint32_t *keys_out, uint32_t *rids_out,
                      uint8_t radix_bits, uint32_t delim[])
{
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	int i, partitions = 1 << (radix_bits + 1);
	// initialize buffers
	for (i = 0 ; i != partitions ; ++i)
		buf[(i << 4) | 15] = offsets[i];
	// main partition loop
	uint32_t convert = ((uint32_t) 1) << 31;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi32((1 << (radix_bits)) - 1);
	__m128i d = _mm_set1_epi32(delim[0] - convert);
	__m128i c = _mm_set1_epi32(convert);
	__m128i k, v;
	// check for unaligned items
	uint32_t unaligned_keys[4];
	uint32_t unaligned_rids[4];
	uint64_t p = 0; i = 0;
	while ((15 & ((uint64_t) keys)) && i != size) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	uint32_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint32_t *keys_end = &keys[size - i];
	if (i) {
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		v = _mm_loadu_si128((__m128i*) unaligned_rids);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k = _mm_load_si128((__m128i*) keys);
		v = _mm_load_si128((__m128i*) rids);
		keys += 4; rids += 4; i = 4;
		unaligned_intro:;
		__m128i k_c = _mm_sub_epi32(k, c);
		__m128i e = _mm_cmpgt_epi32(k_c, d);
		__m128i h = _mm_and_si128(k, m);
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
			uint64_t index = src[15]++;
			uint64_t offset = index & 15;
			// pack and store
			__m128i kvxx = _mm_unpacklo_epi32(k, v);
			_mm_storel_epi64((__m128i*) &src[offset], kvxx);
			if (offset == 15) {
				float *dest_x = (float*) &keys_out[index - 15];
				float *dest_y = (float*) &rids_out[index - 15];
				// load cache line from cache to 8 128-bit registers
				__m128 r0 = _mm_load_ps((float*) &src[0]);
				__m128 r1 = _mm_load_ps((float*) &src[2]);
				__m128 r2 = _mm_load_ps((float*) &src[4]);
				__m128 r3 = _mm_load_ps((float*) &src[6]);
				__m128 r4 = _mm_load_ps((float*) &src[8]);
				__m128 r5 = _mm_load_ps((float*) &src[10]);
				__m128 r6 = _mm_load_ps((float*) &src[12]);
				__m128 r7 = _mm_load_ps((float*) &src[14]);
				// split first column
				__m128 x0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(2, 0, 2, 0));
				// stream first column
				_mm_stream_ps(&dest_x[0], x0);
				_mm_stream_ps(&dest_x[4], x1);
				_mm_stream_ps(&dest_x[8], x2);
				_mm_stream_ps(&dest_x[12],x3);
				// split second column
				__m128 y0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(3, 1, 3, 1));
				// stream second column
				_mm_stream_ps(&dest_y[0], y0);
				_mm_stream_ps(&dest_y[4], y1);
				_mm_stream_ps(&dest_y[8], y2);
				_mm_stream_ps(&dest_y[12],y3);
				// restore overwritten pointer
				src[15] = index + 1;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k = _mm_shuffle_epi32(k, _MM_SHUFFLE(0, 3, 2, 1));
			v = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	if (i) {
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		v = _mm_loadu_si128((__m128i*) unaligned_rids);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
#ifdef BG
	// check partition sanity
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t index = buf[(i << 4) | 15];
		assert(index - offsets[i] == sizes[i]);
	}
#endif
}

void histogram_numa_4(uint32_t *keys, uint64_t size, uint64_t *count,
                      uint8_t radix_bits, uint32_t delim[])
{
	uint32_t convert = ((uint32_t) 1) << 31;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi32((1 << (radix_bits)) - 1);
	__m128i d1 = _mm_set1_epi32(delim[0] - convert);
	__m128i d2 = _mm_set1_epi32(delim[1] - convert);
	__m128i d3 = _mm_set1_epi32(delim[2] - convert);
	__m128i c = _mm_set1_epi32(convert);
	__m128i k; int i;
	// check for unaligned keys
	uint64_t p = 0; i = 0;
	uint32_t unaligned_keys[4];
	while ((15 & ((uint64_t) keys)) && i != size)
		unaligned_keys[i++] = *keys++;
	uint32_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint32_t *keys_end = &keys[size - i];
	if (i) {
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k = _mm_load_si128((__m128i*) keys);
		keys += 4; i = 4;
		unaligned_intro:;
		__m128i k_c = _mm_sub_epi32(k, c);
		__m128i e1 = _mm_cmpgt_epi32(k_c, d2);
		__m128i d13 = _mm_blendv_epi8(d1, d3, e1);
		__m128i e2 = _mm_cmpgt_epi32(k_c, d13);
		__m128i h = _mm_and_si128(k, m);
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
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
}

void partition_numa_4(uint32_t *keys, uint32_t *rids, uint64_t size,
		      uint64_t *offsets, uint64_t *sizes, uint64_t *buf,
		      uint32_t *keys_out, uint32_t *rids_out,
		      uint8_t radix_bits, uint32_t delim[])
{
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	int i, partitions = 1 << (radix_bits + 2);
	// initialize buffers
	for (i = 0 ; i != partitions ; ++i)
		buf[(i << 4) | 15] = offsets[i];
	// main partition loop
	uint32_t convert = ((uint32_t) 1) << 31;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi32((1 << (radix_bits)) - 1);
	__m128i d1 = _mm_set1_epi32(delim[0] - convert);
	__m128i d2 = _mm_set1_epi32(delim[1] - convert);
	__m128i d3 = _mm_set1_epi32(delim[2] - convert);
	__m128i c = _mm_set1_epi32(convert);
	__m128i k, v;
	// check for unaligned items
	uint32_t unaligned_keys[4];
	uint32_t unaligned_rids[4];
	uint64_t p = 0; i = 0;
	while ((15 & ((uint64_t) keys)) && i != size) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	uint32_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint32_t *keys_end = &keys[size - i];
	if (i) {
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		v = _mm_loadu_si128((__m128i*) unaligned_rids);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k = _mm_load_si128((__m128i*) keys);
		v = _mm_load_si128((__m128i*) rids);
		keys += 4; rids += 4; i = 4;
		unaligned_intro:;
		__m128i k_c = _mm_sub_epi32(k, c);
		__m128i e1 = _mm_cmpgt_epi32(k_c, d2);
		__m128i d13 = _mm_blendv_epi8(d1, d3, e1);
		__m128i e2 = _mm_cmpgt_epi32(k_c, d13);
		__m128i h = _mm_and_si128(k, m);
		__m128i r = _mm_setzero_si128();
		r = _mm_sub_epi32(r, e1);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e2);
		r = _mm_sll_epi32(r, s);
		h = _mm_or_si128(h, r);
		h = _mm_slli_epi32(h, 4);
		do {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "a"(p));
			// offset in the cache line pair
			uint64_t *src = &buf[p];
			uint64_t index = src[15]++;
			uint64_t offset = index & 15;
			// pack and store
			__m128i kvxx = _mm_unpacklo_epi32(k, v);
			_mm_storel_epi64((__m128i*) &src[offset], kvxx);
			if (offset == 15) {
				float *dest_x = (float*) &keys_out[index - 15];
				float *dest_y = (float*) &rids_out[index - 15];
				// load cache line from cache to 8 128-bit registers
				__m128 r0 = _mm_load_ps((float*) &src[0]);
				__m128 r1 = _mm_load_ps((float*) &src[2]);
				__m128 r2 = _mm_load_ps((float*) &src[4]);
				__m128 r3 = _mm_load_ps((float*) &src[6]);
				__m128 r4 = _mm_load_ps((float*) &src[8]);
				__m128 r5 = _mm_load_ps((float*) &src[10]);
				__m128 r6 = _mm_load_ps((float*) &src[12]);
				__m128 r7 = _mm_load_ps((float*) &src[14]);
				// split first column
				__m128 x0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(2, 0, 2, 0));
				// stream first column
				_mm_stream_ps(&dest_x[0], x0);
				_mm_stream_ps(&dest_x[4], x1);
				_mm_stream_ps(&dest_x[8], x2);
				_mm_stream_ps(&dest_x[12],x3);
				// split second column
				__m128 y0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(3, 1, 3, 1));
				// stream second column
				_mm_stream_ps(&dest_y[0], y0);
				_mm_stream_ps(&dest_y[4], y1);
				_mm_stream_ps(&dest_y[8], y2);
				_mm_stream_ps(&dest_y[12],y3);
				// restore overwritten pointer
				src[15] = index + 1;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k = _mm_shuffle_epi32(k, _MM_SHUFFLE(0, 3, 2, 1));
			v = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	if (i) {
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		v = _mm_loadu_si128((__m128i*) unaligned_rids);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
#ifdef BG
	// check partition sanity
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t index = buf[(i << 4) | 15];
		assert(index - offsets[i] == sizes[i]);
	}
#endif
}

void histogram_numa_8(uint32_t *keys, uint64_t size, uint64_t *count,
                      uint8_t radix_bits, uint32_t delim[])
{
	uint32_t convert = ((uint32_t) 1) << 31;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi32((1 << (radix_bits)) - 1);
	__m128i d1 = _mm_set1_epi32(delim[0] - convert);
	__m128i d2 = _mm_set1_epi32(delim[1] - convert);
	__m128i d3 = _mm_set1_epi32(delim[2] - convert);
	__m128i d4 = _mm_set1_epi32(delim[3] - convert);
	__m128i d5 = _mm_set1_epi32(delim[4] - convert);
	__m128i d6 = _mm_set1_epi32(delim[5] - convert);
	__m128i d7 = _mm_set1_epi32(delim[6] - convert);
	__m128i c = _mm_set1_epi32(convert);
	__m128i k; int i;
	// check for unaligned keys
	uint64_t p = 0; i = 0;
	uint32_t unaligned_keys[4];
	while ((15 & ((uint64_t) keys)) && i != size)
		unaligned_keys[i++] = *keys++;
	uint32_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint32_t *keys_end = &keys[size - i];
	if (i) {
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k = _mm_load_si128((__m128i*) keys);
		keys += 4; i = 4;
		unaligned_intro:;
		__m128i k_c = _mm_sub_epi32(k, c);
		__m128i e1 = _mm_cmpgt_epi32(k_c, d4);
		__m128i d26 = _mm_blendv_epi8(d2, d6, e1);
		__m128i d15 = _mm_blendv_epi8(d1, d5, e1);
		__m128i d37 = _mm_blendv_epi8(d3, d7, e1);
		__m128i e2 = _mm_cmpgt_epi32(k_c, d26);
		__m128i d1357 = _mm_blendv_epi8(d15, d37, e2);
		__m128i e3 = _mm_cmpgt_epi32(k_c, d1357);
		__m128i h = _mm_and_si128(k, m);
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
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
}

void partition_numa_8(uint32_t *keys, uint32_t *rids, uint64_t size,
                      uint64_t *offsets, uint64_t *sizes, uint64_t *buf,
                      uint32_t *keys_out, uint32_t *rids_out,
                      uint8_t radix_bits, uint32_t delim[])
{
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	int i, partitions = 1 << (radix_bits + 3);
	// initialize buffers
	for (i = 0 ; i != partitions ; ++i)
		buf[(i << 4) | 15] = offsets[i];
	// main partition loop
	uint32_t convert = ((uint32_t) 1) << 31;
	__m128i s = _mm_set_epi32(0, 0, 0, radix_bits);
	__m128i m = _mm_set1_epi32((1 << (radix_bits)) - 1);
	__m128i d1 = _mm_set1_epi32(delim[0] - convert);
	__m128i d2 = _mm_set1_epi32(delim[1] - convert);
	__m128i d3 = _mm_set1_epi32(delim[2] - convert);
	__m128i d4 = _mm_set1_epi32(delim[3] - convert);
	__m128i d5 = _mm_set1_epi32(delim[4] - convert);
	__m128i d6 = _mm_set1_epi32(delim[5] - convert);
	__m128i d7 = _mm_set1_epi32(delim[6] - convert);
	__m128i c = _mm_set1_epi32(convert);
	__m128i k, v;
	// check for unaligned items
	uint32_t unaligned_keys[4];
	uint32_t unaligned_rids[4];
	uint64_t p = 0; i = 0;
	while ((15 & ((uint64_t) keys)) && i != size) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	uint32_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint32_t *keys_end = &keys[size - i];
	if (i) {
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		v = _mm_loadu_si128((__m128i*) unaligned_rids);
		goto unaligned_intro;
	}
	while (keys != keys_aligned_end) {
		k = _mm_load_si128((__m128i*) keys);
		v = _mm_load_si128((__m128i*) rids);
		keys += 4; rids += 4; i = 4;
		unaligned_intro:;
		__m128i k_c = _mm_sub_epi32(k, c);
		__m128i e1 = _mm_cmpgt_epi32(k_c, d4);
		__m128i d26 = _mm_blendv_epi8(d2, d6, e1);
		__m128i d15 = _mm_blendv_epi8(d1, d5, e1);
		__m128i d37 = _mm_blendv_epi8(d3, d7, e1);
		__m128i e2 = _mm_cmpgt_epi32(k_c, d26);
		__m128i d1357 = _mm_blendv_epi8(d15, d37, e2);
		__m128i e3 = _mm_cmpgt_epi32(k_c, d1357);
		__m128i h = _mm_and_si128(k, m);
		__m128i r = _mm_setzero_si128();
		r = _mm_sub_epi32(r, e1);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e2);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e3);
		r = _mm_sll_epi32(r, s);
		h = _mm_or_si128(h, r);
		h = _mm_slli_epi32(h, 4);
		do {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "a"(p));
			// offset in the cache line pair
			uint64_t *src = &buf[p];
			uint64_t index = src[15]++;
			uint64_t offset = index & 15;
			// pack and store
			__m128i kvxx = _mm_unpacklo_epi32(k, v);
			_mm_storel_epi64((__m128i*) &src[offset], kvxx);
			if (offset == 15) {
				float *dest_x = (float*) &keys_out[index - 15];
				float *dest_y = (float*) &rids_out[index - 15];
				// load cache line from cache to 8 128-bit registers
				__m128 r0 = _mm_load_ps((float*) &src[0]);
				__m128 r1 = _mm_load_ps((float*) &src[2]);
				__m128 r2 = _mm_load_ps((float*) &src[4]);
				__m128 r3 = _mm_load_ps((float*) &src[6]);
				__m128 r4 = _mm_load_ps((float*) &src[8]);
				__m128 r5 = _mm_load_ps((float*) &src[10]);
				__m128 r6 = _mm_load_ps((float*) &src[12]);
				__m128 r7 = _mm_load_ps((float*) &src[14]);
				// split first column
				__m128 x0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(2, 0, 2, 0));
				// stream first column
				_mm_stream_ps(&dest_x[0], x0);
				_mm_stream_ps(&dest_x[4], x1);
				_mm_stream_ps(&dest_x[8], x2);
				_mm_stream_ps(&dest_x[12],x3);
				// split second column
				__m128 y0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(3, 1, 3, 1));
				// stream second column
				_mm_stream_ps(&dest_y[0], y0);
				_mm_stream_ps(&dest_y[4], y1);
				_mm_stream_ps(&dest_y[8], y2);
				_mm_stream_ps(&dest_y[12],y3);
				// restore overwritten pointer
				src[15] = index + 1;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k = _mm_shuffle_epi32(k, _MM_SHUFFLE(0, 3, 2, 1));
			v = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// histogram last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	if (i) {
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		v = _mm_loadu_si128((__m128i*) unaligned_rids);
		keys_aligned_end = keys_end;
		goto unaligned_intro;
	}
#ifdef BG
	// check partition sanity
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t index = buf[(i << 4) | 15];
		assert(index - offsets[i] == sizes[i]);
	}
#endif
}

void histogram(uint32_t *keys, uint64_t size, uint64_t *count,
               uint8_t shift_bits, uint8_t radix_bits)
{
	uint32_t *keys_end = &keys[size];
	uint64_t partitions = 1 << radix_bits;
	while ((15 & ((uint64_t) keys)) && keys != keys_end)
		count[(*keys++ >> shift_bits) & (partitions - 1)]++;
	if (keys == keys_end) return;
	__m128i m = _mm_set1_epi32(partitions - 1);
	__m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
	uint32_t *keys_aligned_end = &keys[(keys_end - keys) & ~3];
	uint64_t p1 = 0, p2 = 0, p3 = 0, p4 = 0;
	do {
		__m128i k = _mm_load_si128((__m128i*) keys);
		keys += 4;
		__m128i h = _mm_srl_epi32(k, s);
		h = _mm_and_si128(h, m);
		__m128i h_s1 = _mm_shuffle_epi32(h, 1);
		__m128i h_s2 = _mm_shuffle_epi32(h, 2);
		__m128i h_s3 = _mm_shuffle_epi32(h, 3);
		asm("movd	%1, %%eax" : "=a"(p1) : "x"(h),    "0"(p1));
		asm("movd	%1, %%ebx" : "=b"(p2) : "x"(h_s1), "0"(p2));
		asm("movd	%1, %%ecx" : "=c"(p3) : "x"(h_s2), "0"(p3));
		asm("movd	%1, %%edx" : "=d"(p4) : "x"(h_s3), "0"(p4));
		count[p1]++;
		count[p2]++;
		count[p3]++;
		count[p4]++;
	} while (keys != keys_aligned_end);
	while (keys != keys_end)
		count[(*keys++ >> shift_bits) & (partitions - 1)]++;
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

void partition(uint32_t *keys, uint32_t *rids, uint64_t size,
               uint64_t *offsets, uint64_t *sizes, uint64_t *buf,
               uint32_t *keys_out, uint32_t *rids_out,
               uint8_t shift_bits, uint8_t radix_bits)
{
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	int i, partitions = 1 << radix_bits;
	// initialize partition buffers
	for (i = 0 ; i != partitions ; ++i)
		buf[(i << 4) | 15] = offsets[i];
	// main partition loop that processes input aligned data
	__m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
	__m128i m = _mm_set1_epi32((1 << radix_bits) - 1);
	__m128i k, v;
	// space for unaligned items
	uint32_t unaligned_keys[4];
	uint32_t unaligned_rids[4];
	// first 0-3 unaligned items
	uint64_t p = 0; i = 0;
	while ((15 & ((uint64_t) keys)) && i != size) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	uint32_t *keys_aligned_end = &keys[(size - i) & ~3];
	uint32_t *keys_end = &keys[size - i];
	if (i) {
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		v = _mm_loadu_si128((__m128i*) unaligned_rids);
		goto unaligned_intro;
	}
	// loop of data partitioning
	while (keys != keys_aligned_end) {
		k = _mm_load_si128((__m128i*) keys);
		v = _mm_load_si128((__m128i*) rids);
		keys += 4; rids += 4; i = 4;
		unaligned_intro:;
		__m128i h = _mm_srl_epi32(k, s);
		h = _mm_and_si128(h, m);
		h = _mm_slli_epi32(h, 4);
		do {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			// offset in the cache line pair
			uint64_t *src = &buf[p];
			uint64_t index = src[15]++;
			uint64_t offset = index & 15;
			// pack and store
			__m128i kvxx = _mm_unpacklo_epi32(k, v);
			_mm_storel_epi64((__m128i*) (float*) &src[offset], kvxx);
			if (offset == 15) {
				float *dest_x = (float*) &keys_out[index - 15];
				float *dest_y = (float*) &rids_out[index - 15];
				// load cache line from cache to 8 128-bit registers
				__m128 r0 = _mm_load_ps((float*) &src[0]);
				__m128 r1 = _mm_load_ps((float*) &src[2]);
				__m128 r2 = _mm_load_ps((float*) &src[4]);
				__m128 r3 = _mm_load_ps((float*) &src[6]);
				__m128 r4 = _mm_load_ps((float*) &src[8]);
				__m128 r5 = _mm_load_ps((float*) &src[10]);
				__m128 r6 = _mm_load_ps((float*) &src[12]);
				__m128 r7 = _mm_load_ps((float*) &src[14]);
				// split first column
				__m128 x0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(2, 0, 2, 0));
				// stream first column
				_mm_stream_ps(&dest_x[0], x0);
				_mm_stream_ps(&dest_x[4], x1);
				_mm_stream_ps(&dest_x[8], x2);
				_mm_stream_ps(&dest_x[12],x3);
				// split second columnuint64_t **count
				__m128 y0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(3, 1, 3, 1));
				// stream second column
				_mm_stream_ps(&dest_y[0], y0);
				_mm_stream_ps(&dest_y[4], y1);
				_mm_stream_ps(&dest_y[8], y2);
				_mm_stream_ps(&dest_y[12],y3);
				// restore overwritten pointer
				src[15] = index + 1;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k = _mm_shuffle_epi32(k, _MM_SHUFFLE(0, 3, 2, 1));
			v = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// partition last 0-3 unaligned items
	i = 0; p = 0;
	while (keys != keys_end) {
		unaligned_keys[i] = *keys++;
		unaligned_rids[i++] = *rids++;
	}
	if (i) {
		keys_aligned_end = keys_end;
		k = _mm_loadu_si128((__m128i*) unaligned_keys);
		v = _mm_loadu_si128((__m128i*) unaligned_rids);
		goto unaligned_intro;
	}
#ifdef BG
	// check partition sanity
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t index = buf[(i << 4) | 15];
		assert(index - offsets[i] == sizes[i]);
	}
#endif
}

void finalize(uint64_t *sizes, uint64_t *buf,
              uint32_t *keys_out, uint32_t *rids_out, int partitions)
{	int i;
	// find offset to align output
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	// flush remaining items from buffers to output
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t *src = &buf[i << 4];
		uint64_t index = src[15];
		uint64_t rem = index & 15;
		uint64_t off = 0;
		if (rem > sizes[i])
			off = rem - sizes[i];
		index -= rem - off;
		while (off != rem) {
			uint64_t pair = src[off++];
			keys_out[index] = (uint32_t) pair;
			rids_out[index++] = (uint32_t) (pair >> 32);
		}
	}
}

void swap_ppi(uint32_t ***a, uint32_t ***b)
{
	uint32_t **t = *a; *a = *b; *b = t;
}

void extract_delimiters(uint32_t *sample, uint64_t sample_size, uint32_t *delimiter)
{
	uint64_t i, parts = 0;
	while (delimiter[parts] != ~0) parts++;
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
	int bits_space;
	double fudge;
	uint32_t **keys;
	uint32_t **rids;
	uint64_t *size;
	uint32_t **keys_buf;
	uint32_t **rids_buf;
	uint64_t ***count;
	uint64_t **numa_local_count;
	uint32_t *sample;
	uint32_t *sample_buf;
	uint64_t **sample_hist;
	uint64_t sample_size;
	int *numa_node;
	int *cpu;
	int allocated;
	int interleaved;
	int threads;
	int numa;
	int max_threads;
	int max_numa;
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

int uint32_compare(const void *x, const void *y)
{
	uint32_t a = *((uint32_t*) x);
	uint32_t b = *((uint32_t*) y);
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
	// total array size
	uint64_t total_size = 0;
	for (n = 0 ; n != numa ; ++n)
		total_size += d->size[n];
	// bind thread and its allocation
	if (threads <= d->max_threads)
		cpu_bind(d->cpu[id]);
	if (numa <= d->max_numa)
		memory_bind(d->numa_node[id]);
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
				d->keys_buf[numa_node] = numa_alloc_interleaved(cap * sizeof(uint32_t));
				d->rids_buf[numa_node] = numa_alloc_interleaved(cap * sizeof(uint32_t));
			} else {
				d->keys_buf[numa_node] = mamalloc(cap * sizeof(uint32_t));
				d->rids_buf[numa_node] = mamalloc(cap * sizeof(uint32_t));
			}
		}
		pthread_barrier_wait(&local_barrier[lb++]);
	}
	uint32_t *keys = &d->keys[numa_node][offset];
	uint32_t *rids = &d->rids[numa_node][offset];
	uint32_t *keys_end = NULL, *rids_end = NULL;
	if (!d->allocated) {
		uint32_t *keys_buf = &d->keys_buf[numa_node][offset];
		uint32_t *rids_buf = &d->rids_buf[numa_node][offset];
		uint64_t p;
		for (p = 0 ; p != size ; ++p)
			_mm_stream_si32(&keys_buf[p], 0);
		for (p = 0 ; p != size ; ++p)
			_mm_stream_si32(&rids_buf[p], 0);
		pthread_barrier_wait(&local_barrier[lb++]);
	}
	tim = micro_time() - tim;
	a->alloc_time = tim;
	// sample keys from local data
	tim = micro_time();
	uint32_t *delimiter = calloc(numa, sizeof(uint32_t));
	delimiter[numa - 1] = ~0;
	if (numa > 1) {
		assert((d->sample_size & 3) == 0);
		uint64_t p, sample_size = (d->sample_size / threads) & ~15;
		uint32_t *sample = &d->sample[sample_size * id];
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
		gb += 12;
		extract_delimiters(d->sample, d->sample_size, delimiter);
	}
	tim = micro_time() - tim;
	a->sample_time = tim;
	tim = micro_time();
	if (numa == 1)
		histogram(keys, size, count, 0, radix_bits);
	else if (numa == 2)
		histogram_numa_2(keys, size, count, radix_bits, delimiter);
	else if (numa <= 4)
		histogram_numa_4(keys, size, count, radix_bits, delimiter);
	else if (numa <= 8)
		histogram_numa_8(keys, size, count, radix_bits, delimiter);
	// local counts for numa transfer
	tim = micro_time() - tim;
	a->hist_time[0] = tim;
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
	uint32_t *keys_out = d->keys_buf[numa_node];
	uint32_t *rids_out = d->rids_buf[numa_node];
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
					_mm_stream_si32((int32_t*) keys_out++, *keys++);
				while (rids != rids_end)
					_mm_stream_si32((int32_t*) rids_out++, *rids++);
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
	uint32_t **keys_a = numa > 1 ? d->keys : d->keys_buf;
	uint32_t **rids_a = numa > 1 ? d->rids : d->rids_buf;
	uint32_t **keys_b = numa > 1 ? d->keys_buf : d->keys;
	uint32_t **rids_b = numa > 1 ? d->rids_buf : d->rids;
	// local partitioning phases
	size = (numa_size / threads_per_numa) & ~3;
	offset = size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		size = numa_size - size * numa_local_id;
	counts = d->count[numa_node];
	count = d->count[numa_node][numa_local_id];
	int pass = 0;
	int shift_bits = 0;
	while (d->bits[++pass] != 0) {
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
		swap_ppi(&keys_a, &keys_b);
		swap_ppi(&rids_a, &rids_b);
	}
	free(buf);
	free(offsets);
	if (numa > 1 && !numa_local_id)
		d->size[numa_node] = numa_size;
	pthread_exit(NULL);
}

inline uint64_t max(uint64_t x, uint64_t y) { return x > y ? x : y; }
inline uint64_t min(uint64_t x, uint64_t y) { return x < y ? x : y; }

int sort(uint32_t **keys, uint32_t **rids, uint64_t *size,
         int threads, int numa, int bits, double fudge,
         uint32_t **keys_buf, uint32_t **rids_buf,
         char **description, uint64_t *times, int interleaved)
{
	int i, j, p, t, n, bits_space[4];
	int bit_passes = distribute_bits(bits, numa, bits_space, 0);
	int threads_per_numa = threads / numa;
	pthread_t *id = malloc(threads * sizeof(pthread_t));
	thread_data_t *data = malloc(threads * sizeof(thread_data_t));
	// check aligned input
	for (i = 0 ; i != numa ; ++i) {
		assert((15 & (uint64_t) keys[i]) == 0);
		assert((15 & (uint64_t) rids[i]) == 0);
	}
	// initialize global barriers
	int local_barriers = 32;
	int global_barriers = 16;
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
	global.max_threads = hardware_threads();
	global.max_numa = numa_max_node() + 1;
	global.bits = bits_space;
	global.threads = threads;
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
		global.sample	  = numa_alloc_interleaved(global.sample_size * sizeof(uint32_t));
		global.sample_buf = numa_alloc_interleaved(global.sample_size * sizeof(uint32_t));
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
	uint64_t ht[] = {0, 0, 0, 0};
	uint64_t pt[] = {0, 0, 0, 0};
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
	description[9] = NULL;
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
	free(id);
	free(global.numa_node);
	free(global.cpu);
	for (i = 0 ; i != numa ; ++i) {
		for (j = 0 ; j != threads_per_numa ; ++j)
			free(global.count[i][j]);
		free(global.count[i]);
	}
	if (numa > 1) {
		numa_free(global.sample, global.sample_size * sizeof(uint32_t));
		numa_free(global.sample_buf, global.sample_size * sizeof(uint32_t));
	}
	for (i = 0 ; i != threads ; ++i)
		free(global.numa_local_count[i]);
	free(global.numa_local_count);
	free(global.count);
	free(data);
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
	uint32_t *keys = &d->keys[numa_node][offset];
	uint32_t *rids = NULL;
	if (d->rids != NULL)
		rids =	&d->rids[numa_node][offset];
	uint32_t *keys_end = &keys[size];
	uint64_t sum = 0;
	uint32_t pkey = 0;
	while (keys != keys_end) {
		uint32_t key = *keys++;
		if (rids) assert(key == *rids++);
		assert(key >= pkey);
		sum += key;
		pkey = key;
	}
	a->checksum = sum;
	pthread_exit(NULL);
}

uint64_t check(uint32_t **keys, uint32_t **rids, uint64_t *size, int numa, int same_key_payload)
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
	global.rids = same_key_payload ? rids : NULL;
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

uint64_t read_from_file(uint32_t **keys, uint64_t *size, int numa, const char *name)
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
		uint32_t *key = keys[n];
		while (rem_size) {
			uint64_t size = 4096;
			if (size > rem_size) size = rem_size;
			size = fread(key, sizeof(uint32_t), size, fp);
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
	assert(p == total_size * sizeof(uint32_t));
	fclose(fp);
	return checksum;
}

int main(int argc, char **argv)
{
	int r, i, max_threads = hardware_threads();
	int max_numa = numa_max_node() + 1;
	uint64_t tuples = argc > 1 ? atoi(argv[1]) : 1000;
	tuples *= 1000000;
	int threads = argc > 2 ? atoi(argv[2]) : max_threads;
	int numa = argc > 3 ? atoi(argv[3]) : max_numa;
	int bits = argc > 4 ? atoi(argv[4]) : 32;
	int interleaved = argc > 5 ? atoi(argv[5]) : 0;
	int allocated = argc > 6 ? atoi(argv[6]) : 1;
	assert(numa > 0 && numa <= 8);
	assert(threads >= numa && threads % numa == 0);
	assert(bits > 0 && bits <= 32);
	char *name = NULL;
	double theta = 0.0;
	if (argc > 7) {
		assert(bits == 32);
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
	double fudge = theta != 0.0 ? 2.0 : 1.05;
	uint64_t tuples_per_numa = tuples / numa;
	uint64_t capacity_per_numa = tuples_per_numa * fudge;
	uint32_t *keys[numa], *keys_buf[numa];
	uint32_t *rids[numa], *rids_buf[numa];
	uint64_t size[numa], cap[numa];
	uint32_t seed = micro_time();
	srand(seed);
	fprintf(stderr, "Tuples: %.2f mil. (%.1f GB)\n", tuples / 1000000.0,
			(tuples * 8.0) / (1024 * 1024 * 1024));
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
		sum_k = init_32(keys, size, cap, threads, numa, bits, 0.0, 0, interleaved);
		srand(t);
		sum_v = init_32(rids, size, cap, threads, numa, bits, 0.0, 0, interleaved);
		assert(sum_k == sum_v);
	} else {
		uint64_t ranks[33];
		init_32(keys, size, cap, threads, numa, 0, 0.0, 0, interleaved);
		if (name != NULL) {
			fprintf(stderr, "Opening file: %s\n", name);
			sum_k = read_from_file(keys, size, numa, name);
		} else {
			uint32_t *values = NULL;
			uint64_t size_32_bit = ((uint64_t) 1) << 32;
			fprintf(stderr, "Generating all 32-bit items\n");
			sum_k = init_32(&values, &size_32_bit, &size_32_bit, threads, 1, -1, 0.0, 0, 1);
			assert(sum_k == size_32_bit * (size_32_bit - 1) / 2);
			fprintf(stderr, "Shuffling all 32-bit items\n");
			shuffle_32(values, size_32_bit);
			fprintf(stderr, "Generating Zipf with theta = %.2f\n", theta);
			sum_k = zipf_32(keys, size, values, numa, theta, ranks);
			numa_free(values, size_32_bit * sizeof(uint32_t));
		}
		same_key_payload = 0;
		sum_v = init_32(rids, size, cap, threads, numa, 32, 0.0, 0, interleaved);
	}
	if (allocated) {
		init_32(keys_buf, size, cap, threads, numa, 0, 0.0, 0, interleaved);
		init_32(rids_buf, size, cap, threads, numa, 0, 0.0, 0, interleaved);
	}
	t = micro_time() - t;
	fprintf(stderr, "Generation time: %ld us\n", t);
	fprintf(stderr, "Generation rate: %.1f mrps\n", tuples * 1.0 / t);
	// sort info
	char *desc[12];
	uint64_t times[12];
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
	int bits_space[4];
	int bit_passes = distribute_bits(bits, numa, bits_space, 1);
	// print sort times
	fprintf(stderr, "Sort time: %ld us\n", t);
	double gigs = (tuples * 8.0) / (1024 * 1024 * 1024);
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
	// show allocation per NUMA node
	for (i = 0 ; i != numa ; ++i)
		fprintf(stderr, "Node %d:%6.2f%%\n", i, size[i] * 100.0 / tuples);
	// check sort order and sum
	if (r) fprintf(stderr, "Destination changed\n");
	else fprintf(stderr, "Destination remained the same\n");
	uint32_t **keys_out = r ? keys_buf : keys;
	uint32_t **rids_out = r ? rids_buf : rids;
	uint64_t checksum = check(keys_out, rids_out, size, numa, same_key_payload);
	assert(checksum == sum_k);
	// free sort data
	for (i = 0 ; i != numa ; ++i)
		if (interleaved) {
			numa_free(keys_buf[i], cap[i] * sizeof(uint32_t));
			numa_free(rids_buf[i], cap[i] * sizeof(uint32_t));
			numa_free(keys[i], cap[i] * sizeof(uint32_t));
			numa_free(rids[i], cap[i] * sizeof(uint32_t));
		} else {
			free(keys_buf[i]);
			free(rids_buf[i]);
			free(keys[i]);
			free(rids[i]);
		}
	printf("%.1f mrps (%.2f GB / sec)\n", tuples * 1.0 / t, (gigs * 1000000) / t);
	if (threads != max_threads) {
		int cpu[threads];
		schedule_threads(cpu, NULL, threads, numa);
		//for (t = 0 ; t != threads ; ++t)
			//fprintf(stderr, "Thread %2ld -> %2d\n", t, cpu[t]);
	}
	return EXIT_SUCCESS;
}
