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
	//return 8;
}

// void cpu_bind(int cpu_id)
// {
// 	cpu_set_t cpu_set;
// 	CPU_ZERO(&cpu_set);
// 	CPU_SET(cpu_id, &cpu_set);
// 	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);
// }

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
	// int newCore = cpu_id;
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

void *mamalloc(size_t size)
{
	void *ptr = NULL;
	return posix_memalign(&ptr, 64, size) ? NULL : ptr;
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
			//assert(i != max_threads);
			thread_numa[i] = -1;
			cpu[t] = i;
			if (numa_node != NULL)
				numa_node[t] = n;
			//assert(numa_node_of_cpu(i) == n);
		}
		free(thread_numa);
	}
}

void decide_partitions(uint64_t size, uint64_t part[2], int numa, int print)
{
	//uint64_t cache = 1500;
	uint64_t cache = 1500000;
	uint64_t fanout[4] = {1, 360, 1000, 1800};
	uint64_t i, j = 0;
	for (i = 1 ; i <= 3 ; ++i)
		if (fanout[i] * cache >= size && fanout[i] >= numa)
			goto done;
	for (i = 1 ; i <= 3 ; ++i)
		for (j = 1 ; j <= i ; ++j)
			if (fanout[i] * fanout[j] * cache >= size && fanout[i] >= numa)
				goto done;
	i = j = 3;
done:
	i = fanout[i];
	j = fanout[j];
	if (part != NULL) {
		part[0] = i;
		part[1] = j;
	}
	if (!print) return;
	if (j == 1)
		fprintf(stderr, " -> x %ld -> ~ %ld\n", i, size / i);
	else
		fprintf(stderr, " -> x %ld -> x %ld -> ~ %ld\n", i, j, size / (i * j));
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

inline uint64_t bsf(uint64_t i)
{
	uint64_t o;
	asm("bsf	%1, %0" : "=r"(o) : "r"(i) : "cc");
	return o;
}

inline uint64_t _mm_mask_epi8(__m128i x)
{
	uint64_t r;
	asm("pmovmskb	%1, %0" : "=r"(r) : "x"(x));
	return r;
}

inline uint64_t _mm_mask_epi32(__m128i x)
{
	uint64_t r;
	asm("movmskps	%1, %0" : "=r"(r) : "x"(x));
	return r;
}

inline uint64_t _mm_mask_epi64(__m128i x)
{
	uint64_t r;
	asm("movmskpd	%1, %0" : "=r"(r) : "x"(x));
	return r;
}

inline uint64_t binary_search(uint64_t *keys, uint64_t size, uint64_t key)
{
	uint64_t low = 0;
	uint64_t high = size;
	while (low < high) {
		uint64_t mid = (low + high) >> 1;
		if (key > keys[mid])
			low = mid + 1;
		else
			high = mid;
	}
	return low;
}

void scalar_combsort_keys(uint64_t *keys, uint64_t size)
{
	const float shrink = 0.77;
	uint64_t gap = size * shrink;
	for (;;) {
		uint64_t i = 0, j = gap, done = 1;
		do {
			uint64_t ki = keys[i];
			uint64_t kj = keys[j];
			if (ki > kj) {
				keys[i] = kj;
				keys[j] = ki;
				done = 0;
			}
			i++; j++;
		} while (j != size);
		if (gap > 1) gap *= shrink;
		else if (done) break;
	}
}

inline void copy(int64_t *dst, const int64_t *src, uint64_t size)
{
	if (size == 0) return;
	const int64_t *src_end = &src[size];
	do {
		_mm_stream_si64((long long *) dst++, *src++);
	} while (src != src_end);
}

inline void insertsort(uint64_t *keys, uint64_t *rids, uint64_t size)
{
	if (size <= 1) return;
	uint64_t prev_key = keys[0];
	uint64_t i = 1;
	do {
		uint64_t next_key = keys[i];
		if (next_key >= prev_key)
			prev_key = next_key;
		else {
			uint64_t next_rid = rids[i];
			uint64_t temp_key = prev_key;
			uint64_t j = i - 1;
			do {
				rids[j + 1] = rids[j];
				keys[j + 1] = temp_key;
				if (j-- == 0) break;
				temp_key = keys[j];
			} while (next_key < temp_key);
			keys[j + 1] = next_key;
			rids[j + 1] = next_rid;
		}
	} while (++i != size);
}


void simd_combsort(uint64_t *keys, uint64_t *rids, uint64_t size,
                   uint64_t *keys_out, uint64_t *rids_out)
{
    if (size <= 9) {
        insertsort(keys, rids, size);
        memcpy(keys_out, keys, size * sizeof(uint64_t));
        memcpy(rids_out, rids, size * sizeof(uint64_t));
        return;
    }
    assert(size <= 0x7FFFFFF0);

    const double shrink = 0.77;
    uint64_t unaligned = (15 & (uint64_t) keys) >> 3;
    uint64_t beg = unaligned == 0 ? 0 : 2 - unaligned;
    uint64_t end = (size - beg) & 3;
    uint64_t gap = ((size - beg - end) >> 2) * shrink;
    uint64_t size_middle = size - end;

    __m128i mask_I = _mm_set1_epi32(-1);
    __m128i mask_C = _mm_slli_epi64(mask_I, 63);

    for (;;) {
        uint64_t c, i = beg, j = (gap << 2);
        __m128i swap = _mm_setzero_si128();

        if (beg) {
            uint64_t ki = keys[0];
            uint64_t kj = keys[j];
            if (ki > kj) {
                swap = mask_C;
                uint64_t r  = rids[j];
                rids[j] = rids[0];
                rids[0] = r;
                keys[0] = kj;
                keys[j] = ki;
            }
        }
        j += beg;

        do {
            _mm_prefetch((char*)&keys[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&rids[i + 32], _MM_HINT_T0);

            __m128i ki_L = _mm_load_si128((__m128i*)&keys[i]);
            __m128i ki_H = _mm_load_si128((__m128i*)&keys[i + 2]);
            __m128i kj_L = _mm_load_si128((__m128i*)&keys[j]);
            __m128i kj_H = _mm_load_si128((__m128i*)&keys[j + 2]);
            __m128i vi_L = _mm_load_si128((__m128i*)&rids[i]);
            __m128i vi_H = _mm_load_si128((__m128i*)&rids[i + 2]);
            __m128i vj_L = _mm_load_si128((__m128i*)&rids[j]);
            __m128i vj_H = _mm_load_si128((__m128i*)&rids[j + 2]);

            __m128i ki_LC = _mm_sub_epi64(ki_L, mask_C);
            __m128i ki_HC = _mm_sub_epi64(ki_H, mask_C);
            __m128i kj_LC = _mm_sub_epi64(kj_L, mask_C);
            __m128i kj_HC = _mm_sub_epi64(kj_H, mask_C);

            __m128i cmp_L = _mm_cmpgt_epi64(ki_LC, kj_LC);
            __m128i cmp_H = _mm_cmpgt_epi64(ki_HC, kj_HC);

            __m128i k_min_L = _mm_blendv_epi8(ki_L, kj_L, cmp_L);
            __m128i k_max_L = _mm_blendv_epi8(kj_L, ki_L, cmp_L);
            __m128i k_min_H = _mm_blendv_epi8(ki_H, kj_H, cmp_H);
            __m128i k_max_H = _mm_blendv_epi8(kj_H, ki_H, cmp_H);

            __m128i v_min_L = _mm_blendv_epi8(vi_L, vj_L, cmp_L);
            __m128i v_max_L = _mm_blendv_epi8(vj_L, vi_L, cmp_L);
            __m128i v_min_H = _mm_blendv_epi8(vi_H, vj_H, cmp_H);
            __m128i v_max_H = _mm_blendv_epi8(vj_H, vi_H, cmp_H);

            swap = _mm_or_si128(swap, cmp_L);
            swap = _mm_or_si128(swap, cmp_H);

            _mm_store_si128((__m128i*)&keys[i], k_min_L);
            _mm_store_si128((__m128i*)&keys[i + 2], k_min_H);
            _mm_store_si128((__m128i*)&keys[j], k_max_L);
            _mm_store_si128((__m128i*)&keys[j + 2], k_max_H);
            _mm_store_si128((__m128i*)&rids[i], v_min_L);
            _mm_store_si128((__m128i*)&rids[i + 2], v_min_H);
            _mm_store_si128((__m128i*)&rids[j], v_max_L);
            _mm_store_si128((__m128i*)&rids[j + 2], v_max_H);

            i += 4;
            j += 4;
        } while (j != size_middle);

        for (c = 0; c != end; ++c) {
            uint64_t ki = keys[c + i];
            uint64_t kj = keys[c + j];
            if (ki > kj) {
                swap = mask_C;
                uint64_t r  = rids[c + j];
                rids[c + j] = rids[c + i];
                rids[c + i] = r;
                keys[c + i] = kj;
                keys[c + j] = ki;
            }
        }

        if (gap > 1) {
            gap *= shrink;
        } else if (_mm_testz_si128(swap, swap)) {
            break;
        }
    }

    __m128i mask_7654 = _mm_cvtsi64_si128(0x07060504);
    __m128i mask_size = _mm_cvtsi64_si128(size - 1);
    mask_7654 = _mm_cvtepi8_epi32(mask_7654);
    mask_size = _mm_shuffle_epi32(mask_size, 0);
    __m128i mask_4 = _mm_shuffle_epi32(mask_7654, 0);

    __m128i key_L = _mm_loadu_si128((__m128i*)&keys[0]);
    __m128i key_H = _mm_loadu_si128((__m128i*)&keys[2]);
    __m128i rid_L = _mm_loadu_si128((__m128i*)&rids[0]);
    __m128i rid_H = _mm_loadu_si128((__m128i*)&rids[2]);
    __m128i loc = _mm_sub_epi32(mask_7654, mask_4);

    uint64_t *keys_end = &keys_out[size];
    uint64_t i = 0;
    do {
        __m128i key_LC = _mm_sub_epi64(key_L, mask_C);
        __m128i key_HC = _mm_sub_epi64(key_H, mask_C);
        __m128i m_cmp = _mm_cmpgt_epi64(key_LC, key_HC);
        __m128i m_key = _mm_blendv_epi8(key_LC, key_HC, m_cmp);
        __m128i t_key = _mm_shuffle_epi32(m_key, _MM_SHUFFLE(1, 0, 3, 2));
        m_cmp = _mm_cmpgt_epi64(m_key, t_key);
        m_key = _mm_blendv_epi8(m_key, t_key, m_cmp);
        m_key = _mm_add_epi64(m_key, mask_C);

        __m128i m_loc_L = _mm_cmpeq_epi64(m_key, key_L);
        __m128i m_loc_H = _mm_cmpeq_epi64(m_key, key_H);
        __m128i m_loc = _mm_packs_epi32(m_loc_L, m_loc_H);
        m_loc = _mm_xor_si128(m_loc, mask_I);
        m_loc = _mm_or_si128(m_loc, loc);
        __m128i t_loc = _mm_shuffle_epi32(m_loc, _MM_SHUFFLE(2, 3, 0, 1));
        m_loc = _mm_min_epu32(m_loc, t_loc);
        t_loc = _mm_shuffle_epi32(m_loc, _MM_SHUFFLE(1, 0, 3, 2));
        m_loc = _mm_min_epu32(m_loc, t_loc);

        __m128i m_pos = _mm_cmpeq_epi32(m_loc, loc);
        m_loc = _mm_add_epi32(m_loc, mask_4);

        __m128i m_pos_L = _mm_shuffle_epi32(m_pos, _MM_SHUFFLE(1, 1, 0, 0));
        __m128i m_pos_H = _mm_shuffle_epi32(m_pos, _MM_SHUFFLE(3, 3, 2, 2));
        __m128i m_rid_L = _mm_and_si128(rid_L, m_pos_L);
        __m128i m_rid_H = _mm_and_si128(rid_H, m_pos_H);
        __m128i m_rid = _mm_or_si128(m_rid_L, m_rid_H);
        __m128i t_rid = _mm_shuffle_epi32(m_rid, _MM_SHUFFLE(1, 0, 3, 2));
        m_rid = _mm_or_si128(m_rid, t_rid);

        asm("movd %1, %%eax" : "=a"(i) : "x"(m_loc), "0"(i));
        __m128i n_key = _mm_loadl_epi64((__m128i*)&keys[i]);
        __m128i n_rid = _mm_loadl_epi64((__m128i*)&rids[i]);

        n_key = _mm_unpacklo_epi64(n_key, n_key);
        n_rid = _mm_unpacklo_epi64(n_rid, n_rid);
        loc = _mm_blendv_epi8(loc, m_loc, m_pos);
        key_L = _mm_blendv_epi8(key_L, n_key, m_pos_L);
        key_H = _mm_blendv_epi8(key_H, n_key, m_pos_H);
        rid_L = _mm_blendv_epi8(rid_L, n_rid, m_pos_L);
        rid_H = _mm_blendv_epi8(rid_H, n_rid, m_pos_H);

        __m128i inv = _mm_cmpgt_epi32(loc, mask_size);
        __m128i inv_L = _mm_shuffle_epi32(inv, _MM_SHUFFLE(1, 1, 0, 0));
        __m128i inv_H = _mm_shuffle_epi32(inv, _MM_SHUFFLE(3, 3, 2, 2));
        key_L = _mm_or_si128(key_L, inv_L);
        key_H = _mm_or_si128(key_H, inv_H);

        uint64_t k, r;
        asm("movq %1, %0" : "=r"(k) : "x"(m_key));
        asm("movq %1, %0" : "=r"(r) : "x"(m_rid));
        _mm_stream_si64((long long *)keys_out++, k);
        _mm_stream_si64((long long *)rids_out++, r);
    } while (keys_out != keys_end);
}

void simd_combsort_old(uint64_t *keys, uint64_t *rids, uint64_t size,
                   uint64_t *keys_out, uint64_t *rids_out)
{
	if (size <= 9) {
		insertsort(keys, rids, size);
		copy(keys_out, keys, size);
		copy(rids_out, rids, size);
		return;
	}
	assert(size <= 0x7FFFFFF0);
	// comb sort loop (4-wide)
	const double shrink = 0.77;
	uint64_t unaligned = (15 & (uint64_t) keys) >> 3;
	uint64_t beg = unaligned == 0 ? 0 : 2 - unaligned;
	uint64_t end = (size - beg) & 3;
	uint64_t gap = ((size - beg - end) >> 2) * shrink;
	uint64_t size_middle = size - end;
	__m128i mask_I = _mm_cmpeq_epi32(mask_I, mask_I);
	__m128i mask_C = _mm_slli_epi64(mask_I, 63);
	for (;;) {
		uint64_t c, i = beg, j = (gap << 2);
		__m128i swap = _mm_xor_si128(swap, swap);
		// beginning
		if (beg) {
			uint64_t ki = keys[0];
			uint64_t kj = keys[j];
			if (ki > kj) {
				swap = mask_C;
				uint64_t r  = rids[j];
				rids[j] = rids[0];
				rids[0] = r;
				keys[0] = kj;
				keys[j] = ki;
			}
		}
		j += beg;
		// middle case
		do {
			__m128i ki_L = _mm_load_si128((__m128i*) &keys[i]);
			__m128i ki_H = _mm_load_si128((__m128i*) &keys[i + 2]);
			__m128i kj_L = _mm_load_si128((__m128i*) &keys[j]);
			__m128i kj_H = _mm_load_si128((__m128i*) &keys[j + 2]);
			__m128i vi_L = _mm_load_si128((__m128i*) &rids[i]);
			__m128i vi_H = _mm_load_si128((__m128i*) &rids[i + 2]);
			__m128i vj_L = _mm_load_si128((__m128i*) &rids[j]);
			__m128i vj_H = _mm_load_si128((__m128i*) &rids[j + 2]);
			__m128i ki_LC = _mm_sub_epi64(ki_L, mask_C);
			__m128i ki_HC = _mm_sub_epi64(ki_H, mask_C);
			__m128i kj_LC = _mm_sub_epi64(kj_L, mask_C);
			__m128i kj_HC = _mm_sub_epi64(kj_H, mask_C);
			__m128i cmp_L = _mm_cmpgt_epi64(ki_LC, kj_LC);
			__m128i cmp_H = _mm_cmpgt_epi64(ki_HC, kj_HC);
			__m128i k_min_L = _mm_blendv_epi8(ki_L, kj_L, cmp_L);
			__m128i k_max_L = _mm_blendv_epi8(kj_L, ki_L, cmp_L);
			__m128i k_min_H = _mm_blendv_epi8(ki_H, kj_H, cmp_H);
			__m128i k_max_H = _mm_blendv_epi8(kj_H, ki_H, cmp_H);
			__m128i v_min_L = _mm_blendv_epi8(vi_L, vj_L, cmp_L);
			__m128i v_max_L = _mm_blendv_epi8(vj_L, vi_L, cmp_L);
			__m128i v_min_H = _mm_blendv_epi8(vi_H, vj_H, cmp_H);
			__m128i v_max_H = _mm_blendv_epi8(vj_H, vi_H, cmp_H);
			swap = _mm_or_si128(swap, cmp_L);
			swap = _mm_or_si128(swap, cmp_H);
			_mm_store_si128((__m128i*) &keys[i], k_min_L);
			_mm_store_si128((__m128i*) &keys[i + 2], k_min_H);
			_mm_store_si128((__m128i*) &keys[j], k_max_L);
			_mm_store_si128((__m128i*) &keys[j + 2], k_max_H);
			_mm_store_si128((__m128i*) &rids[i], v_min_L);
			_mm_store_si128((__m128i*) &rids[i + 2], v_min_H);
			_mm_store_si128((__m128i*) &rids[j], v_max_L);
			_mm_store_si128((__m128i*) &rids[j + 2], v_max_H);
			i += 4;
			j += 4;
		} while (j != size_middle);
		// end
		for (c = 0 ; c != end ; ++c) {
			uint64_t ki = keys[c + i];
			uint64_t kj = keys[c + j];
			if (ki > kj) {
				swap = mask_C;
				uint64_t r  = rids[c + j];
				rids[c + j] = rids[c + i];
				rids[c + i] = r;
				keys[c + i] = kj;
				keys[c + j] = ki;
			}
		}
		if (gap > 1) gap *= shrink;
		else if (_mm_testz_si128(swap, swap)) break;
	}
	// create masks for merging
	__m128i mask_7654 = _mm_cvtsi64_si128(0x07060504);
	__m128i mask_size = _mm_cvtsi64_si128(size - 1);
	mask_7654 = _mm_cvtepi8_epi32(mask_7654);
	mask_size = _mm_shuffle_epi32(mask_size, 0);
	__m128i mask_4 = _mm_shuffle_epi32(mask_7654, 0);
	// initial key, rid, location
	__m128i key_L = _mm_loadu_si128((__m128i*) &keys[0]);
	__m128i key_H = _mm_loadu_si128((__m128i*) &keys[2]);
	__m128i rid_L = _mm_loadu_si128((__m128i*) &rids[0]);
	__m128i rid_H = _mm_loadu_si128((__m128i*) &rids[2]);
	__m128i loc = _mm_sub_epi32(mask_7654, mask_4);
	// merging loop
	uint64_t *keys_end = &keys_out[size];
	uint64_t i = 0;
	do {
		// get leftmost and valid minimum key
		__m128i key_LC = _mm_sub_epi64(key_L, mask_C);
		__m128i key_HC = _mm_sub_epi64(key_H, mask_C);
		__m128i m_cmp = _mm_cmpgt_epi64(key_LC, key_HC);
		__m128i m_key = _mm_blendv_epi8(key_LC, key_HC, m_cmp);
		__m128i t_key = _mm_shuffle_epi32(m_key, _MM_SHUFFLE(1, 0, 3, 2));
		m_cmp = _mm_cmpgt_epi64(m_key, t_key);
		m_key = _mm_blendv_epi8(m_key, t_key, m_cmp);
		m_key = _mm_add_epi64(m_key, mask_C);
		// get min index
		__m128i m_loc_L = _mm_cmpeq_epi64(m_key, key_L);
		__m128i m_loc_H = _mm_cmpeq_epi64(m_key, key_H);
		__m128i m_loc = _mm_packs_epi32(m_loc_L, m_loc_H);
		m_loc = _mm_xor_si128(m_loc, mask_I);
		m_loc = _mm_or_si128(m_loc, loc);
		__m128i t_loc = _mm_shuffle_epi32(m_loc, _MM_SHUFFLE(2, 3, 0, 1));
		m_loc = _mm_min_epu32(m_loc, t_loc);
		t_loc = _mm_shuffle_epi32(m_loc, _MM_SHUFFLE(1, 0, 3, 2));
		m_loc = _mm_min_epu32(m_loc, t_loc);
		// get (unique) min location
		__m128i m_pos = _mm_cmpeq_epi32(m_loc, loc);
		m_loc = _mm_add_epi32(m_loc, mask_4);
		// get min rid
		__m128i m_pos_L = _mm_shuffle_epi32(m_pos, _MM_SHUFFLE(1, 1, 0, 0));
		__m128i m_pos_H = _mm_shuffle_epi32(m_pos, _MM_SHUFFLE(3, 3, 2, 2));
		__m128i m_rid_L = _mm_and_si128(rid_L, m_pos_L);
		__m128i m_rid_H = _mm_and_si128(rid_H, m_pos_H);
		__m128i m_rid = _mm_or_si128(m_rid_L, m_rid_H);
		__m128i t_rid = _mm_shuffle_epi32(m_rid, _MM_SHUFFLE(1, 0, 3, 2));
		m_rid = _mm_or_si128(m_rid, t_rid);
		// get index of minimum key and load new pair
		asm("movd	%1,	%%eax" : "=a"(i) : "x"(m_loc), "0"(i));
		__m128i n_key = _mm_loadl_epi64((__m128i*) &keys[i]);
		__m128i n_rid = _mm_loadl_epi64((__m128i*) &rids[i]);
		// insert new pair and reset keys
		n_key = _mm_unpacklo_epi64(n_key, n_key);
		n_rid = _mm_unpacklo_epi64(n_rid, n_rid);
		loc = _mm_blendv_epi8(loc, m_loc, m_pos);
		key_L = _mm_blendv_epi8(key_L, n_key, m_pos_L);
		key_H = _mm_blendv_epi8(key_H, n_key, m_pos_H);
		rid_L = _mm_blendv_epi8(rid_L, n_rid, m_pos_L);
		rid_H = _mm_blendv_epi8(rid_H, n_rid, m_pos_H);
		// invalidate keys that exceed size
		__m128i inv = _mm_cmpgt_epi32(loc, mask_size);
		__m128i inv_L = _mm_shuffle_epi32(inv, _MM_SHUFFLE(1, 1, 0, 0));
		__m128i inv_H = _mm_shuffle_epi32(inv, _MM_SHUFFLE(3, 3, 2, 2));
		key_L = _mm_or_si128(key_L, inv_L);
		key_H = _mm_or_si128(key_H, inv_H);
		// stream minimum key and rid
		uint64_t k, r;
		asm("movq	%1,	%0" : "=r"(k) : "x"(m_key));
		asm("movq	%1, %0" : "=r"(r) : "x"(m_rid));
		_mm_stream_si64((long long *) keys_out++, k);
		_mm_stream_si64((long long *) rids_out++, r);
	} while (keys_out != keys_end);
}

inline __m128i histogram_root(__m128i k1, __m128i k2, __m128i k3, __m128i k4,
                              __m128i del_1, __m128i del_2, __m128i del_3, __m128i del_4,
                              __m128i del_5, __m128i del_6, __m128i del_7)
{	// L1
	__m128i e1_L1 = _mm_cmpgt_epi64(k1, del_4);
	__m128i e2_L1 = _mm_cmpgt_epi64(k2, del_4);
	__m128i e3_L1 = _mm_cmpgt_epi64(k3, del_4);
	__m128i e4_L1 = _mm_cmpgt_epi64(k4, del_4);
	__m128i e12_L1 = _mm_packs_epi32(e1_L1, e2_L1);
	__m128i e34_L1 = _mm_packs_epi32(e3_L1, e4_L1);
	__m128i e_L1 = _mm_packs_epi32(e12_L1, e34_L1);
	// 2-6
	__m128i d1_26 = _mm_blendv_epi8(del_2, del_6, e1_L1);
	__m128i d2_26 = _mm_blendv_epi8(del_2, del_6, e2_L1);
	__m128i d3_26 = _mm_blendv_epi8(del_2, del_6, e3_L1);
	__m128i d4_26 = _mm_blendv_epi8(del_2, del_6, e4_L1);
	// 1-5
	__m128i d1_15 = _mm_blendv_epi8(del_1, del_5, e1_L1);
	__m128i d2_15 = _mm_blendv_epi8(del_1, del_5, e2_L1);
	__m128i d3_15 = _mm_blendv_epi8(del_1, del_5, e3_L1);
	__m128i d4_15 = _mm_blendv_epi8(del_1, del_5, e4_L1);
	// 3-7
	__m128i d1_37 = _mm_blendv_epi8(del_3, del_7, e1_L1);
	__m128i d2_37 = _mm_blendv_epi8(del_3, del_7, e2_L1);
	__m128i d3_37 = _mm_blendv_epi8(del_3, del_7, e3_L1);
	__m128i d4_37 = _mm_blendv_epi8(del_3, del_7, e4_L1);
	// L2
	__m128i e1_L2 = _mm_cmpgt_epi64(k1, d1_26);
	__m128i e2_L2 = _mm_cmpgt_epi64(k2, d2_26);
	__m128i e3_L2 = _mm_cmpgt_epi64(k3, d3_26);
	__m128i e4_L2 = _mm_cmpgt_epi64(k4, d4_26);
	__m128i e12_L2 = _mm_packs_epi32(e1_L2, e2_L2);
	__m128i e34_L2 = _mm_packs_epi32(e3_L2, e4_L2);
	__m128i e_L2 = _mm_packs_epi32(e12_L2, e34_L2);
	// 15-37
	__m128i d1_1357 = _mm_blendv_epi8(d1_15, d1_37, e1_L2);
	__m128i d2_1357 = _mm_blendv_epi8(d2_15, d2_37, e2_L2);
	__m128i d3_1357 = _mm_blendv_epi8(d3_15, d3_37, e3_L2);
	__m128i d4_1357 = _mm_blendv_epi8(d4_15, d4_37, e4_L2);
	// L3
	__m128i e1_L3 = _mm_cmpgt_epi64(k1, d1_1357);
	__m128i e2_L3 = _mm_cmpgt_epi64(k2, d2_1357);
	__m128i e3_L3 = _mm_cmpgt_epi64(k3, d3_1357);
	__m128i e4_L3 = _mm_cmpgt_epi64(k4, d4_1357);
	__m128i e12_L3 = _mm_packs_epi32(e1_L3, e2_L3);
	__m128i e34_L3 = _mm_packs_epi32(e3_L3, e4_L3);
	__m128i e_L3 = _mm_packs_epi32(e12_L3, e34_L3);
	// combine
	__m128i r = _mm_xor_si128(r, r);
	r = _mm_sub_epi16(r, e_L1);
	r = _mm_add_epi16(r, r);
	r = _mm_sub_epi16(r, e_L2);
	r = _mm_add_epi16(r, r);
	r = _mm_sub_epi16(r, e_L3);
	return r;
}

inline void histogram_8_3_3_5_part(__m128i k12, __m128i k34, __m128i r,
				   int64_t *index_L1, int64_t *index_L2, int64_t *index_L3,
				   uint64_t *count, uint64_t *ranges)
{
	__m128i r_s1 = _mm_shuffle_epi32(r, 1);
	__m128i r_s2 = _mm_shuffle_epi32(r, 2);
	__m128i r_s3 = _mm_shuffle_epi32(r, 3);
	uint64_t p1 = 0, p2 = 0, p3 = 0, p4 = 0;
	asm("movd	%1, %%eax" : "=a"(p1) : "x"(r),    "0"(p1));
	asm("movd	%1, %%ebx" : "=b"(p2) : "x"(r_s1), "0"(p2));
	asm("movd	%1, %%ecx" : "=c"(p3) : "x"(r_s2), "0"(p3));
	asm("movd	%1, %%edx" : "=d"(p4) : "x"(r_s3), "0"(p4));
	uint64_t q1 = p1 << 1;
	uint64_t q2 = p2 << 1;
	uint64_t q3 = p3 << 1;
	uint64_t q4 = p4 << 1;
	__m128i d1 = _mm_load_si128((__m128i*) &index_L1[q1]);
	__m128i d2 = _mm_load_si128((__m128i*) &index_L1[q2]);
	__m128i d3 = _mm_load_si128((__m128i*) &index_L1[q3]);
	__m128i d4 = _mm_load_si128((__m128i*) &index_L1[q4]);
	__m128i k1_x2 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(1, 0, 1, 0));
	__m128i k2_x2 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 2, 3, 2));
	__m128i k3_x2 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(1, 0, 1, 0));
	__m128i k4_x2 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 2, 3, 2));
	d1 = _mm_cmpgt_epi64(k1_x2, d1);
	d2 = _mm_cmpgt_epi64(k2_x2, d2);
	d3 = _mm_cmpgt_epi64(k3_x2, d3);
	d4 = _mm_cmpgt_epi64(k4_x2, d4);
	q1 += p1;
	q2 += p2;
	q3 += p3;
	q4 += p4;
	p1 = _mm_mask_epi64(d1);
	p2 = _mm_mask_epi64(d2);
	p3 = _mm_mask_epi64(d3);
	p4 = _mm_mask_epi64(d4);
	p1 ^= 7;
	p2 ^= 7;
	p3 ^= 7;
	p4 ^= 7;
	p1 = bsf(p1);
	p2 = bsf(p2);
	p3 = bsf(p3);
	p4 = bsf(p4);
	p1 += q1;
	p2 += q2;
	p3 += q3;
	p4 += q4;
	q1 = p1 << 1;
	q2 = p2 << 1;
	q3 = p3 << 1;
	q4 = p4 << 1;
	d1 = _mm_load_si128((__m128i*) &index_L2[q1]);
	d2 = _mm_load_si128((__m128i*) &index_L2[q2]);
	d3 = _mm_load_si128((__m128i*) &index_L2[q3]);
	d4 = _mm_load_si128((__m128i*) &index_L2[q4]);
	d1 = _mm_cmpgt_epi64(k1_x2, d1);
	d2 = _mm_cmpgt_epi64(k2_x2, d2);
	d3 = _mm_cmpgt_epi64(k3_x2, d3);
	d4 = _mm_cmpgt_epi64(k4_x2, d4);
	q1 += p1;
	q2 += p2;
	q3 += p3;
	q4 += p4;
	p1 = _mm_mask_epi64(d1);
	p2 = _mm_mask_epi64(d2);
	p3 = _mm_mask_epi64(d3);
	p4 = _mm_mask_epi64(d4);
	p1 ^= 7;
	p2 ^= 7;
	p3 ^= 7;
	p4 ^= 7;
	p1 = bsf(p1);
	p2 = bsf(p2);
	p3 = bsf(p3);
	p4 = bsf(p4);
	p1 += q1;
	p2 += q2;
	p3 += q3;
	p4 += q4;
	q1 = p1 << 2;
	q2 = p2 << 2;
	q3 = p3 << 2;
	q4 = p4 << 2;
	__m128i d1_L = _mm_load_si128((__m128i*) &index_L3[q1]);
	__m128i d2_L = _mm_load_si128((__m128i*) &index_L3[q2]);
	__m128i d3_L = _mm_load_si128((__m128i*) &index_L3[q3]);
	__m128i d4_L = _mm_load_si128((__m128i*) &index_L3[q4]);
	__m128i d1_H = _mm_load_si128((__m128i*) &index_L3[q1 + 2]);
	__m128i d2_H = _mm_load_si128((__m128i*) &index_L3[q2 + 2]);
	__m128i d3_H = _mm_load_si128((__m128i*) &index_L3[q3 + 2]);
	__m128i d4_H = _mm_load_si128((__m128i*) &index_L3[q4 + 2]);
	d1_L = _mm_cmpgt_epi64(k1_x2, d1_L);
	d2_L = _mm_cmpgt_epi64(k2_x2, d2_L);
	d3_L = _mm_cmpgt_epi64(k3_x2, d3_L);
	d4_L = _mm_cmpgt_epi64(k4_x2, d4_L);
	d1_H = _mm_cmpgt_epi64(k1_x2, d1_H);
	d2_H = _mm_cmpgt_epi64(k2_x2, d2_H);
	d3_H = _mm_cmpgt_epi64(k3_x2, d3_H);
	d4_H = _mm_cmpgt_epi64(k4_x2, d4_H);
	d1 = _mm_packs_epi32(d1_L, d1_H);
	d2 = _mm_packs_epi32(d2_L, d2_H);
	d3 = _mm_packs_epi32(d3_L, d3_H);
	d4 = _mm_packs_epi32(d4_L, d4_H);
	q1 += p1;
	q2 += p2;
	q3 += p3;
	q4 += p4;
	p1 = _mm_mask_epi32(d1);
	p2 = _mm_mask_epi32(d2);
	p3 = _mm_mask_epi32(d3);
	p4 = _mm_mask_epi32(d4);
	p1 ^= 31;
	p2 ^= 31;
	p3 ^= 31;
	p4 ^= 31;
	p1 = bsf(p1);
	p2 = bsf(p2);
	p3 = bsf(p3);
	p4 = bsf(p4);
	p1 += q1;
	p2 += q2;
	p3 += q3;
	p4 += q4;
	count[p1]++;
	count[p2]++;
	count[p3]++;
	count[p4]++;
	p2 <<= 16;
	p3 <<= 32;
	p4 <<= 48;
	_mm_stream_si64((long long int*) ranges, p1 | p2 | p3 | p4);
}

void histogram_360(uint64_t *keys, uint64_t size, uint64_t delim[],
		   uint64_t *count, uint16_t *ranges, uint64_t *index)
{
	uint64_t *keys_end = &keys[size];
	uint64_t sign = 1ull << 63;
	__m128i sign_x2 = _mm_set1_epi64x(sign);
	int64_t *index_L0 = (uint64_t*) index;
	int64_t *index_L1 = &index_L0[8];
	int64_t *index_L2 = &index_L1[16];
	int64_t *index_L3 = &index_L2[48];
	uint64_t rem_1 = 2, rem_2 = 2, rem_3 = 4;
	uint64_t i, i0 = 0, i1 = 0, i2 = 0, i3 = 0;
	for (i = 0 ; i != 359 ; ++i)
		if (rem_3--)
			index_L3[i3++] = delim[i] - sign;
		else if (rem_2--) {
			rem_3 = 4;
			index_L2[i2++] = delim[i] - sign;
		} else if (rem_1--) {
			rem_3 = 4;
			rem_2 = 2;
			index_L1[i1++] = delim[i] - sign;
		} else {
			rem_3 = 4;
			rem_1 = rem_2 = 2;
			index_L0[i0++] = delim[i] - sign;
		}
	assert(i0 == 7);
	assert(i1 == 16);
	assert(i2 == 48);
	assert(i3 == 288);
	__m128i del_1 = _mm_set1_epi64x(index_L0[0]);
	__m128i del_2 = _mm_set1_epi64x(index_L0[1]);
	__m128i del_3 = _mm_set1_epi64x(index_L0[2]);
	__m128i del_4 = _mm_set1_epi64x(index_L0[3]);
	__m128i del_5 = _mm_set1_epi64x(index_L0[4]);
	__m128i del_6 = _mm_set1_epi64x(index_L0[5]);
	__m128i del_7 = _mm_set1_epi64x(index_L0[6]);
	while ((7 & (uint64_t) ranges) != 0 && keys != keys_end) {
		i = binary_search(delim, 359, *keys++);
		count[i]++;
		*ranges++ = i;
	}
	assert((15 & (uint64_t) keys) == 0);
	uint64_t *ranges_64 = (uint64_t*) ranges;
	uint64_t *keys_aligned_end = &keys[(keys_end - keys) & ~7];
	while (keys != keys_aligned_end) {
		// load 8 keys
		__m128i k1 = _mm_load_si128((__m128i*) &keys[0]);
		__m128i k2 = _mm_load_si128((__m128i*) &keys[2]);
		__m128i k3 = _mm_load_si128((__m128i*) &keys[4]);
		__m128i k4 = _mm_load_si128((__m128i*) &keys[6]);
		keys += 8;
		// convert keys
		k1 = _mm_sub_epi64(k1, sign_x2);
		k2 = _mm_sub_epi64(k2, sign_x2);
		k3 = _mm_sub_epi64(k3, sign_x2);
		k4 = _mm_sub_epi64(k4, sign_x2);
		// root comparisons
		__m128i r = histogram_root(k1, k2, k3, k4,
		                           del_1, del_2, del_3,
		                           del_4, del_5, del_6, del_7);
		// open-up comparisons
		__m128i r1 = _mm_cvtepi16_epi32(r);
		r = _mm_shuffle_epi32(r, _MM_SHUFFLE(1, 0, 3, 2));
		__m128i r2 = _mm_cvtepi16_epi32(r);
		// call rest histogram
		histogram_8_3_3_5_part(k1, k2, r1, index_L1, index_L2, index_L3, count, &ranges_64[0]);
		histogram_8_3_3_5_part(k3, k4, r2, index_L1, index_L2, index_L3, count, &ranges_64[1]);
		ranges_64 += 2;
	}
	ranges = (uint16_t*) ranges_64;
	while (keys != keys_end) {
		i = binary_search(delim, 359, *keys++);
		count[i]++;
		*ranges++ = i;
	}
#ifdef BG
	keys -= size;
	ranges -= size;
	for (i = 0 ; i != size ; ++i)
		assert(binary_search(delim, 359, keys[i]) == ranges[i]);
#endif
}

inline void histogram_8_5_5_5_part(__m128i k12, __m128i k34, __m128i r,
				   int64_t *index_L1, int64_t *index_L2,
				   int64_t *index_L3, uint64_t *count, uint64_t *ranges)
{
	__m128i r_s1 = _mm_shuffle_epi32(r, 1);
	__m128i r_s2 = _mm_shuffle_epi32(r, 2);
	__m128i r_s3 = _mm_shuffle_epi32(r, 3);
	uint64_t p1 = 0, p2 = 0, p3 = 0, p4 = 0;
	asm("movd	%1, %%eax" : "=a"(p1) : "x"(r),    "0"(p1));
	asm("movd	%1, %%ebx" : "=b"(p2) : "x"(r_s1), "0"(p2));
	asm("movd	%1, %%ecx" : "=c"(p3) : "x"(r_s2), "0"(p3));
	asm("movd	%1, %%edx" : "=d"(p4) : "x"(r_s3), "0"(p4));
	uint64_t q1 = p1 << 2;
	uint64_t q2 = p2 << 2;
	uint64_t q3 = p3 << 2;
	uint64_t q4 = p4 << 2;
	__m128i d1_L = _mm_load_si128((__m128i*) &index_L1[q1]);
	__m128i d2_L = _mm_load_si128((__m128i*) &index_L1[q2]);
	__m128i d3_L = _mm_load_si128((__m128i*) &index_L1[q3]);
	__m128i d4_L = _mm_load_si128((__m128i*) &index_L1[q4]);
	__m128i d1_H = _mm_load_si128((__m128i*) &index_L1[q1 + 2]);
	__m128i d2_H = _mm_load_si128((__m128i*) &index_L1[q2 + 2]);
	__m128i d3_H = _mm_load_si128((__m128i*) &index_L1[q3 + 2]);
	__m128i d4_H = _mm_load_si128((__m128i*) &index_L1[q4 + 2]);
	__m128i k1_x2 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(1, 0, 1, 0));
	__m128i k2_x2 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 2, 3, 2));
	__m128i k3_x2 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(1, 0, 1, 0));
	__m128i k4_x2 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 2, 3, 2));
	d1_L = _mm_cmpgt_epi64(k1_x2, d1_L);
	d2_L = _mm_cmpgt_epi64(k2_x2, d2_L);
	d3_L = _mm_cmpgt_epi64(k3_x2, d3_L);
	d4_L = _mm_cmpgt_epi64(k4_x2, d4_L);
	d1_H = _mm_cmpgt_epi64(k1_x2, d1_H);
	d2_H = _mm_cmpgt_epi64(k2_x2, d2_H);
	d3_H = _mm_cmpgt_epi64(k3_x2, d3_H);
	d4_H = _mm_cmpgt_epi64(k4_x2, d4_H);
	__m128i d1 = _mm_packs_epi32(d1_L, d1_H);
	__m128i d2 = _mm_packs_epi32(d2_L, d2_H);
	__m128i d3 = _mm_packs_epi32(d3_L, d3_H);
	__m128i d4 = _mm_packs_epi32(d4_L, d4_H);
	q1 += p1;
	q2 += p2;
	q3 += p3;
	q4 += p4;
	p1 = _mm_mask_epi32(d1);
	p2 = _mm_mask_epi32(d2);
	p3 = _mm_mask_epi32(d3);
	p4 = _mm_mask_epi32(d4);
	p1 ^= 31;
	p2 ^= 31;
	p3 ^= 31;
	p4 ^= 31;
	p1 = bsf(p1);
	p2 = bsf(p2);
	p3 = bsf(p3);
	p4 = bsf(p4);
	p1 += q1;
	p2 += q2;
	p3 += q3;
	p4 += q4;
	q1 = p1 << 2;
	q2 = p2 << 2;
	q3 = p3 << 2;
	q4 = p4 << 2;
	d1_L = _mm_load_si128((__m128i*) &index_L2[q1]);
	d2_L = _mm_load_si128((__m128i*) &index_L2[q2]);
	d3_L = _mm_load_si128((__m128i*) &index_L2[q3]);
	d4_L = _mm_load_si128((__m128i*) &index_L2[q4]);
	d1_H = _mm_load_si128((__m128i*) &index_L2[q1 + 2]);
	d2_H = _mm_load_si128((__m128i*) &index_L2[q2 + 2]);
	d3_H = _mm_load_si128((__m128i*) &index_L2[q3 + 2]);
	d4_H = _mm_load_si128((__m128i*) &index_L2[q4 + 2]);
	d1_L = _mm_cmpgt_epi64(k1_x2, d1_L);
	d2_L = _mm_cmpgt_epi64(k2_x2, d2_L);
	d3_L = _mm_cmpgt_epi64(k3_x2, d3_L);
	d4_L = _mm_cmpgt_epi64(k4_x2, d4_L);
	d1_H = _mm_cmpgt_epi64(k1_x2, d1_H);
	d2_H = _mm_cmpgt_epi64(k2_x2, d2_H);
	d3_H = _mm_cmpgt_epi64(k3_x2, d3_H);
	d4_H = _mm_cmpgt_epi64(k4_x2, d4_H);
	d1 = _mm_packs_epi32(d1_L, d1_H);
	d2 = _mm_packs_epi32(d2_L, d2_H);
	d3 = _mm_packs_epi32(d3_L, d3_H);
	d4 = _mm_packs_epi32(d4_L, d4_H);
	q1 += p1;
	q2 += p2;
	q3 += p3;
	q4 += p4;
	p1 = _mm_mask_epi32(d1);
	p2 = _mm_mask_epi32(d2);
	p3 = _mm_mask_epi32(d3);
	p4 = _mm_mask_epi32(d4);
	p1 ^= 31;
	p2 ^= 31;
	p3 ^= 31;
	p4 ^= 31;
	p1 = bsf(p1);
	p2 = bsf(p2);
	p3 = bsf(p3);
	p4 = bsf(p4);
	p1 += q1;
	p2 += q2;
	p3 += q3;
	p4 += q4;
	q1 = p1 << 2;
	q2 = p2 << 2;
	q3 = p3 << 2;
	q4 = p4 << 2;
	d1_L = _mm_load_si128((__m128i*) &index_L3[q1]);
	d2_L = _mm_load_si128((__m128i*) &index_L3[q2]);
	d3_L = _mm_load_si128((__m128i*) &index_L3[q3]);
	d4_L = _mm_load_si128((__m128i*) &index_L3[q4]);
	d1_H = _mm_load_si128((__m128i*) &index_L3[q1 + 2]);
	d2_H = _mm_load_si128((__m128i*) &index_L3[q2 + 2]);
	d3_H = _mm_load_si128((__m128i*) &index_L3[q3 + 2]);
	d4_H = _mm_load_si128((__m128i*) &index_L3[q4 + 2]);
	d1_L = _mm_cmpgt_epi64(k1_x2, d1_L);
	d2_L = _mm_cmpgt_epi64(k2_x2, d2_L);
	d3_L = _mm_cmpgt_epi64(k3_x2, d3_L);
	d4_L = _mm_cmpgt_epi64(k4_x2, d4_L);
	d1_H = _mm_cmpgt_epi64(k1_x2, d1_H);
	d2_H = _mm_cmpgt_epi64(k2_x2, d2_H);
	d3_H = _mm_cmpgt_epi64(k3_x2, d3_H);
	d4_H = _mm_cmpgt_epi64(k4_x2, d4_H);
	d1 = _mm_packs_epi32(d1_L, d1_H);
	d2 = _mm_packs_epi32(d2_L, d2_H);
	d3 = _mm_packs_epi32(d3_L, d3_H);
	d4 = _mm_packs_epi32(d4_L, d4_H);
	q1 += p1;
	q2 += p2;
	q3 += p3;
	q4 += p4;
	p1 = _mm_mask_epi32(d1);
	p2 = _mm_mask_epi32(d2);
	p3 = _mm_mask_epi32(d3);
	p4 = _mm_mask_epi32(d4);
	p1 ^= 31;
	p2 ^= 31;
	p3 ^= 31;
	p4 ^= 31;
	p1 = bsf(p1);
	p2 = bsf(p2);
	p3 = bsf(p3);
	p4 = bsf(p4);
	p1 += q1;
	p2 += q2;
	p3 += q3;
	p4 += q4;
	count[p1]++;
	count[p2]++;
	count[p3]++;
	count[p4]++;
	p2 <<= 16;
	p3 <<= 32;
	p4 <<= 48;
	_mm_stream_si64((long long int*) ranges, p1 | p2 | p3 | p4);
}

void histogram_1000(uint64_t *keys, uint64_t size, uint64_t delim[],
		    uint64_t *count, uint16_t *ranges, uint64_t *index)
{
	uint64_t *keys_end = &keys[size];
	uint64_t sign = 1ull << 63;
	__m128i sign_x2 = _mm_set1_epi64x(sign);
	int64_t *index_L0 = (uint64_t*) index;
	int64_t *index_L1 = &index_L0[8];
	int64_t *index_L2 = &index_L1[32];
	int64_t *index_L3 = &index_L2[160];
	uint64_t rem_1 = 4, rem_2 = 4, rem_3 = 4;
	uint64_t i, i0 = 0, i1 = 0, i2 = 0, i3 = 0;
	for (i = 0 ; i != 999 ; ++i)
		if (rem_3--)
			index_L3[i3++] = delim[i] - sign;
		else if (rem_2--) {
			rem_3 = 4;
			index_L2[i2++] = delim[i] - sign;
		} else if (rem_1--) {
			rem_2 = rem_3 = 4;
			index_L1[i1++] = delim[i] - sign;
		} else {
			rem_1 = rem_2 = rem_3 = 4;
			index_L0[i0++] = delim[i] - sign;
		}
	assert(i0 == 7);
	assert(i1 == 32);
	assert(i2 == 160);
	assert(i3 == 800);
	__m128i del_1 = _mm_set1_epi64x(index_L0[0]);
	__m128i del_2 = _mm_set1_epi64x(index_L0[1]);
	__m128i del_3 = _mm_set1_epi64x(index_L0[2]);
	__m128i del_4 = _mm_set1_epi64x(index_L0[3]);
	__m128i del_5 = _mm_set1_epi64x(index_L0[4]);
	__m128i del_6 = _mm_set1_epi64x(index_L0[5]);
	__m128i del_7 = _mm_set1_epi64x(index_L0[6]);
	uint64_t p1 = 0, p2 = 0, p3 = 0, p4 = 0;
	while ((7 & (uint64_t) ranges) != 0 && keys != keys_end) {
		i = binary_search(delim, 999, *keys++);
		count[i]++;
		*ranges++ = i;
	}
	assert((15 & (uint64_t) keys) == 0);
	uint64_t *ranges_64 = (uint64_t*) ranges;
	uint64_t *keys_aligned_end = &keys[(keys_end - keys) & ~7];
	while (keys != keys_aligned_end) {
		// load 8 keys
		__m128i k1 = _mm_load_si128((__m128i*) &keys[0]);
		__m128i k2 = _mm_load_si128((__m128i*) &keys[2]);
		__m128i k3 = _mm_load_si128((__m128i*) &keys[4]);
		__m128i k4 = _mm_load_si128((__m128i*) &keys[6]);
		keys += 8;
		// convert keys
		k1 = _mm_sub_epi64(k1, sign_x2);
		k2 = _mm_sub_epi64(k2, sign_x2);
		k3 = _mm_sub_epi64(k3, sign_x2);
		k4 = _mm_sub_epi64(k4, sign_x2);
		// root comparisons
		__m128i r = histogram_root(k1, k2, k3, k4,
		                           del_1, del_2, del_3,
		                           del_4, del_5, del_6, del_7);
		// open-up comparisons
		__m128i r1 = _mm_cvtepi16_epi32(r);
		r = _mm_shuffle_epi32(r, _MM_SHUFFLE(1, 0, 3, 2));
		__m128i r2 = _mm_cvtepi16_epi32(r);
		// call rest histogram
		histogram_8_5_5_5_part(k1, k2, r1, index_L1, index_L2, index_L3, count, &ranges_64[0]);
		histogram_8_5_5_5_part(k3, k4, r2, index_L1, index_L2, index_L3, count, &ranges_64[1]);
		ranges_64 += 2;
	}
	ranges = (uint16_t*) ranges_64;
	while (keys != keys_end) {
		i = binary_search(delim, 999, *keys++);
		count[i]++;
		*ranges++ = i;
	}
#ifdef BG
	keys -= size;
	ranges -= size;
	for (i = 0 ; i != size ; ++i)
		assert(binary_search(delim, 999, keys[i]) == ranges[i]);
#endif
}

inline void histogram_8_3_3_5_5_part(__m128i k12, __m128i k34, __m128i r,
				     int64_t *index_L1, int64_t *index_L2,
				     int64_t *index_L3, int64_t *index_L4,
				     uint64_t *count, uint64_t *ranges)
{
	__m128i r_s1 = _mm_shuffle_epi32(r, 1);
	__m128i r_s2 = _mm_shuffle_epi32(r, 2);
	__m128i r_s3 = _mm_shuffle_epi32(r, 3);
	uint64_t p1 = 0, p2 = 0, p3 = 0, p4 = 0;
	asm("movd	%1, %%eax" : "=a"(p1) : "x"(r),    "0"(p1));
	asm("movd	%1, %%ebx" : "=b"(p2) : "x"(r_s1), "0"(p2));
	asm("movd	%1, %%ecx" : "=c"(p3) : "x"(r_s2), "0"(p3));
	asm("movd	%1, %%edx" : "=d"(p4) : "x"(r_s3), "0"(p4));
	uint64_t q1 = p1 << 1;
	uint64_t q2 = p2 << 1;
	uint64_t q3 = p3 << 1;
	uint64_t q4 = p4 << 1;
	__m128i d1 = _mm_load_si128((__m128i*) &index_L1[q1]);
	__m128i d2 = _mm_load_si128((__m128i*) &index_L1[q2]);
	__m128i d3 = _mm_load_si128((__m128i*) &index_L1[q3]);
	__m128i d4 = _mm_load_si128((__m128i*) &index_L1[q4]);
	__m128i k1_x2 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(1, 0, 1, 0));
	__m128i k2_x2 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 2, 3, 2));
	__m128i k3_x2 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(1, 0, 1, 0));
	__m128i k4_x2 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 2, 3, 2));
	d1 = _mm_cmpgt_epi64(k1_x2, d1);
	d2 = _mm_cmpgt_epi64(k2_x2, d2);
	d3 = _mm_cmpgt_epi64(k3_x2, d3);
	d4 = _mm_cmpgt_epi64(k4_x2, d4);
	q1 += p1;
	q2 += p2;
	q3 += p3;
	q4 += p4;
	p1 = _mm_mask_epi64(d1);
	p2 = _mm_mask_epi64(d2);
	p3 = _mm_mask_epi64(d3);
	p4 = _mm_mask_epi64(d4);
	p1 ^= 7;
	p2 ^= 7;
	p3 ^= 7;
	p4 ^= 7;
	p1 = bsf(p1);
	p2 = bsf(p2);
	p3 = bsf(p3);
	p4 = bsf(p4);
	p1 += q1;
	p2 += q2;
	p3 += q3;
	p4 += q4;
	q1 = p1 << 1;
	q2 = p2 << 1;
	q3 = p3 << 1;
	q4 = p4 << 1;
	d1 = _mm_load_si128((__m128i*) &index_L2[q1]);
	d2 = _mm_load_si128((__m128i*) &index_L2[q2]);
	d3 = _mm_load_si128((__m128i*) &index_L2[q3]);
	d4 = _mm_load_si128((__m128i*) &index_L2[q4]);
	d1 = _mm_cmpgt_epi64(k1_x2, d1);
	d2 = _mm_cmpgt_epi64(k2_x2, d2);
	d3 = _mm_cmpgt_epi64(k3_x2, d3);
	d4 = _mm_cmpgt_epi64(k4_x2, d4);
	q1 += p1;
	q2 += p2;
	q3 += p3;
	q4 += p4;
	p1 = _mm_mask_epi64(d1);
	p2 = _mm_mask_epi64(d2);
	p3 = _mm_mask_epi64(d3);
	p4 = _mm_mask_epi64(d4);
	p1 ^= 7;
	p2 ^= 7;
	p3 ^= 7;
	p4 ^= 7;
	p1 = bsf(p1);
	p2 = bsf(p2);
	p3 = bsf(p3);
	p4 = bsf(p4);
	p1 += q1;
	p2 += q2;
	p3 += q3;
	p4 += q4;
	q1 = p1 << 2;
	q2 = p2 << 2;
	q3 = p3 << 2;
	q4 = p4 << 2;
	__m128i d1_L = _mm_load_si128((__m128i*) &index_L3[q1]);
	__m128i d2_L = _mm_load_si128((__m128i*) &index_L3[q2]);
	__m128i d3_L = _mm_load_si128((__m128i*) &index_L3[q3]);
	__m128i d4_L = _mm_load_si128((__m128i*) &index_L3[q4]);
	__m128i d1_H = _mm_load_si128((__m128i*) &index_L3[q1 + 2]);
	__m128i d2_H = _mm_load_si128((__m128i*) &index_L3[q2 + 2]);
	__m128i d3_H = _mm_load_si128((__m128i*) &index_L3[q3 + 2]);
	__m128i d4_H = _mm_load_si128((__m128i*) &index_L3[q4 + 2]);
	d1_L = _mm_cmpgt_epi64(k1_x2, d1_L);
	d2_L = _mm_cmpgt_epi64(k2_x2, d2_L);
	d3_L = _mm_cmpgt_epi64(k3_x2, d3_L);
	d4_L = _mm_cmpgt_epi64(k4_x2, d4_L);
	d1_H = _mm_cmpgt_epi64(k1_x2, d1_H);
	d2_H = _mm_cmpgt_epi64(k2_x2, d2_H);
	d3_H = _mm_cmpgt_epi64(k3_x2, d3_H);
	d4_H = _mm_cmpgt_epi64(k4_x2, d4_H);
	d1 = _mm_packs_epi32(d1_L, d1_H);
	d2 = _mm_packs_epi32(d2_L, d2_H);
	d3 = _mm_packs_epi32(d3_L, d3_H);
	d4 = _mm_packs_epi32(d4_L, d4_H);
	q1 += p1;
	q2 += p2;
	q3 += p3;
	q4 += p4;
	p1 = _mm_mask_epi32(d1);
	p2 = _mm_mask_epi32(d2);
	p3 = _mm_mask_epi32(d3);
	p4 = _mm_mask_epi32(d4);
	p1 ^= 31;
	p2 ^= 31;
	p3 ^= 31;
	p4 ^= 31;
	p1 = bsf(p1);
	p2 = bsf(p2);
	p3 = bsf(p3);
	p4 = bsf(p4);
	p1 += q1;
	p2 += q2;
	p3 += q3;
	p4 += q4;
	q1 = p1 << 2;
	q2 = p2 << 2;
	q3 = p3 << 2;
	q4 = p4 << 2;
	d1_L = _mm_load_si128((__m128i*) &index_L4[q1]);
	d2_L = _mm_load_si128((__m128i*) &index_L4[q2]);
	d3_L = _mm_load_si128((__m128i*) &index_L4[q3]);
	d4_L = _mm_load_si128((__m128i*) &index_L4[q4]);
	d1_H = _mm_load_si128((__m128i*) &index_L4[q1 + 2]);
	d2_H = _mm_load_si128((__m128i*) &index_L4[q2 + 2]);
	d3_H = _mm_load_si128((__m128i*) &index_L4[q3 + 2]);
	d4_H = _mm_load_si128((__m128i*) &index_L4[q4 + 2]);
	d1_L = _mm_cmpgt_epi64(k1_x2, d1_L);
	d2_L = _mm_cmpgt_epi64(k2_x2, d2_L);
	d3_L = _mm_cmpgt_epi64(k3_x2, d3_L);
	d4_L = _mm_cmpgt_epi64(k4_x2, d4_L);
	d1_H = _mm_cmpgt_epi64(k1_x2, d1_H);
	d2_H = _mm_cmpgt_epi64(k2_x2, d2_H);
	d3_H = _mm_cmpgt_epi64(k3_x2, d3_H);
	d4_H = _mm_cmpgt_epi64(k4_x2, d4_H);
	d1 = _mm_packs_epi32(d1_L, d1_H);
	d2 = _mm_packs_epi32(d2_L, d2_H);
	d3 = _mm_packs_epi32(d3_L, d3_H);
	d4 = _mm_packs_epi32(d4_L, d4_H);
	q1 += p1;
	q2 += p2;
	q3 += p3;
	q4 += p4;
	p1 = _mm_mask_epi32(d1);
	p2 = _mm_mask_epi32(d2);
	p3 = _mm_mask_epi32(d3);
	p4 = _mm_mask_epi32(d4);
	p1 ^= 31;
	p2 ^= 31;
	p3 ^= 31;
	p4 ^= 31;
	p1 = bsf(p1);
	p2 = bsf(p2);
	p3 = bsf(p3);
	p4 = bsf(p4);
	p1 += q1;
	p2 += q2;
	p3 += q3;
	p4 += q4;
	count[p1]++;
	count[p2]++;
	count[p3]++;
	count[p4]++;
	p2 <<= 16;
	p3 <<= 32;
	p4 <<= 48;
	_mm_stream_si64((long long int*) ranges, p1 | p2 | p3 | p4);
}

void histogram_1800(uint64_t *keys, uint64_t size, uint64_t delim[],
		    uint64_t *count, uint16_t *ranges, uint64_t *index)
{
	uint64_t *keys_end = &keys[size];
	uint64_t sign = 1ull << 63;
	__m128i sign_x2 = _mm_set1_epi64x(sign);
	int64_t *index_L0 = (uint64_t*) index;
	int64_t *index_L1 = &index_L0[8];
	int64_t *index_L2 = &index_L1[16];
	int64_t *index_L3 = &index_L2[48];
	int64_t *index_L4 = &index_L3[288];
	uint64_t rem_1 = 2, rem_2 = 2, rem_3 = 4, rem_4 = 4;
	uint64_t i, i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0;
	for (i = 0 ; i != 1799 ; ++i)
		if (rem_4--)
			index_L4[i4++] = delim[i] - sign;
		else if (rem_3--) {
			rem_4 = 4;
			index_L3[i3++] = delim[i] - sign;
		} else if (rem_2--) {
			rem_3 = rem_4 = 4;
			index_L2[i2++] = delim[i] - sign;
		} else if (rem_1--) {
			rem_3 = rem_4 = 4;
			rem_2 = 2;
			index_L1[i1++] = delim[i] - sign;
		} else {
			rem_3 = rem_4 = 4;
			rem_1 = rem_2 = 2;
			index_L0[i0++] = delim[i] - sign;
		}
	assert(i0 == 7);
	assert(i1 == 16);
	assert(i2 == 48);
	assert(i3 == 288);
	assert(i4 == 1440);
	__m128i del_1 = _mm_set1_epi64x(index_L0[0]);
	__m128i del_2 = _mm_set1_epi64x(index_L0[1]);
	__m128i del_3 = _mm_set1_epi64x(index_L0[2]);
	__m128i del_4 = _mm_set1_epi64x(index_L0[3]);
	__m128i del_5 = _mm_set1_epi64x(index_L0[4]);
	__m128i del_6 = _mm_set1_epi64x(index_L0[5]);
	__m128i del_7 = _mm_set1_epi64x(index_L0[6]);
	uint64_t p1 = 0, p2 = 0, p3 = 0, p4 = 0;
	while ((7 & (uint64_t) ranges) != 0 && keys != keys_end) {
		i = binary_search(delim, 1799, *keys++);
		count[i]++;
		*ranges++ = i;
	}
	assert((15 & (uint64_t) keys) == 0);
	uint64_t *ranges_64 = (uint64_t*) ranges;
	uint64_t *keys_aligned_end = &keys[(keys_end - keys) & ~7];
	while (keys != keys_aligned_end) {
		// load 8 keys
		__m128i k1 = _mm_load_si128((__m128i*) &keys[0]);
		__m128i k2 = _mm_load_si128((__m128i*) &keys[2]);
		__m128i k3 = _mm_load_si128((__m128i*) &keys[4]);
		__m128i k4 = _mm_load_si128((__m128i*) &keys[6]);
		keys += 8;
		// convert keys
		k1 = _mm_sub_epi64(k1, sign_x2);
		k2 = _mm_sub_epi64(k2, sign_x2);
		k3 = _mm_sub_epi64(k3, sign_x2);
		k4 = _mm_sub_epi64(k4, sign_x2);
		// root comparisons
		__m128i r = histogram_root(k1, k2, k3, k4,
		                           del_1, del_2, del_3,
		                           del_4, del_5, del_6, del_7);
		// open-up comparisons
		__m128i r1 = _mm_cvtepi16_epi32(r);
		r = _mm_shuffle_epi32(r, _MM_SHUFFLE(1, 0, 3, 2));
		__m128i r2 = _mm_cvtepi16_epi32(r);
		// call rest histogram
		histogram_8_3_3_5_5_part(k1, k2, r1, index_L1, index_L2, index_L3, index_L4, count, &ranges_64[0]);
		histogram_8_3_3_5_5_part(k3, k4, r2, index_L1, index_L2, index_L3, index_L4, count, &ranges_64[1]);
		ranges_64 += 2;
	}
	ranges = (uint16_t*) ranges_64;
	while (keys != keys_end) {
		i = binary_search(delim, 1799, *keys++);
		count[i]++;
		*ranges++ = i;
	}
#ifdef BG
	keys -= size;
	ranges -= size;
	for (i = 0 ; i != size ; ++i)
		assert(binary_search(delim, 1799, keys[i]) == ranges[i]);
#endif
}

void partition_offsets(uint64_t **count, uint64_t partitions, uint64_t id,
		       uint64_t threads, uint64_t *offsets)
{
	uint64_t i, t, p = 0;
	for (i = 0 ; i != partitions ; ++i) {
		for (t = 0 ; t != id ; ++t)
			p += count[t][i];
		offsets[i] = p;
		for (; t != threads ; ++t)
			p += count[t][i];
	}
}

typedef struct {
	int32_t n[4];
} i4; i4 i4v;

typedef struct {
	uint32_t n[4];
} u4; u4 u4v;

void known_partition(uint64_t *keys, uint64_t *rids, uint16_t *ranges,
		     uint64_t size, uint64_t *offsets, uint64_t *sizes,
		     uint64_t *keys_out, uint64_t *rids_out, uint64_t *buf,
		     uint64_t partitions)
{
	// inputs must be pairwise aligned for SIMD access
	assert((15 & (uint64_t) keys) == (15 & (uint64_t) rids));
	assert((31 & (uint64_t) keys) == 4 * (7 & (uint64_t) ranges));
	// outputs must be pairwise aligned for cache line access
	assert((63 & (uint64_t) keys_out) == (63 & (uint64_t) rids_out));
	// basic parameters
	uint64_t *keys_end = &keys[size];
	uint64_t p, i;
	__m128i h, k12, k34, v12, v34;
	// space for unaligned access
	uint64_t unaligned_keys[4];
	uint64_t unaligned_vals[4];
	uint16_t unaligned_ranges[8];
	// compute offset to align output
	uint64_t to_align = 0;
	while (63 & (uint64_t) &keys_out[to_align])
		to_align++;
	assert(to_align < 8);
	uint64_t virtual_add = 0;
	if (to_align && to_align < size)
		virtual_add = 8 - to_align;
	keys_out -= virtual_add;
	rids_out -= virtual_add;
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	uint32_t *keys_32_out = (uint32_t*) keys_out;
	uint32_t *rids_32_out = (uint32_t*) rids_out;
	// initialize partition buffers including alignment offset
	if (offsets != NULL)
		for (i = p = 0 ; p != partitions ; ++p)
			buf[(p << 4) | 14] = (offsets[p] + virtual_add) << 1;
	else {
		for (i = p = 0 ; p != partitions ; ++p) {
			buf[(p << 4) | 14] = (i + virtual_add) << 1;
			i += sizes[p];
		}
		assert(i == size);
	}
	// partition first 0-3 unaligned items
	for (i = 0 ; (7 & (uint64_t) ranges) && i != size ; ++i) {
		unaligned_ranges[i] = *ranges++;
		unaligned_keys[i] = *keys++;
		unaligned_vals[i] = *rids++;
	}
	assert((15 & (uint64_t) keys) == 0);
	uint64_t *keys_loop_end = &keys[(size - i) & ~3];
	if (i) {
		h = _mm_loadu_si128((__m128i*) unaligned_ranges);
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		v12 = _mm_loadu_si128((__m128i*) &unaligned_vals[0]);
		v34 = _mm_loadu_si128((__m128i*) &unaligned_vals[2]);
		goto unaligned_part_intro;
	}
	// loop of data partitioning
	while (keys != keys_loop_end) {
		h = _mm_loadl_epi64((__m128i*) ranges);
		k12 = _mm_load_si128((__m128i*) &keys[0]);
		k34 = _mm_load_si128((__m128i*) &keys[2]);
		v12 = _mm_load_si128((__m128i*) &rids[0]);
		v34 = _mm_load_si128((__m128i*) &rids[2]);
		keys += 4; rids += 4;
		ranges += 4; i = 4;
		unaligned_part_intro:
		h = _mm_cvtepu16_epi32(h);
		k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
		k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
		v12 = _mm_shuffle_epi32(v12, _MM_SHUFFLE(3, 1, 2, 0));
		v34 = _mm_shuffle_epi32(v34, _MM_SHUFFLE(3, 1, 2, 0));
		h = _mm_slli_epi32(h, 4);
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
			if (offset != 14) ;
			else if (index != 14) {
				uint32_t *dest_x = &keys_32_out[index - 14];
				uint32_t *dest_y = &rids_32_out[index - 14];
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
			} else {
				// special case to write the 1st cache line
				index = virtual_add;
				for (p >>= 4 ; p ; --p)
					index += sizes[p - 1];
				while (index != 8) {
					uint64_t key_val = src[index + index];
					_mm_stream_si64((long long *) &keys_out[index],   src[index + index]);
					_mm_stream_si64((long long *) &rids_out[index++], src[index + index + 1]);
				}
				src[14] = 16;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
			k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
			v_L = _mm_shuffle_epi32(v_L, _MM_SHUFFLE(0, 3, 2, 1));
			v_H = _mm_shuffle_epi32(v_H, _MM_SHUFFLE(0, 3, 2, 1));
		} while (--i);
	}
	// partition last 0-3 unaligned items
	for (i = p = 0 ; keys != keys_end ; ++i) {
		unaligned_ranges[i] = *ranges++;
		unaligned_keys[i] = *keys++;
		unaligned_vals[i] = *rids++;
	}
	if (i) {
		keys_loop_end = keys_end;
		h = _mm_loadu_si128((__m128i*) unaligned_ranges);
		k12 = _mm_loadu_si128((__m128i*) &unaligned_keys[0]);
		k34 = _mm_loadu_si128((__m128i*) &unaligned_keys[2]);
		v12 = _mm_loadu_si128((__m128i*) &unaligned_vals[0]);
		v34 = _mm_loadu_si128((__m128i*) &unaligned_vals[2]);
		goto unaligned_part_intro;
	}
	// flush remaining items from buffers to output
	for (p = 0 ; p != partitions ; ++p) {
		uint64_t *src = &buf[p << 4];
		uint64_t index = src[14] >> 1;
		uint64_t remain = index & 7;
		uint64_t offset = 0;
		if (remain > sizes[p])
			offset = remain - sizes[p];
		index -= remain - offset;
		while (offset != remain) {
			_mm_stream_si64((long long *) &keys_out[index], src[offset + offset]);
			_mm_stream_si64((long long *) &rids_out[index], src[offset + offset + 1]);
			offset++; index++;
		}
	}
	// check sizes of partitions
	if (offsets != NULL)
		for (p = 0 ; p != partitions ; ++p)
			assert(offsets[p] + sizes[p] + virtual_add == (buf[(p << 4) | 14] >> 1));
	else {
		for (i = p = 0 ; p != partitions ; ++p) {
			i += sizes[p];
			assert(i + virtual_add == (buf[(p << 4) | 14] >> 1));
		}
		assert(i == size);
	}
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
	free(buf);
}

typedef struct {
	double fudge;
	uint64_t **keys;
	uint64_t **rids;
	uint64_t *size;
	uint16_t **ranges;
	uint64_t **keys_buf;
	uint64_t **rids_buf;
	uint64_t ***count;
	uint64_t partitions_1;
	uint64_t partitions_2;
	uint64_t *sample;
	uint64_t *sample_buf;
	uint64_t sample_size;
	uint64_t **sample_hist;
	int *seed;
	int *numa_node;
	int *cpu;
	int threads;
	int numa;
	int max_threads;
	int max_numa;
	int allocated;
	int interleaved;
	volatile uint64_t *numa_counter;
	volatile uint64_t *part_counter;
	pthread_barrier_t *global_barrier;
	pthread_barrier_t **local_barrier;
	pthread_barrier_t *sample_barrier;
} global_data_t;

typedef struct {
	int id;
	int seed;
	uint64_t checksum;
	uint64_t alloc_time;
	uint64_t sample_time;
	uint64_t histogram_1_time;
	uint64_t partition_1_time;
	uint64_t numa_shuffle_time;
	uint64_t histogram_2_time;
	uint64_t partition_2_time;
	uint64_t sorting_time;
	global_data_t *global;
} thread_data_t;

typedef struct {
	uint64_t *src_key;
	uint64_t *src_rid;
	uint64_t *dst_key;
	uint64_t *dst_rid;
	uint64_t size;
} transfer_t;

void *sort_thread(void *arg)
{
	thread_data_t *a = (thread_data_t*) arg;
	global_data_t *d = a->global;
	uint64_t s, p, r, i, j, o, k, t, n;
	uint64_t lb = 0, gb = 0, id = a->id;
	uint64_t numa = d->numa;
	uint64_t numa_node = d->numa_node[id];
	uint64_t threads = d->threads;
	uint64_t threads_per_numa = threads / numa;
	pthread_barrier_t *local_barrier = d->local_barrier[numa_node];
	pthread_barrier_t *global_barrier = d->global_barrier;
	// id in local numa threads
	uint64_t numa_local_id = 0;
	for (i = 0 ; i != id ; ++i)
		if (d->numa_node[i] == numa_node)
			numa_local_id++;
	assert(numa > 1 || numa_local_id == id);
	uint64_t total_size = 0;
	for (n = 0 ; n != numa ; ++n)
		total_size += d->size[n];
	// bind thread and its allocation
	if (threads <= d->max_threads)
		cpu_bind(id);
	//	cpu_bind(d->cpu[id]);
	// if (numa <= d->max_numa)
	// 	memory_bind(d->cpu[id]);
	// initial histogram and buffers
	uint64_t partitions_1 = d->partitions_1;
	uint64_t partitions_2 = d->partitions_2;
	uint64_t max_partitions = partitions_1 > partitions_2 ?
				  partitions_1 : partitions_2;
	uint64_t *index = malloc(max_partitions * sizeof(uint64_t));
	uint64_t *delim_1 = malloc((partitions_1 - 1) * sizeof(uint64_t));
	uint64_t *delim_2 = malloc((partitions_2 - 1) * sizeof(uint64_t));
	uint64_t *count = calloc(partitions_1, sizeof(uint64_t));
	uint64_t *offsets = malloc(max_partitions * sizeof(uint64_t));
	uint64_t *buf = mamalloc((max_partitions << 4) * sizeof(uint64_t));
	d->count[numa_node][numa_local_id] = count;
	uint64_t numa_size = d->size[numa_node];
	uint64_t size = (numa_size / threads_per_numa) & ~7;
	uint64_t offset = size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		size = numa_size - offset;
	uint64_t tim = micro_time();
	if (!d->allocated) {
		if (!numa_local_id) {
			uint64_t cap = d->size[numa_node] * d->fudge;
			if (d->interleaved) {
				d->keys_buf[numa_node] = numa_alloc_interleaved(cap * sizeof(uint64_t));
				d->rids_buf[numa_node] = numa_alloc_interleaved(cap * sizeof(uint64_t));
				d->ranges[numa_node]   = numa_alloc_interleaved(cap * sizeof(uint16_t));
			} else {
				d->keys_buf[numa_node] = mamalloc(cap * sizeof(uint64_t));
				d->rids_buf[numa_node] = mamalloc(cap * sizeof(uint64_t));
				d->ranges[numa_node]   = mamalloc(cap * sizeof(uint16_t));
			}
		}
		pthread_barrier_wait(&local_barrier[lb++]);
	}
	uint64_t *keys = &d->keys[numa_node][offset];
	uint64_t *rids = &d->rids[numa_node][offset];
	uint64_t *keys_out = d->keys_buf[numa_node];
	uint64_t *rids_out = d->rids_buf[numa_node];
	uint16_t *ranges = &d->ranges[numa_node][offset];
	if (!d->allocated) {
		uint64_t size_aligned = size & ~3;
		for (p = 0 ; p != size ; ++p)
			_mm_stream_si64((long long *) &keys_out[p], 0);
		for (p = 0 ; p != size ; ++p)
			_mm_stream_si64((long long *) &rids_out[p], 0);
		for (p = 0 ; p != size_aligned ; p += 2)
			_mm_stream_si32((int32_t*) &ranges[p], 0);
		pthread_barrier_wait(&local_barrier[lb++]);
	}
	tim = micro_time() - tim;
	a->alloc_time = tim;
	// sample keys from local data
	tim = micro_time();
	assert((d->sample_size & 3) == 0);
	uint64_t sample_size = (d->sample_size / threads) & ~15;
	uint64_t *sample = &d->sample[sample_size * id];
	if (id + 1 == threads)
		sample_size = d->sample_size - sample_size * id;
	rand64_t *gen = rand64_init(a->seed);
	for (p = 0 ; p != sample_size ; ++p)
		sample[p] = keys[mulhi(rand64_next(gen), size)];
#ifdef BG
	if (id == 0) fprintf(stderr, "Sampling done!\n");
#endif
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
#ifdef BG
	for (i = 1 ; i != d->sample_size ; ++i)
		assert(d->sample[i] >= d->sample[i - 1]);
#endif
	// get delimiters for 1st phase range partitioning
	j = d->sample_size / d->partitions_1;
	for (i = 0 ; i != partitions_1 - 1 ; ++i)
		delim_1[i] = d->sample[j * (i + 1) - 1];
	tim = micro_time() - tim;
	a->sample_time = tim;
	// 1st histogram function
	void (*histogram_1st)(uint64_t *keys, uint64_t size, uint64_t *delim,
			      uint64_t *count, uint16_t *ranges, uint64_t *index);
	if	(partitions_1 == 360)
		histogram_1st = histogram_360;
	else if (partitions_1 == 1000)
		histogram_1st = histogram_1000;
	else if (partitions_1 == 1800)
		histogram_1st = histogram_1800;
	else abort();
	// 2nd histogram function
	void (*histogram_2nd)(uint64_t *keys, uint64_t size, uint64_t *delim,
			      uint64_t *count, uint16_t *ranges, uint64_t *index);
	if	(partitions_2 == 360)
		histogram_2nd = histogram_360;
	else if (partitions_2 == 1000)
		histogram_2nd = histogram_1000;
	else if (partitions_2 == 1800)
		histogram_2nd = histogram_1800;
	else if (partitions_2 != 1)
		abort();
	// generate histogram for 1st partition phase
	tim = micro_time();
	histogram_1st(keys, size, delim_1, count, ranges, index);
	tim = micro_time() - tim;
	a->histogram_1_time = tim;
#ifdef BG
	for (i = 0 ; i != size ; ++i)
		assert(binary_search(delim_1, partitions_1 - 1, keys[i]) == ranges[i]);
	if (id == 0) fprintf(stderr, "1st histogram done!\n");
#endif
	pthread_barrier_wait(&local_barrier[lb++]);
	uint64_t **counts = d->count[numa_node];
	// align to cache lines if single NUMA
	tim = micro_time();
	partition_offsets(counts, partitions_1, numa_local_id,
			  threads_per_numa, offsets);
	// partition 1st pass
	known_partition(keys, rids, ranges, size, offsets, count,
			keys_out, rids_out, buf, partitions_1);
	tim = micro_time() - tim;
	a->partition_1_time = tim;
#ifdef BG
	if (id == 0) fprintf(stderr, "1st partition done!\n");
#endif
	// wait all other threads to do first partition
	pthread_barrier_wait(d->sample_barrier);
	// start time second phase
	tim = micro_time();
	// compute total partition sizes
	uint64_t *part_total_size = calloc(partitions_1, sizeof(uint64_t));
	for (n = 0 ; n != numa ; ++n)
		for (t = 0 ; t != threads_per_numa ; ++t)
			for (p = 0 ; p != partitions_1 ; ++p)
				part_total_size[p] += d->count[n][t][p];
	// info for partitions per node
	uint64_t part_per_numa[numa];
	uint64_t size_per_numa[numa];
	uint64_t previous_numa_partitions = 0;
	uint64_t numa_partitions = partitions_1;
	part_per_numa[0] = partitions_1;
	size_per_numa[0] = total_size;
	// [shuffle] -> keys 1 -> keys 2 -> keys 1 (final)
	uint64_t *keys_1 = d->keys_buf[numa_node];
	uint64_t *rids_1 = d->rids_buf[numa_node];
	uint64_t *keys_2 = d->keys[numa_node];
	uint64_t *rids_2 = d->rids[numa_node];
	if (numa > 1) {
		keys_1 = d->keys[numa_node];
		rids_1 = d->rids[numa_node];
		keys_2 = d->keys_buf[numa_node];
		rids_2 = d->rids_buf[numa_node];
		// decide # of partitions per NUMA node
		uint64_t prev_o = 0, prev_p = 0;
		for (n = 0 ; n != numa - 1 ; ++n) {
			uint64_t g = (total_size / numa) * (n + 1);
			for (p = o = 0; p != partitions_1 ; ++p) {
				if (o + part_total_size[p] >= g) break;
				o += part_total_size[p];
			}
			if (p != partitions_1 && part_total_size[p] + o - g < g - o)
				o += part_total_size[p++];
			part_per_numa[n] = p - prev_p;
			size_per_numa[n] = o - prev_o;
			prev_p = p;
			prev_o = o;
		}
		part_per_numa[n] = partitions_1 - p;
		size_per_numa[n] = total_size - o;
		// check that sizes indeed fit
		for (n = 0 ; n != numa ; ++n) {
			if (size_per_numa[n] > d->size[n] * d->fudge)
				fprintf(stderr, "NUMA %ld is %.2f%% of input\n", numa_node,
						 numa_size * 100.0 / total_size);
			assert(size_per_numa[n] <= d->size[n] * d->fudge);
		}
		for (n = 0 ; n != numa_node ; ++n)
			previous_numa_partitions += part_per_numa[n];
		// current offset per thread
		uint64_t remote_offset[numa];
		for (n = 0 ; n != numa ; ++n) {
			uint64_t offset = 0;
			for (p = 0 ; p != previous_numa_partitions ; ++p)
				for (t = 0 ; t != threads_per_numa ; ++t)
					offset += d->count[n][t][p];
			remote_offset[n] = offset;
		}
		// create array of transfers for local numa node
		numa_partitions = part_per_numa[numa_node];
		transfer_t *transfers = malloc(numa_partitions * numa * sizeof(transfer_t));
		uint64_t transfer_unit = 0;
		uint64_t local_offset = 0;
		for (p = 0 ; p != numa_partitions ; ++p) {
			uint64_t lp = p + previous_numa_partitions;
			// partition comes from all numa nodes
			for (n = 0 ; n != numa ; ++n) {
				// accumulate from all threads
				uint64_t remote_size = 0;
				for (t = 0 ; t != threads_per_numa ; ++t)
					remote_size += d->count[n][t][lp];
				// set transfer
				transfer_t *tr = &transfers[transfer_unit++];
				tr->src_key = &d->keys_buf[n][remote_offset[n]];
				tr->src_rid = &d->rids_buf[n][remote_offset[n]];
				tr->dst_key = &d->keys[numa_node][local_offset];
				tr->dst_rid = &d->rids[numa_node][local_offset];
				tr->size = remote_size;
				// update offsets
				local_offset += remote_size;
				remote_offset[n] += remote_size;
			}
		}
		assert(local_offset <= d->size[numa_node] * d->fudge);
		assert(transfer_unit == numa_partitions * numa);
		// randomize transfer order per numa
		rand64_t *common_gen = rand64_init(d->seed[numa_node]);
		for (p = 0 ; p != transfer_unit ; ++p) {
			r = rand64_next(common_gen);
			r = mulhi(r, transfer_unit - p) + p;
			transfer_t tmp = transfers[p];
			transfers[p] = transfers[r];
			transfers[r] = tmp;
		}
		free(common_gen);
		// get a partition per numa node
		volatile uint64_t *numa_counter = &d->numa_counter[numa_node << 8];
		while ((p = __sync_fetch_and_add(numa_counter, 1)) < transfer_unit) {
			// copy partitions across NUMA
			transfer_t *tr = &transfers[p];
			copy(tr->dst_key, tr->src_key, tr->size);
			copy(tr->dst_rid, tr->src_rid, tr->size);
		}
		pthread_barrier_wait(&global_barrier[gb++]);
	}
	tim = micro_time() - tim;
	a->numa_shuffle_time = tim;
	// space for smaller samples
	sample_size = (partitions_2 << 3) - 1;
	sample = malloc(sample_size * sizeof(uint64_t));
	count = calloc(partitions_2, sizeof(uint64_t));
	// partition again and sort
	uint64_t h_tim = 0, p_tim = 0;
	tim = micro_time();
	volatile uint64_t *part_counter = &d->part_counter[numa_node << 8];
	uint64_t target_p = __sync_fetch_and_add(part_counter, 1);
	ranges = d->ranges[numa_node];
	for (p = 0 ; p != numa_partitions ; ++p) {
		// check handling partition
		size = part_total_size[p + previous_numa_partitions];
		if (p != target_p) {
			keys_1 += size; keys_2 += size;
			rids_1 += size; rids_2 += size;
			ranges += size;
			continue;
		}
		if (partitions_2 == 1) {
			simd_combsort(keys_1, rids_1, size, keys_2, rids_2);
			keys_1 += size; keys_2 += size;
			rids_1 += size; rids_2 += size;
			ranges += size;
		} else {
			// small sample
			for (i = 0 ; i != sample_size ; ++i)
				sample[i] = keys_1[mulhi(rand64_next(gen), size)];
			scalar_combsort_keys(sample, sample_size);
			for (i = 0 ; i != partitions_2 - 1 ; ++i)
				delim_2[i] = sample[(i << 3) + 7];
			// 2nd level histogram
			t = micro_time();
			for (i = 0 ; i != partitions_2 ; ++i) count[i] = 0;
			histogram_2nd(keys_1, size, delim_2, count, ranges, index);
			t = micro_time() - t;
			h_tim += t;
			// 2nd level partition
			t = micro_time();
			known_partition(keys_1, rids_1, ranges, size, NULL, count,
					keys_2, rids_2, buf, partitions_2);
			t = micro_time() - t;
			p_tim += t;
			// sort sub-parts (or single part)
			for (i = 0 ; i != partitions_2 ; ++i) {
				size = count[i];
				simd_combsort(keys_2, rids_2, size, keys_1, rids_1);
				keys_1 += size; keys_2 += size;
				rids_1 += size; rids_2 += size;
				ranges += size;
			}
			t = micro_time() - t;
		}
		// request new
		target_p = __sync_fetch_and_add(part_counter, 1);
	}
	tim = micro_time() - tim;
	a->histogram_2_time = h_tim;
	a->partition_2_time = p_tim;
	a->sorting_time = tim - p_tim - h_tim;
	free(buf);
	free(index);
	free(count);
	free(sample);
	free(offsets);
	free(delim_1);
	free(delim_2);
#ifdef BG
	if (id == 0) fprintf(stderr, "2nd partition done!\n");
#endif
	if (numa > 1 && !numa_local_id)
		d->size[numa_node] = size_per_numa[numa_node];
	pthread_exit(NULL);
}

int sort(uint64_t **keys, uint64_t **rids, uint64_t *size, int threads,
         int numa, double fudge, uint64_t **keys_buf, uint64_t **rids_buf,
         uint16_t **ranges, char **description, uint64_t *times, int interleaved)
{
	int i, j, p, t, n;
	int threads_per_numa = threads / numa;
	pthread_t *id = malloc(threads * sizeof(pthread_t));
	thread_data_t *data = malloc(threads * sizeof(thread_data_t));
	// check aligned input
	for (i = 0 ; i != numa ; ++i) {
		assert((15 & (uint64_t) keys[i]) == 0);
		assert((15 & (uint64_t) rids[i]) == 0);
	}
	// initialize global barriers
	int local_barriers = 5;
	int global_barriers = 50;
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
	global.threads = threads;
	global.numa = numa;
	global.fudge = fudge;
	global.keys = keys;
	global.rids = rids;
	global.size = size;
	global.max_threads = hardware_threads();
	global.max_numa = numa_max_node() + 1;
	global.ranges = ranges;
	global.keys_buf = keys_buf;
	global.rids_buf = rids_buf;
	global.interleaved = interleaved;
	global.global_barrier = global_barrier;
	global.local_barrier = local_barrier;
	global.sample_barrier = &sample_barrier;
	// compute total size
	uint64_t total_size = 0;
	for (n = 0 ; n != numa ; ++n)
		total_size += size[n];
	// check if allocation needed
	if (keys_buf[0] == NULL)
		for (n = 0 ; n != numa ; ++n) {
			assert(keys_buf[n] == NULL);
			assert(rids_buf[n] == NULL);
			assert(ranges[n] == NULL);
		}
	else
		for (n = 0 ; n != numa ; ++n) {
			assert(keys_buf[n] != NULL);
			assert(rids_buf[n] != NULL);
			assert(ranges[n] != NULL);
		}
	global.allocated = keys_buf[0] != NULL;
	// decide partitions
	uint64_t partition_fanout[2];
	decide_partitions(total_size, partition_fanout, numa, 0);
	global.partitions_1 = partition_fanout[0];
	global.partitions_2 = partition_fanout[1];
	// counts
	global.count = malloc(numa * sizeof(uint64_t**));
	for (n = 0 ; n != numa ; ++n)
		global.count[n] = malloc(threads_per_numa * sizeof(uint64_t*));
	global.sample_size = 0.01 * total_size;
	if (global.sample_size > 1000000)
		global.sample_size = 1000000;
	global.sample	  = numa_alloc_interleaved(global.sample_size * sizeof(uint64_t));
	global.sample_buf = numa_alloc_interleaved(global.sample_size * sizeof(uint64_t));
	global.numa_counter = calloc(numa << 8, sizeof(uint64_t));
	global.part_counter = calloc(numa << 8, sizeof(uint64_t));
	global.sample_hist = malloc(threads * sizeof(uint64_t*));
	for (t = 0 ; t != threads ; ++t)
		global.sample_hist[t] = malloc(256 * sizeof(uint64_t));
	global.cpu = malloc(threads * sizeof(int));
	global.numa_node = malloc(threads * sizeof(int));
	schedule_threads(global.cpu, global.numa_node, threads, numa);
	global.seed = malloc(numa * sizeof(int));
	for (n = 0 ; n != numa ; ++n)
		global.seed[n] = rand();
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
	uint64_t  at = 0, sat = 0, h1t = 0, h2t = 0;
	uint64_t nst = 0, p1t = 0, p2t = 0, sot = 0;
	for (t = 0 ; t != threads ; ++t) {
		at += data[t].alloc_time;
		sat += data[t].sample_time;
		h1t += data[t].histogram_1_time;
		p1t += data[t].partition_1_time;
		nst += data[t].numa_shuffle_time;
		h2t += data[t].histogram_2_time;
		p2t += data[t].partition_2_time;
		sot += data[t].sorting_time;
	}
	times[0] = at / threads;    description[0] = "Allocation time:	  ";
	times[1] = sat / threads;   description[1] = "Sampling time:	  ";
	times[2] = h1t / threads;   description[2] = "1st histogram time: ";
	times[3] = p1t / threads;   description[3] = "1st partition time: ";
	times[4] = nst / threads;   description[4] = "Shuffling time:	  ";
	times[5] = h2t / threads;   description[5] = "2nd histogram time: ";
	times[6] = p2t / threads;   description[6] = "2nd partition time: ";
	times[7] = sot / threads;   description[7] = "Cache sorting time: ";
	description[8] = NULL;
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
	free(global.numa_node);
	free(global.seed);
	free(global.cpu);
	for (n = 0 ; n != numa ; ++n) {
		for (t = 0 ; t != threads_per_numa ; ++t)
			free(global.count[n][t]);
		free(global.count[n]);
	}
	free(global.count);
	numa_free(global.sample,     global.sample_size * sizeof(uint64_t));
	numa_free(global.sample_buf, global.sample_size * sizeof(uint64_t));
	free((void*) global.numa_counter);
	free((void*) global.part_counter);
	free(data);
	free(id);
	return (numa == 1) ^ (global.partitions_2 == 1);
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
		if (rids) assert(key == *rids++);
		assert(key >= pkey);
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
	uint64_t p, perc = total_size / 100;
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
	int i, r, n, max_threads = hardware_threads();
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
	tuples *= 1000000;
	assert(bits > 0 && bits <= 64);
	assert(numa > 0 && threads >= numa && threads % numa == 0);
	uint64_t tuples_per_numa = tuples / numa;
	double fudge = 1.1;
	uint64_t capacity_per_numa = tuples_per_numa * fudge;
	uint64_t *keys[numa], *keys_buf[numa];
	uint64_t *rids[numa], *rids_buf[numa];
	uint16_t *ranges[numa];
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
	for (i = 0 ; i != numa ; ++i) {
		size[i] = tuples_per_numa;
		cap[i] = size[i] * fudge;
		keys_buf[i] = NULL;
		rids_buf[i] = NULL;
		ranges[i] = NULL;
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
		uint64_t **ranges_64 = (uint64_t**) ranges;
		uint64_t quarter_size[numa];
		uint64_t quarter_cap[numa];
		for (n = 0 ; n != numa ; ++n) {
			quarter_size[n] = size[n] >> 2;
			quarter_cap[n] = cap[n] >> 2;
		}
		init_64(keys_buf, size, cap, threads, numa, 0, 0.0, 0, interleaved);
		init_64(rids_buf, size, cap, threads, numa, 0, 0.0, 0, interleaved);
		init_64(ranges_64, quarter_size, quarter_cap, threads, numa, 0, 0.0, 0, interleaved);
	}
	t = micro_time() - t;
	fprintf(stderr, "Generation time: %ld us\n", t);
	fprintf(stderr, "Generation rate: %.1f mrps\n", tuples * 1.0 / t);
	// sort info
	char *desc[12];
	uint64_t times[12];
	
	PerfCounter *pc = PerfCounter_init();
    if (pc == NULL) {
        fprintf(stderr, "Failed to initialize PerfCounter\n");
        return 1;
    }

    printf("Starting counters...\n");
    PerfCounter_startCounters(pc);
	
	// call parallel sort
	t = micro_time();
	r = sort(keys, rids, size, threads, numa, fudge,
	         keys_buf, rids_buf, ranges, desc, times, interleaved);
	t = micro_time() - t;

	PerfCounter_stopCounters(pc);
    printf("Stopped counters.\n");

    printf("Performance counters report:\n");
    PerfCounter_printReport(pc, stdout, 1);

    PerfCounter_cleanup(pc);

	// show partition sizes
	decide_partitions(tuples, NULL, numa, 1);
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
	// show allocation per NUMA node
	for (i = 0 ; i != numa ; ++i)
		fprintf(stderr, "Node %d:%6.2f%%\n", i, size[i] * 100.0 / tuples);
	// check sort order and sum
	if (r) fprintf(stderr, "Destination changed\n");
	else fprintf(stderr, "Destination remained the same\n");
	uint64_t **keys_out = r ? keys_buf : keys;
	uint64_t **rids_out = r ? rids_buf : rids;
	uint64_t checksum = check(keys_out, rids_out, size, numa, same_key_payload);
	assert(checksum == sum_k);
	// free sort data
	for (i = 0 ; i != numa ; ++i)
		if (interleaved) {
			numa_free(keys_buf[i], cap[i] * sizeof(uint64_t));
			numa_free(rids_buf[i], cap[i] * sizeof(uint64_t));
			numa_free(ranges[i], (cap[i] >> 2) * sizeof(uint64_t));
			numa_free(keys[i], cap[i] * sizeof(uint64_t));
			numa_free(rids[i], cap[i] * sizeof(uint64_t));
		} else {
			free(keys_buf[i]);
			free(rids_buf[i]);
			free(ranges[i]);
			free(keys[i]);
			free(rids[i]);
		}
	printf("%.1f mrps (%.2f GB / sec)\n",
		tuples * 1.0 / t, (gigs * 1000000) / t);
	return EXIT_SUCCESS;
}
