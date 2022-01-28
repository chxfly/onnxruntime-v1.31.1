// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(__wasm__)

// This file uses mallinfo() + dlmalloc to analyze memory usage in web assembly.

#include <unistd.h>
#include <emscripten.h>

struct s_mallinfo {
	int arena;    /* non-mmapped space allocated from system */
	int ordblks;  /* number of free chunks */
	int smblks;   /* always 0 */
	int hblks;    /* always 0 */
	int hblkhd;   /* space in mmapped regions */
	int usmblks;  /* maximum total allocated space */
	int fsmblks;  /* always 0 */
	int uordblks; /* total allocated space */
	int fordblks; /* total free space */
	int keepcost; /* releasable (via malloc_trim) space */
};

extern "C" {
	extern s_mallinfo mallinfo();
}

unsigned int getTotalMemory() {
	return EM_ASM_INT(return HEAP8.length);
}

unsigned int getFreeMemory() {
	s_mallinfo i = mallinfo();
	unsigned int totalMemory = getTotalMemory();
	unsigned int dynamicTop = (unsigned int)sbrk(0);
	return totalMemory - dynamicTop + i.fordblks;
}

void checkMemory(const char* msg) {
    auto total = getTotalMemory();
    auto free = getFreeMemory();
    printf("+++ MEM [%s]: free=%u, total=%u\n", msg, free, total);
}


#endif