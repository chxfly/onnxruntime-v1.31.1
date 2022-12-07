/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sgemm.cpp

Abstract:

    This module implements the single precision matrix/matrix multiply
    operation (SGEMM).

--*/

#include "mlasi.h"
#include <cassert>

inline static size_t RoundDownPo2(size_t n, size_t q) {
  return n & -q;
}

inline static size_t RoundUpPo2(size_t n, size_t q) {
  return RoundDownPo2(n + q - 1, q);
}

//Copied from https://github.com/google/XNNPACK/blob/master/src/packing.c
void MlasPackFp16GemmGoiW(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_weights,
  size_t extra_bytes)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = std::min(nc - nr_block_start, nr);
      if (b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < RoundUpPo2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = RoundDownPo2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if (b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void
MLASCALL
MlasFP16GemmBatch(
    CBLAS_TRANSPOSE /*TransA*/,
    CBLAS_TRANSPOSE /*TransB*/,
    size_t /*M*/,
    size_t /*N*/,
    size_t /*K*/,
    const MLAS_FP16GEMM_DATA_PARAMS* /*Data*/,
    size_t /*BatchSize*/,
    MLAS_THREADPOOL* /*ThreadPool*/
    ){

    }
