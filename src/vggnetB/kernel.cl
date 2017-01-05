// Copyright 2016 HeeHoon Kim
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// should sync with vggnet.c
// CONV_BIJ = CONV_LXY * CONV_RXY
#define CONV_BK 16
#define CONV_BKH 8
#define CONV_BIJ 64
#define CONV_LXY 16
#define CONV_RXY 4

// if x <= 4608 = 512 * 9
#define DIV3(x) (((x)*5462)>>14)

#define GEMM44(u, v) \
  rA[0] = smA[u][lx * CONV_RXY]; \
  rA[1] = smA[u][lx * CONV_RXY + 1]; \
  rB[0] = smB[u][ly * CONV_RXY]; \
  rB[1] = smB[u][ly * CONV_RXY + 1]; \
  for (int ki = u; ki < v; ++ki) { \
    rB[2] = smB[ki][ly * CONV_RXY + 2]; \
    rB[3] = smB[ki][ly * CONV_RXY + 3]; \
    accum[0] += rA[0] * rB[0]; \
    accum[1] += rA[0] * rB[1]; \
    accum[4] += rA[1] * rB[0]; \
    accum[5] += rA[1] * rB[1]; \
    rA[2] = smA[ki][lx * CONV_RXY + 2]; \
    rA[3] = smA[ki][lx * CONV_RXY + 3]; \
    accum[2] += rA[0] * rB[2]; \
    accum[3] += rA[0] * rB[3]; \
    accum[6] += rA[1] * rB[2]; \
    accum[7] += rA[1] * rB[3]; \
    rA[0] = smA[ki + 1][lx * CONV_RXY]; \
    rA[1] = smA[ki + 1][lx * CONV_RXY + 1]; \
    accum[8] += rA[2] * rB[0]; \
    accum[9] += rA[2] * rB[1]; \
    accum[12] += rA[3] * rB[0]; \
    accum[13] += rA[3] * rB[1]; \
    rB[0] = smB[ki + 1][ly * CONV_RXY]; \
    rB[1] = smB[ki + 1][ly * CONV_RXY + 1]; \
    accum[10] += rA[2] * rB[2]; \
    accum[11] += rA[2] * rB[3]; \
    accum[14] += rA[3] * rB[2]; \
    accum[15] += rA[3] * rB[3]; \
  }

// global size IA * KA
// local size 256
__kernel void conv_preA(__global float *AU, __global float *A, int KU, int IA, int KA) {
  int gid = get_global_id(0);
  int bn = gid / (KA * CONV_BIJ), bo = gid - bn * (KA * CONV_BIJ);
  int k = bo / CONV_BIJ, i = bo % CONV_BIJ + bn * CONV_BIJ;
  A[gid] = k < KU ? AU[i * KU + k] : 0.0f;
}

// global size JA, IA, batch
// local size 256, 1, 1
__kernel void conv_postC(__global float *C, __global float *CU, int IA, int JA, int IU, int JU) {
  int c = get_global_id(2), i = get_global_id(1), j = get_global_id(0);
  if (j < JU) {
    CU[(c * IU + i) * JU + j] = C[(c * IA + i) * JA + j];
  }
}

// global size (JA / CONV_RXY), (IA / CONV_RXY), batch
// local size 16, 16, 1
__kernel void conv(__global float *A, __global float *B, __global float *C, __global float *D, int K, int IA, int JA, int KA, int CH, int N) {
  // +1 prevent overflow in innermost loop
  __local float smA[CONV_BK + 1][CONV_BIJ];
  __local float smB[CONV_BK + 1][CONV_BIJ];
  float rA[CONV_RXY], rB[CONV_RXY], accum[CONV_RXY * CONV_RXY] = {0};
  int gb = get_group_id(2);
  int gi = get_group_id(1), gj = get_group_id(0);
  int lx = get_local_id(1), ly = get_local_id(0);
  int lid = lx * CONV_LXY + ly;
  // CONV_BIJ
  int smx = lid >> 6, smy = lid & 63;
  int jj = gj * CONV_BIJ + smy, jf = jj < N * N;
  int hb = jj / N - 1, wb = jj % N - 1;
  int kk = smx, kk3, kk9, h, w;
  A += (gi * KA + smx) * CONV_BIJ + smy;
  B += gb * CH * N * N;

  smA[smx][smy] = A[0];
  kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
  smB[smx][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;
  smA[smx + 4][smy] = A[CONV_BIJ * 4];
  kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
  smB[smx + 4][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;
  for (; ; --K) {
    barrier(CLK_LOCAL_MEM_FENCE);

    smA[smx + 8][smy] = A[CONV_BIJ * 8];
    kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
    smB[smx + 8][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;

    GEMM44(0, 4);

    smA[smx + 12][smy] = A[CONV_BIJ * 12];
    kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
    smB[smx + 12][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;
    A += CONV_BIJ * 16;

    GEMM44(4, 8);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (K > 1) {
      smA[smx][smy] = A[0];
      kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
      smB[smx][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;
    }

    GEMM44(8, 12);

    if (K > 1) {
      smA[smx + 4][smy] = A[CONV_BIJ * 4];
      kk3 = DIV3(kk), kk9 = DIV3(kk3), w = kk - kk3 * 3 + wb, h = kk3 - kk9 * 3 + hb;
      smB[smx + 4][smy] = jf && kk9 < CH && 0 <= h && h < N && 0 <= w && w < N ? B[(kk9 * N + h) * N + w] : 0.0f, kk += 4;
    }

    GEMM44(12, 16);

    if (K == 1) break;
  }
  C += (gb * IA + gi * CONV_BIJ + lx * CONV_RXY) * JA + gj * CONV_BIJ + ly * CONV_RXY;
  D += gi * CONV_BIJ + lx * CONV_RXY;
  C[0] = max(accum[0] + D[0], 0.0f);
  C[1] = max(accum[1] + D[0], 0.0f);
  C[2] = max(accum[2] + D[0], 0.0f);
  C[3] = max(accum[3] + D[0], 0.0f);
  C[JA] = max(accum[4] + D[1], 0.0f);
  C[JA + 1] = max(accum[5] + D[1], 0.0f);
  C[JA + 2] = max(accum[6] + D[1], 0.0f);
  C[JA + 3] = max(accum[7] + D[1], 0.0f);
  C[JA * 2] = max(accum[8] + D[2], 0.0f);
  C[JA * 2 + 1] = max(accum[9] + D[2], 0.0f);
  C[JA * 2 + 2] = max(accum[10] + D[2], 0.0f);
  C[JA * 2 + 3] = max(accum[11] + D[2], 0.0f);
  C[JA * 3] = max(accum[12] + D[3], 0.0f);
  C[JA * 3 + 1] = max(accum[13] + D[3], 0.0f);
  C[JA * 3 + 2] = max(accum[14] + D[3], 0.0f);
  C[JA * 3 + 3] = max(accum[15] + D[3], 0.0f);
}

#define BLOCK_SIZE 16
__kernel void pool(__global float *DI, __global float *DO, int N, int C) {
  int bid = get_group_id(2);
  int ti = get_local_id(1), tj = get_local_id(0);
  int mi = get_global_id(1), mj = get_global_id(0);
  __local float a[BLOCK_SIZE][BLOCK_SIZE];
  int N2 = N << 1;
  if (mi < N2 && mj < N2)
    a[ti][tj] = DI[(bid * N2 + mi) * N2 + mj];
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ti & 1) == 0)
    a[ti][tj] = max(a[ti][tj], a[ti + 1][tj]);
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((ti & 1) == 0 && (tj & 1) == 0 && mi < N2 && mj < N2)
    DO[(bid * N + (mi >> 1)) * N + (mj >> 1)] = max(a[ti][tj], a[ti][tj + 1]);
}

#define FC_BLOCK_SIZE 256
__kernel void fc(__global float *DI, __global float *DO, __global float *W, __global float *B, int CI, int CO) {
  int bid = get_group_id(1), bi = get_group_id(0);
  W += bi * CI, DI += bid * CI;
  int ti = get_local_id(0);
  float s = 0.0f;
  for (int mk = ti; mk < CI; mk += FC_BLOCK_SIZE) {
    s += W[mk] * DI[mk];
  }
  __local float a[FC_BLOCK_SIZE];
  a[ti] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int ofs = FC_BLOCK_SIZE / 2; ofs > 0; ofs /= 2) {
    if (ti < ofs) {
      a[ti] += a[ti + ofs];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (ti == 0) {
    DO[bid * CO + bi] = max(a[0] + B[bi], 0.0f);
  }
}
