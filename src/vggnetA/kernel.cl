// Copyright 2016 HeeHoon Kim
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

__kernel void convWeightPad(__global float *DI, __global float *DO, int CI, int CO) {
  int KP = (CI * 9 + 1) / 2 * 2;
  for (int i = 0; i < CO; ++i) {
    int k = 0;
    for (; k < CI * 9; ++k) {
      DO[i * KP + k] = DI[i * CI * 9 + k];
    }
    for (; k < KP; ++k) {
      DO[i * KP + k] = 0.0f;
    }
  }
}

__kernel void convInputPad(__global float *DI, __global float *DO, int N, int CI) {
  int KP = (CI * 9 + 1) / 2 * 2;
  int JP = (N * N + 7) / 8 * 8;
  int gid = get_group_id(0);
  int k = 0;
  __global float *p0 = DO + gid * KP * JP;
  for (; k < CI * 9; ++k) {
    int ci = k / 9, ho = k / 3 % 3 - 1, wo = k % 3 - 1;
    __global float *p1 = DI + gid * CI * N * N + ci * N * N + ho * N + wo;
    for (int jh = 0; jh < N; ++jh) {
      for (int jw = 0; jw < N; ++jw) {
        int h = jh + ho, w = jw + wo;
        if (0 <= h && h < N && 0 <= w && w < N) {
          *p0++ = *p1;
        } else {
          *p0++ = 0.0f;
        }
        ++p1;
      }
    }
    for (int j = N * N; j < JP; ++j) {
      *p0++ = 0.0f;
    }
  }
  for (; k < KP; ++k) {
    for (int j = 0; j < JP; ++j) {
      *p0++ = 0.0f;
    }
  }
}

__kernel void convOutputUnpad(__global float *DI, __global float *DO, int N, int CO) {
  int JP = (N * N + 7) / 8 * 8;
  int gid = get_group_id(0);
  for (int i = 0; i < CO; ++i) {
    for (int j = 0; j < N * N; ++j) {
      DO[gid * CO * N * N + i * N * N + j] = DI[gid * CO * JP + i * JP + j];
    }
  }
}

__kernel void conv(__global float8 *DI, __global float8 *DO, __global float *W, __global float *B, int N, int CI, int CO) {
  int KP = (CI * 9 + 1) / 2 * 2;
  int JP = (N * N + 7) / 8 * 8;
  int JP8 = JP / 8;
  int gid = get_group_id(0);
  for (int i = 0; i < CO; i += 2) {
    __global float8 *p0 = DO + gid * CO * JP8 + i * JP8;
    float8 bias0 = (float8)(B[i]);
    float8 bias1 = (float8)(B[i + 1]);
    for (int j = 0; j < JP8; ++j) {
      p0[0] = bias0;
      p0[JP8] = bias1;
      ++p0;
    }
    __global float *p1 = W + i * KP;
    for (int k = 0; k < KP; k += 2) {
      float8 a00 = (float8)(p1[0]);
      float8 a10 = (float8)(p1[KP]);
      float8 a01 = (float8)(p1[1]);
      float8 a11 = (float8)(p1[KP + 1]);
      p1 += 2;
      __global float8 *p2 = DI + gid * KP * JP8 + k * JP8;
      __global float8 *p3 = DO + gid * CO * JP8 + i * JP8;
      for (int j = 0; j < JP8; ++j) {
        float8 b0 = p2[0];
        float8 b1 = p2[JP8];
        float8 c0 = p3[0];
        float8 c1 = p3[JP8];
        c0 += a00 * b0;
        c1 += a10 * b0;
        c0 += a01 * b1;
        c1 += a11 * b1;
        p3[0] = c0;
        p3[JP8] = c1;
        ++p2;
        ++p3;
      }
    }
    __global float8 *p4 = DO + gid * CO * JP8 + i * JP8;
    for (int j = 0; j < JP8; ++j) {
      p4[0] = max(p4[0], 0.0f);
      p4[JP8] = max(p4[JP8], 0.0f);
      ++p4;
    }
  }
}

__kernel void pool(__global float2 *DI, __global float *DO, int N, int C) {
  int gid = get_group_id(0);
  __global float2 *p0 = DI + gid * C * N * N * 2;
  __global float *p1 = DO + gid * C * N * N;
  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        float2 m = max(p0[0], p0[N]);
        *p1++ = max(m.x, m.y);
        ++p0;
      }
      p0 += N;
    }
  }
}

__kernel void fc(__global float8 *DI, __global float *DO, __global float8 *W, __global float *B, int CI, int CO) {
  int gid = get_group_id(0);
  __global float *p0 = DO + gid * CO;
  for (int i = 0; i < CO; ++i) {
    float8 s = (float8)(0.0f);
    for (int k = 0; k < CI; k += 8) {
      float8 a = W[(i * CI + k) / 8];
      float8 b = DI[(gid * CI + k) / 8];
      s += a * b;
    }
    float4 f4 = s.lo + s.hi;
    float2 f2 = f4.lo + f4.hi;
    *p0++ = max(f2.x + f2.y + B[i], 0.0f);
  }
}

__kernel void softmax(__global float *DI, __global float *DO, __global int *label, int N) {
  int gid = get_group_id(0);
  int mi = 0;
  float max = DI[gid * N];
  for (int i = 1; i < N; ++i) {
    float t = DI[gid * N + i];
    if (max < t) {
      max = t;
      mi = i;
    }
  }
  label[gid] = mi;
  float s = 0;
  for (int i = 0; i < N; ++i) {
    float t = exp(DI[gid * N + i] - max);
    DO[gid * N + i] = t;
    s += t;
  }
  for (int i = 0; i < N; ++i) {
    DO[gid * N + i] /= s;
  }
}
