// Copyright 2016 HeeHoon Kim
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#include <pthread.h>
#include <mpi.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

#define DEBUG 0

#define NUM_GPU 1
#define MAX_BATCH 32

#define CONV_BK 16
#define CONV_BKH 8
#define CONV_BIJ 64
#define CONV_LXY 16
#define CONV_RXY 4

cl_int err;
cl_platform_id platform;
cl_device_id dev[4];
cl_context context;
float *images, *network, *confidences;
int *labels, num_images, mpi_rank, mpi_size;

struct Global {
  cl_device_id dev;
  cl_command_queue queue[2];
  cl_program program;
  cl_kernel kernel_conv, kernel_conv_preA, kernel_conv_postC;
  cl_kernel kernel_pool, kernel_fc;
  int batch;
  cl_mem mem_convA, mem_convC;
  struct timespec start;
} G[NUM_GPU];

static int timespec_subtract(struct timespec* result, struct timespec *x, struct timespec *y) {
  if (x->tv_nsec < y->tv_nsec) {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000) {
    int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  return x->tv_sec < y->tv_sec;
}

static void mystart(int tid) {
  clock_gettime(CLOCK_MONOTONIC, &G[tid].start);
}

static void myend(const char *label, int tid) {
  struct timespec end, spent;
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &G[tid].start);
  printf("[Thread%d:%s] Elapsed time: %ld.%03ld sec\n", tid, label, spent.tv_sec, spent.tv_nsec/1000/1000);
}

static int myceil(int x, int r) {
  return (x + r - 1) / r * r;
}

static int mymin(int x, int y) {
  return x < y ? x : y;
}

static char *get_source_code(const char *file_name, size_t *len) {
  char *source_code;
  size_t length;
  FILE *file = fopen(file_name, "r");
  if (file == NULL) {
    printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  length = (size_t)ftell(file);
  rewind(file);

  source_code = (char *)malloc(length + 1);
  fread(source_code, length, 1, file);
  source_code[length] = '\0';

  fclose(file);

  *len = length;
  return source_code;
}

static cl_program get_program(const char *file_name, cl_context context, cl_device_id *device, int nDevice) {
  const char *source_code;
  size_t source_size;
  source_code = get_source_code(file_name, &source_size);

  cl_program program;
  program = clCreateProgramWithSource(context, 1, &source_code, &source_size, &err);
  CHECK_ERROR(err);

  err = clBuildProgram(program, nDevice, device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    for (int i = 0; i < nDevice; ++i) {
      char *log;
      size_t log_size;
      clGetProgramBuildInfo(program, device[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      log = (char*)malloc(log_size + 1);
      clGetProgramBuildInfo(program, device[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      log[log_size] = 0;
      printf("Compile error:\n%s\n", log);
      free(log);
    }
  }
  CHECK_ERROR(err);

  return program;
}

static void debug(cl_mem m, int sz, const char *fn, int tid) {
  struct Global *g = &G[tid];

  float *buf = (float*)malloc(sizeof(float) * sz);
  err = clEnqueueReadBuffer(g->queue[0], m, CL_TRUE, 0, sizeof(float) * sz, buf, 0, NULL, NULL);
  CHECK_ERROR(err);
  FILE *o = fopen(fn, "w");
  for (int j = 0; j < sz; ++j) {
    fprintf(o, "%f\n", buf[j]);
  }
  free(buf);
}

static void convolution_layer(cl_mem inputs, cl_mem outputs, cl_mem filters, cl_mem biases, int N, int D1, int D2, int tid)
{
  struct Global *g = &G[tid];

  int IU = D2, JU = N * N, KU = D1 * 9;
  int IA = myceil(IU, CONV_BIJ);
  int JA = myceil(JU, CONV_BIJ);
  int KA = myceil(KU, CONV_BK);
  int K = KA / CONV_BK;
  int cf = JU != JA;

  {
    err = clSetKernelArg(g->kernel_conv, 0, sizeof(cl_mem), &filters);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv, 1, sizeof(cl_mem), &inputs);
    CHECK_ERROR(err);
    if (cf) {
      err = clSetKernelArg(g->kernel_conv, 2, sizeof(cl_mem), &g->mem_convC);
      CHECK_ERROR(err);
    } else {
      err = clSetKernelArg(g->kernel_conv, 2, sizeof(cl_mem), &outputs);
      CHECK_ERROR(err);
    }
    err = clSetKernelArg(g->kernel_conv, 3, sizeof(cl_mem), &biases);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv, 4, sizeof(int), &K);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv, 5, sizeof(int), &IA);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv, 6, sizeof(int), &JA);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv, 7, sizeof(int), &KA);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv, 8, sizeof(int), &D1);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv, 9, sizeof(int), &N);
    CHECK_ERROR(err);
    size_t global_size[3] = {JA / CONV_RXY, IA / CONV_RXY, g->batch};
    size_t local_size[3] = {CONV_LXY, CONV_LXY, 1};
    err = clEnqueueNDRangeKernel(g->queue[0], g->kernel_conv, 3, NULL, global_size, local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  if (cf) {
    err = clSetKernelArg(g->kernel_conv_postC, 0, sizeof(cl_mem), &g->mem_convC);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv_postC, 1, sizeof(cl_mem), &outputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv_postC, 2, sizeof(int), &IA);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv_postC, 3, sizeof(int), &JA);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv_postC, 4, sizeof(int), &IU);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv_postC, 5, sizeof(int), &JU);
    CHECK_ERROR(err);
    size_t global_size[3] = {JA, IA, g->batch};
    size_t local_size[3] = {CONV_BIJ, 256 / CONV_BIJ, 1};
    err = clEnqueueNDRangeKernel(g->queue[0], g->kernel_conv_postC, 3, NULL, global_size, local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  if (DEBUG) {
    clFinish(g->queue[0]);
    myend("CONV", tid);
    mystart(tid);
  }
}

static void pooling_layer(cl_mem inputs, cl_mem outputs, int N, int D, int tid)
{
  struct Global *g = &G[tid];

  err = clSetKernelArg(g->kernel_pool, 0, sizeof(cl_mem), &inputs);
  CHECK_ERROR(err);
  err = clSetKernelArg(g->kernel_pool, 1, sizeof(cl_mem), &outputs);
  CHECK_ERROR(err);
  err = clSetKernelArg(g->kernel_pool, 2, sizeof(int), &N);
  CHECK_ERROR(err);
  err = clSetKernelArg(g->kernel_pool, 3, sizeof(int), &D);
  CHECK_ERROR(err);
  size_t global_size[3] = {myceil(N * 2, 16), myceil(N * 2, 16), D * g->batch};
  size_t local_size[3] = {16, 16, 1};
  err = clEnqueueNDRangeKernel(g->queue[0], g->kernel_pool, 3, NULL, global_size, local_size, 0, NULL, NULL);
  CHECK_ERROR(err);

  if (DEBUG) {
    clFinish(g->queue[0]);
    myend("POOL", tid);
    mystart(tid);
  }
}

static void fc_layer(cl_mem input_neuron, cl_mem output_neuron, cl_mem weights, cl_mem biases, int N, int M, int tid)
{
  struct Global *g = &G[tid];

  err = clSetKernelArg(g->kernel_fc, 0, sizeof(cl_mem), &input_neuron);
  CHECK_ERROR(err);
  err = clSetKernelArg(g->kernel_fc, 1, sizeof(cl_mem), &output_neuron);
  CHECK_ERROR(err);
  err = clSetKernelArg(g->kernel_fc, 2, sizeof(cl_mem), &weights);
  CHECK_ERROR(err);
  err = clSetKernelArg(g->kernel_fc, 3, sizeof(cl_mem), &biases);
  CHECK_ERROR(err);
  err = clSetKernelArg(g->kernel_fc, 4, sizeof(int), &N);
  CHECK_ERROR(err);
  err = clSetKernelArg(g->kernel_fc, 5, sizeof(int), &M);
  CHECK_ERROR(err);
  size_t global_size[2] = {256 * M, g->batch};
  size_t local_size[2] = {256, 1};
  err = clEnqueueNDRangeKernel(g->queue[0], g->kernel_fc, 2, NULL, global_size, local_size, 0, NULL, NULL);
  CHECK_ERROR(err);

  if (DEBUG) {
    clFinish(g->queue[0]);
    myend("FC", tid);
    mystart(tid);
  }
}

static void softmax(float *D, int *label, int N, int tid)
{
  struct Global *g = &G[tid];

  for (int i = 0; i < g->batch; ++i) {
    float m = D[i * N];
    int mi = 0;
    for (int j = 1; j < N; ++j) {
      float t = D[i * N + j];
      if (m < t) {
        m = t;
        mi = j;
      }
    }
    label[i] = mi;
    float s = 0;
    for (int j = 0; j < N; ++j) {
      float t = exp(D[i * N + j] - m);
      D[i * N + j] = t;
      s += t;
    }
    for (int j = 0; j < N; ++j) {
      D[i * N + j] /= s;
    }
  }
}

static void get_param(float ** array, int size, cl_mem *m, int D1, int D2, int tid, int qid)
{
  struct Global *g = &G[tid];

  if (D1) {
    int IU = D2, KU = D1 * 9;
    int IA = myceil(IU, CONV_BIJ);
    int KA = myceil(KU, CONV_BK);

    err = clEnqueueWriteBuffer(g->queue[qid], g->mem_convA, CL_FALSE, 0, sizeof(float) * size, *array, 0, NULL, NULL);
    CHECK_ERROR(err);
    *array += size;

    *m = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * KA * IA, NULL, &err);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv_preA, 0, sizeof(cl_mem), &g->mem_convA);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv_preA, 1, sizeof(cl_mem), m);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv_preA, 2, sizeof(int), &KU);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv_preA, 3, sizeof(int), &IA);
    CHECK_ERROR(err);
    err = clSetKernelArg(g->kernel_conv_preA, 4, sizeof(int), &KA);
    CHECK_ERROR(err);
    size_t global_size[] = {IA * KA};
    size_t local_size[] = {256};
    err = clEnqueueNDRangeKernel(g->queue[qid], g->kernel_conv_preA, 1, NULL, global_size, local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
  } else {
    *m = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(g->queue[qid], *m, CL_FALSE, 0, sizeof(float) * size, *array, 0, NULL, NULL);
    CHECK_ERROR(err);
    *array += size;
  }
}

void *proc(void *arg) {
  int tid = (size_t)arg;
  struct Global *g = &G[tid];
  g->dev = dev[mpi_rank % 4];

  if (DEBUG) {
    mystart(tid);
  }

  context = clCreateContext(NULL, 1, &g->dev, NULL, NULL, &err);
  CHECK_ERROR(err);

  g->queue[0] = clCreateCommandQueue(context, g->dev, 0, &err);
  CHECK_ERROR(err);
  g->queue[1] = clCreateCommandQueue(context, g->dev, 0, &err);
  CHECK_ERROR(err);

  if (DEBUG) {
    myend("OpenCL INIT / thread", tid);
    mystart(tid);
  }

  cl_mem mem[2];
  cl_mem f1_1, f1_2, f2_1, f2_2, f3_1, f3_2, f3_3, f4_1, f4_2, f4_3, f5_1, f5_2, f5_3, w1, w2, w3; // Filters and weights
  cl_mem b1_1, b1_2, b2_1, b2_2, b3_1, b3_2, b3_3, b4_1, b4_2, b4_3, b5_1, b5_2, b5_3, b1, b2, b3; // Biases

  mem[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 64 * 224 * 224, NULL, &err);
  CHECK_ERROR(err);
  mem[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 64 * 224 * 224, NULL, &err);
  CHECK_ERROR(err);

  g->mem_convA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
  CHECK_ERROR(err);
  g->mem_convC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 512 * 832, NULL, &err);
  CHECK_ERROR(err);

  if (DEBUG) {
    myend("CREATE BUFFER", tid);
    mystart(tid);
  }

  g->program = get_program("kernel.cl", context, &g->dev, 1);
  g->kernel_conv = clCreateKernel(g->program, "conv", &err);
  CHECK_ERROR(err);
  g->kernel_conv_preA = clCreateKernel(g->program, "conv_preA", &err);
  CHECK_ERROR(err);
  g->kernel_conv_postC = clCreateKernel(g->program, "conv_postC", &err);
  CHECK_ERROR(err);
  g->kernel_pool = clCreateKernel(g->program, "pool", &err);
  CHECK_ERROR(err);
  g->kernel_fc = clCreateKernel(g->program, "fc", &err);
  CHECK_ERROR(err);

  if (DEBUG) {
    myend("COMPILE", tid);
    mystart(tid);
  }

  float *mynet = network;
  get_param(&mynet, 3 * 3 * 3 * 64, &f1_1, 3, 64, tid, 0);
  get_param(&mynet, 64, &b1_1, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 64 * 64, &f1_2, 64, 64, tid, 0);
  get_param(&mynet, 64, &b1_2, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 64 * 128, &f2_1, 64, 128, tid, 0);
  get_param(&mynet, 128, &b2_1, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 128 * 128, &f2_2, 128, 128, tid, 0);
  get_param(&mynet, 128, &b2_2, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 128 * 256, &f3_1, 128, 256, tid, 0);
  get_param(&mynet, 256, &b3_1, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 256 * 256, &f3_2, 256, 256, tid, 0);
  get_param(&mynet, 256, &b3_2, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 256 * 256, &f3_3, 256, 256, tid, 0);
  get_param(&mynet, 256, &b3_3, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 256 * 512, &f4_1, 256, 512, tid, 0);
  get_param(&mynet, 512, &b4_1, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 512 * 512, &f4_2, 512, 512, tid, 0);
  get_param(&mynet, 512, &b4_2, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 512 * 512, &f4_3, 512, 512, tid, 0);
  get_param(&mynet, 512, &b4_3, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 512 * 512, &f5_1, 512, 512, tid, 0);
  get_param(&mynet, 512, &b5_1, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 512 * 512, &f5_2, 512, 512, tid, 0);
  get_param(&mynet, 512, &b5_2, 0, 0, tid, 0);
  get_param(&mynet, 3 * 3 * 512 * 512, &f5_3, 512, 512, tid, 0);
  get_param(&mynet, 512, &b5_3, 0, 0, tid, 0);
  get_param(&mynet, 7 * 7 * 512 * 4096, &w1, 0, 0, tid, 0);
  get_param(&mynet, 4096, &b1, 0, 0, tid, 0);
  get_param(&mynet, 4096 * 4096, &w2, 0, 0, tid, 0);
  get_param(&mynet, 4096, &b2, 0, 0, tid, 0);
  get_param(&mynet, 4096 * 1000, &w3, 0, 0, tid, 0);
  get_param(&mynet, 1000, &b3, 0, 0, tid, 0);

  float *fc3_host;
  fc3_host = (float*)malloc(sizeof(float) * MAX_BATCH * 1000);

  if (DEBUG) {
    clFinish(g->queue[0]);
    myend("NET SETUP", tid);
    mystart(tid);
  }

  int mq = num_images / mpi_size;
  int mr = num_images % mpi_size;
  int ms = mq * mpi_rank + mymin(mr, mpi_rank);
  int me = mq * (mpi_rank + 1) + mymin(mr, mpi_rank + 1);
  for(int i = ms; i < me; i += MAX_BATCH)
  {
    g->batch = mymin(me - i, MAX_BATCH);

    err = clEnqueueWriteBuffer(g->queue[0], mem[0], CL_FALSE, 0, sizeof(float) * g->batch * 224 * 224 * 3, images + i * 224 * 224 * 3, 0, NULL, NULL);
    CHECK_ERROR(err);

    if (DEBUG) {
      clFinish(g->queue[0]);
      myend("WRITE", tid);
      mystart(tid);
    }

    convolution_layer(mem[0], mem[1], f1_1, b1_1, 224, 3, 64, tid);
    convolution_layer(mem[1], mem[0], f1_2, b1_2, 224, 64, 64, tid);
    pooling_layer(mem[0], mem[1], 112, 64, tid);

    convolution_layer(mem[1], mem[0], f2_1, b2_1, 112, 64, 128, tid);
    convolution_layer(mem[0], mem[1], f2_2, b2_2, 112, 128, 128, tid);
    pooling_layer(mem[1], mem[0], 56, 128, tid);

    convolution_layer(mem[0], mem[1], f3_1, b3_1, 56, 128, 256, tid);
    convolution_layer(mem[1], mem[0], f3_2, b3_2, 56, 256, 256, tid);
    convolution_layer(mem[0], mem[1], f3_3, b3_3, 56, 256, 256, tid);
    pooling_layer(mem[1], mem[0], 28, 256, tid);

    convolution_layer(mem[0], mem[1], f4_1, b4_1, 28, 256, 512, tid);
    convolution_layer(mem[1], mem[0], f4_2, b4_2, 28, 512, 512, tid);
    convolution_layer(mem[0], mem[1], f4_3, b4_3, 28, 512, 512, tid);
    pooling_layer(mem[1], mem[0], 14, 512, tid);

    convolution_layer(mem[0], mem[1], f5_1, b5_1, 14, 512, 512, tid);
    convolution_layer(mem[1], mem[0], f5_2, b5_2, 14, 512, 512, tid);
    convolution_layer(mem[0], mem[1], f5_3, b5_3, 14, 512, 512, tid);
    pooling_layer(mem[1], mem[0], 7, 512, tid);

    if (i == ms) {
      clFinish(g->queue[1]);
    }

    fc_layer(mem[0], mem[1], w1, b1, 7 * 7 * 512, 4096, tid);
    fc_layer(mem[1], mem[0], w2, b2, 4096, 4096, tid);
    fc_layer(mem[0], mem[1], w3, b3, 4096, 1000, tid);

    err = clEnqueueReadBuffer(g->queue[0], mem[1], CL_TRUE, 0, sizeof(float) * g->batch * 1000, fc3_host, 0, NULL, NULL);
    CHECK_ERROR(err);

    if (DEBUG) {
      myend("READ", tid);
      mystart(tid);
    }

    softmax(fc3_host, labels + i, 1000, tid);

    for (int j = 0; j < g->batch; ++j) {
      confidences[i + j] = fc3_host[j * 1000 + labels[i + j]];
    }

    if (DEBUG) {
      myend("SOFTMAX", tid);
      mystart(tid);
    }
  }

  clReleaseMemObject(mem[0]);
  clReleaseMemObject(mem[1]);
  free(fc3_host);

  clReleaseMemObject(g->mem_convA);
  clReleaseMemObject(g->mem_convC);
  clReleaseMemObject(f1_1);
  clReleaseMemObject(f1_2);
  clReleaseMemObject(f2_1);
  clReleaseMemObject(f2_2);
  clReleaseMemObject(f3_1);
  clReleaseMemObject(f3_2);
  clReleaseMemObject(f3_3);
  clReleaseMemObject(f4_1);
  clReleaseMemObject(f4_2);
  clReleaseMemObject(f4_3);
  clReleaseMemObject(f5_1);
  clReleaseMemObject(f5_2);
  clReleaseMemObject(f5_3);
  clReleaseMemObject(w1);
  clReleaseMemObject(w2);
  clReleaseMemObject(w3);
  clReleaseMemObject(b1_1);
  clReleaseMemObject(b1_2);
  clReleaseMemObject(b2_1);
  clReleaseMemObject(b2_2);
  clReleaseMemObject(b3_1);
  clReleaseMemObject(b3_2);
  clReleaseMemObject(b3_3);
  clReleaseMemObject(b4_1);
  clReleaseMemObject(b4_2);
  clReleaseMemObject(b4_3);
  clReleaseMemObject(b5_1);
  clReleaseMemObject(b5_2);
  clReleaseMemObject(b5_3);
  clReleaseMemObject(b1);
  clReleaseMemObject(b2);
  clReleaseMemObject(b3);

  clReleaseKernel(g->kernel_conv);
  clReleaseKernel(g->kernel_conv_preA);
  clReleaseKernel(g->kernel_conv_postC);
  clReleaseKernel(g->kernel_pool);
  clReleaseKernel(g->kernel_fc);
  clReleaseProgram(g->program);
  clReleaseCommandQueue(g->queue[0]);
  clReleaseCommandQueue(g->queue[1]);
  clReleaseContext(context);

  if (DEBUG) {
    myend("RELEASE", tid);
  }

  return NULL;
}

void vggnet(float *_images, float *_network, int *_labels, float *_confidences, int _num_images)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (DEBUG) {
    mystart(0);
  }

  images = _images, network = _network, labels = _labels, confidences = _confidences, num_images = _num_images;
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 4, dev, NULL);
  CHECK_ERROR(err);

  if (DEBUG) {
    myend("OpenCL INIT", 0);
  }

  proc(0);

  for (int i = 0; i < mpi_size; ++i) {
    int mq = num_images / mpi_size;
    int mr = num_images % mpi_size;
    int ms = mq * i + mymin(mr, i);
    int me = mq * (i + 1) + mymin(mr, i + 1);
    MPI_Bcast(labels + ms, me - ms, MPI_INT, i, MPI_COMM_WORLD);
    MPI_Bcast(confidences + ms, me - ms, MPI_FLOAT, i, MPI_COMM_WORLD);
  }
}
