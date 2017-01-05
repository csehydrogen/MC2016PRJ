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
#include <math.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

const size_t MAX_BATCH = 16;

cl_int err;
cl_platform_id platform;
cl_device_id dev;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel_conv, kernel_convInputPad, kernel_convWeightPad, kernel_convOutputUnpad;
cl_kernel kernel_pool, kernel_fc, kernel_softmax;
int batch;

char *get_source_code(const char *file_name, size_t *len) {
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

cl_program get_program(const char *file_name, cl_context context, cl_device_id *device, int nDevice) {
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

static void pooling_layer(cl_mem inputs, cl_mem outputs, int N, int D)
{
  err = clSetKernelArg(kernel_pool, 0, sizeof(cl_mem), &inputs);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_pool, 1, sizeof(cl_mem), &outputs);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_pool, 2, sizeof(int), &N);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_pool, 3, sizeof(int), &D);
  CHECK_ERROR(err);
  size_t global_size[1] = {batch};
  size_t local_size[1] = {1};
  err = clEnqueueNDRangeKernel(queue, kernel_pool, 1, NULL, global_size, local_size, 0, NULL, NULL);
  CHECK_ERROR(err);
}

static void pad_weight(cl_mem DI, cl_mem DO, int CI, int CO) {
  err = clSetKernelArg(kernel_convWeightPad, 0, sizeof(cl_mem), &DI);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convWeightPad, 1, sizeof(cl_mem), &DO);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convWeightPad, 2, sizeof(int), &CI);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convWeightPad, 3, sizeof(int), &CO);
  CHECK_ERROR(err);
  size_t global_size[1] = {1};
  size_t local_size[1] = {1};
  err = clEnqueueNDRangeKernel(queue, kernel_convWeightPad, 1, NULL, global_size, local_size, 0, NULL, NULL);
  CHECK_ERROR(err);
}

#define ReLU(x) (((x)>0)?(x):0)
static void convolution_layer(cl_mem inputs, cl_mem outputs, cl_mem filters, cl_mem biases, int N, int D1, int D2)
{
  int JF = (N * N) % 8 != 0;
  int JP = (N * N + 7) / 8 * 8;
  int KP = (D1 * 9 + 1) / 2 * 2;

  cl_mem DI = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * KP * JP, NULL, &err);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convInputPad, 0, sizeof(cl_mem), &inputs);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convInputPad, 1, sizeof(cl_mem), &DI);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convInputPad, 2, sizeof(int), &N);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_convInputPad, 3, sizeof(int), &D1);
  CHECK_ERROR(err);
  size_t global_size[1] = {batch};
  size_t local_size[1] = {1};
  err = clEnqueueNDRangeKernel(queue, kernel_convInputPad, 1, NULL, global_size, local_size, 0, NULL, NULL);
  CHECK_ERROR(err);

  cl_mem DO = outputs;
  if (JF) {
    DO = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * D2 * JP, NULL, &err);
    CHECK_ERROR(err);
  }

  {
    err = clSetKernelArg(kernel_conv, 0, sizeof(cl_mem), &DI);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 1, sizeof(cl_mem), &DO);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 2, sizeof(cl_mem), &filters);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 3, sizeof(cl_mem), &biases);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 4, sizeof(int), &N);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 5, sizeof(int), &D1);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 6, sizeof(int), &D2);
    CHECK_ERROR(err);
    size_t global_size[1] = {batch};
    size_t local_size[1] = {1};
    err = clEnqueueNDRangeKernel(queue, kernel_conv, 1, NULL, global_size, local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  clReleaseMemObject(DI);

  if (JF) {
    err = clSetKernelArg(kernel_convOutputUnpad, 0, sizeof(cl_mem), &DO);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_convOutputUnpad, 1, sizeof(cl_mem), &outputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_convOutputUnpad, 2, sizeof(int), &N);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_convOutputUnpad, 3, sizeof(int), &D2);
    CHECK_ERROR(err);
    size_t global_size[1] = {batch};
    size_t local_size[1] = {1};
    err = clEnqueueNDRangeKernel(queue, kernel_convOutputUnpad, 1, NULL, global_size, local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
    clReleaseMemObject(DO);
  }
}

static void fc_layer(cl_mem input_neuron, cl_mem output_neuron, cl_mem weights, cl_mem biases, int N, int M)
{
  err = clSetKernelArg(kernel_fc, 0, sizeof(cl_mem), &input_neuron);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_fc, 1, sizeof(cl_mem), &output_neuron);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_fc, 2, sizeof(cl_mem), &weights);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_fc, 3, sizeof(cl_mem), &biases);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_fc, 4, sizeof(int), &N);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_fc, 5, sizeof(int), &M);
  CHECK_ERROR(err);
  size_t global_size[1] = {batch};
  size_t local_size[1] = {1};
  err = clEnqueueNDRangeKernel(queue, kernel_fc, 1, NULL, global_size, local_size, 0, NULL, NULL);
  CHECK_ERROR(err);
}

static void softmax(cl_mem input, cl_mem output, cl_mem label, int N)
{
  err = clSetKernelArg(kernel_softmax, 0, sizeof(cl_mem), &input);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_softmax, 1, sizeof(cl_mem), &output);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_softmax, 2, sizeof(cl_mem), &label);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_softmax, 3, sizeof(int), &N);
  CHECK_ERROR(err);
  size_t global_size[1] = {batch};
  size_t local_size[1] = {1};
  err = clEnqueueNDRangeKernel(queue, kernel_softmax, 1, NULL, global_size, local_size, 0, NULL, NULL);
  CHECK_ERROR(err);
}

static void get_param(float ** array, int size, cl_mem m)
{
  err = clEnqueueWriteBuffer(queue, m, CL_FALSE, 0, sizeof(float) * size, *array, 0, NULL, NULL);
  CHECK_ERROR(err);
  *array += size;
}

static void debug(cl_mem m, int sz, const char *fn) {
  float *buf = (float*)malloc(sizeof(float) * sz);
  err = clEnqueueReadBuffer(queue, m, CL_TRUE, 0, sizeof(float) * sz, buf, 0, NULL, NULL);
  CHECK_ERROR(err);
  FILE *o = fopen(fn, "w");
  for (int j = 0; j < sz; ++j) {
    fprintf(o, "%f\n", buf[j]);
  }
  free(buf);
}

void vggnet(float * images, float * network, int * labels, float * confidences, int num_images)
{
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
  CHECK_ERROR(err);

  context = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
  CHECK_ERROR(err);

  queue = clCreateCommandQueue(context, dev, 0, &err);
  CHECK_ERROR(err);

  program = get_program("kernel.cl", context, &dev, 1);

  kernel_conv = clCreateKernel(program, "conv", &err);
  CHECK_ERROR(err);
  kernel_convInputPad = clCreateKernel(program, "convInputPad", &err);
  CHECK_ERROR(err);
  kernel_convWeightPad = clCreateKernel(program, "convWeightPad", &err);
  CHECK_ERROR(err);
  kernel_convOutputUnpad = clCreateKernel(program, "convOutputUnpad", &err);
  CHECK_ERROR(err);
  kernel_pool = clCreateKernel(program, "pool", &err);
  CHECK_ERROR(err);
  kernel_fc = clCreateKernel(program, "fc", &err);
  CHECK_ERROR(err);
  kernel_softmax = clCreateKernel(program, "softmax", &err);
  CHECK_ERROR(err);

  cl_mem data; // image layer
  cl_mem c1_1, c1_2, c2_1, c2_2, c3_1, c3_2, c3_3, c4_1, c4_2, c4_3, c5_1, c5_2, c5_3; // Convolution layers
  cl_mem p1, p2, p3, p4, p5; // Pooling layers
  cl_mem fc1, fc2, fc3; // Fully connected layers
  cl_mem layer_softmax, layer_label;
  cl_mem f1_1, f1_1p, f1_2, f2_1, f2_2, f3_1, f3_2, f3_3, f4_1, f4_2, f4_3, f5_1, f5_2, f5_3, w1, w2, w3; // Filters and weights
  cl_mem b1_1, b1_2, b2_1, b2_2, b3_1, b3_2, b3_3, b4_1, b4_2, b4_3, b5_1, b5_2, b5_3, b1, b2, b3; // Biases

  data = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 224 * 224 * 3, NULL, &err);
  CHECK_ERROR(err);
  c1_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 224 * 224 * 64, NULL, &err);
  CHECK_ERROR(err);
  c1_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 224 * 224 * 64, NULL, &err);
  CHECK_ERROR(err);
  p1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 112 * 112 * 64, NULL, &err);
  CHECK_ERROR(err);
  c2_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 112 * 112 * 128, NULL, &err);
  CHECK_ERROR(err);
  c2_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 112 * 112 * 128, NULL, &err);
  CHECK_ERROR(err);
  p2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 56 * 56 * 128, NULL, &err);
  CHECK_ERROR(err);
  c3_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 56 * 56 * 256, NULL, &err);
  CHECK_ERROR(err);
  c3_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 56 * 56 * 256, NULL, &err);
  CHECK_ERROR(err);
  c3_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 56 * 56 * 256, NULL, &err);
  CHECK_ERROR(err);
  p3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 28 * 28 * 256, NULL, &err);
  CHECK_ERROR(err);
  c4_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 28 * 28 * 512, NULL, &err);
  CHECK_ERROR(err);
  c4_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 28 * 28 * 512, NULL, &err);
  CHECK_ERROR(err);
  c4_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 28 * 28 * 512, NULL, &err);
  CHECK_ERROR(err);
  p4 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 14 * 14 * 512, NULL, &err);
  CHECK_ERROR(err);
  c5_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 14 * 14 * 512, NULL, &err);
  CHECK_ERROR(err);
  c5_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 14 * 14 * 512, NULL, &err);
  CHECK_ERROR(err);
  c5_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 14 * 14 * 512, NULL, &err);
  CHECK_ERROR(err);
  p5 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 7 * 7 * 512, NULL, &err);
  CHECK_ERROR(err);
  fc1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 4096, NULL, &err);
  CHECK_ERROR(err);
  fc2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 4096, NULL, &err);
  CHECK_ERROR(err);
  fc3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 1000, NULL, &err);
  CHECK_ERROR(err);
  layer_softmax = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * MAX_BATCH * 1000, NULL, &err);
  CHECK_ERROR(err);
  layer_label = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * MAX_BATCH, NULL, &err);
  CHECK_ERROR(err);

  f1_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 3 * 64, NULL, &err);
  CHECK_ERROR(err);
  f1_1p = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 4 * 64, NULL, &err);
  CHECK_ERROR(err);
  b1_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64, NULL, &err);
  CHECK_ERROR(err);
  f1_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 64 * 64, NULL, &err);
  CHECK_ERROR(err);
  b1_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64, NULL, &err);
  CHECK_ERROR(err);
  f2_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 64 * 128, NULL, &err);
  CHECK_ERROR(err);
  b2_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 128, NULL, &err);
  CHECK_ERROR(err);
  f2_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 128 * 128, NULL, &err);
  CHECK_ERROR(err);
  b2_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 128, NULL, &err);
  CHECK_ERROR(err);
  f3_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 128 * 256, NULL, &err);
  CHECK_ERROR(err);
  b3_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 256, NULL, &err);
  CHECK_ERROR(err);
  f3_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 256 * 256, NULL, &err);
  CHECK_ERROR(err);
  b3_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 256, NULL, &err);
  CHECK_ERROR(err);
  f3_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 256 * 256, NULL, &err);
  CHECK_ERROR(err);
  b3_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 256, NULL, &err);
  CHECK_ERROR(err);
  f4_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 256 * 512, NULL, &err);
  CHECK_ERROR(err);
  b4_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512, NULL, &err);
  CHECK_ERROR(err);
  f4_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
  CHECK_ERROR(err);
  b4_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512, NULL, &err);
  CHECK_ERROR(err);
  f4_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
  CHECK_ERROR(err);
  b4_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512, NULL, &err);
  CHECK_ERROR(err);
  f5_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
  CHECK_ERROR(err);
  b5_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512, NULL, &err);
  CHECK_ERROR(err);
  f5_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
  CHECK_ERROR(err);
  b5_2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512, NULL, &err);
  CHECK_ERROR(err);
  f5_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 512 * 512, NULL, &err);
  CHECK_ERROR(err);
  b5_3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512, NULL, &err);
  CHECK_ERROR(err);
  w1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 7 * 7 * 512 * 4096, NULL, &err);
  CHECK_ERROR(err);
  b1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4096, NULL, &err);
  CHECK_ERROR(err);
  w2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4096 * 4096, NULL, &err);
  CHECK_ERROR(err);
  b2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4096, NULL, &err);
  CHECK_ERROR(err);
  w3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4096 * 1000, NULL, &err);
  CHECK_ERROR(err);
  b3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 1000, NULL, &err);
  CHECK_ERROR(err);

  get_param(&network, 3 * 3 * 3 * 64, f1_1);
  pad_weight(f1_1, f1_1p, 3, 64);
  get_param(&network, 64, b1_1);
  get_param(&network, 3 * 3 * 64 * 64, f1_2);
  get_param(&network, 64, b1_2);
  get_param(&network, 3 * 3 * 64 * 128, f2_1);
  get_param(&network, 128, b2_1);
  get_param(&network, 3 * 3 * 128 * 128, f2_2);
  get_param(&network, 128, b2_2);
  get_param(&network, 3 * 3 * 128 * 256, f3_1);
  get_param(&network, 256, b3_1);
  get_param(&network, 3 * 3 * 256 * 256, f3_2);
  get_param(&network, 256, b3_2);
  get_param(&network, 3 * 3 * 256 * 256, f3_3);
  get_param(&network, 256, b3_3);
  get_param(&network, 3 * 3 * 256 * 512, f4_1);
  get_param(&network, 512, b4_1);
  get_param(&network, 3 * 3 * 512 * 512, f4_2);
  get_param(&network, 512, b4_2);
  get_param(&network, 3 * 3 * 512 * 512, f4_3);
  get_param(&network, 512, b4_3);
  get_param(&network, 3 * 3 * 512 * 512, f5_1);
  get_param(&network, 512, b5_1);
  get_param(&network, 3 * 3 * 512 * 512, f5_2);
  get_param(&network, 512, b5_2);
  get_param(&network, 3 * 3 * 512 * 512, f5_3);
  get_param(&network, 512, b5_3);
  get_param(&network, 7 * 7 * 512 * 4096, w1);
  get_param(&network, 4096, b1);
  get_param(&network, 4096 * 4096, w2);
  get_param(&network, 4096, b2);
  get_param(&network, 4096 * 1000, w3);
  get_param(&network, 1000, b3);

  float *layer_softmax_host;
  int *layer_label_host;
  layer_softmax_host = (float*)malloc(sizeof(float) * MAX_BATCH * 1000);
  layer_label_host = (int*)malloc(sizeof(int) * MAX_BATCH);

  size_t i;
  for(i = 0; i < num_images; i += MAX_BATCH)
  {
    batch = num_images - i < MAX_BATCH ? num_images - i : MAX_BATCH;
    err = clEnqueueWriteBuffer(queue, data, CL_FALSE, 0, sizeof(float) * batch * 224 * 224 * 3, images + i * 224 * 224 * 3, 0, NULL, NULL);
    CHECK_ERROR(err);

    convolution_layer(data, c1_1, f1_1p, b1_1, 224, 3, 64);
    convolution_layer(c1_1, c1_2, f1_2, b1_2, 224, 64, 64);
    pooling_layer(c1_2, p1, 112, 64);

    convolution_layer(p1, c2_1, f2_1, b2_1, 112, 64, 128);
    convolution_layer(c2_1, c2_2, f2_2, b2_2, 112, 128, 128);
    pooling_layer(c2_2, p2, 56, 128);

    convolution_layer(p2, c3_1, f3_1, b3_1, 56, 128, 256);
    convolution_layer(c3_1, c3_2, f3_2, b3_2, 56, 256, 256);
    convolution_layer(c3_2, c3_3, f3_3, b3_3, 56, 256, 256);
    pooling_layer(c3_3, p3, 28, 256);

    convolution_layer(p3, c4_1, f4_1, b4_1, 28, 256, 512);
    convolution_layer(c4_1, c4_2, f4_2, b4_2, 28, 512, 512);
    convolution_layer(c4_2, c4_3, f4_3, b4_3, 28, 512, 512);
    pooling_layer(c4_3, p4, 14, 512);

    convolution_layer(p4, c5_1, f5_1, b5_1, 14, 512, 512);
    convolution_layer(c5_1, c5_2, f5_2, b5_2, 14, 512, 512);
    convolution_layer(c5_2, c5_3, f5_3, b5_3, 14, 512, 512);
    pooling_layer(c5_3, p5, 7, 512);

    fc_layer(p5, fc1, w1, b1, 7 * 7 * 512, 4096);
    fc_layer(fc1, fc2, w2, b2, 4096, 4096);
    fc_layer(fc2, fc3, w3, b3, 4096, 1000);

    softmax(fc3, layer_softmax, layer_label, 1000);


    err = clEnqueueReadBuffer(queue, layer_softmax, CL_FALSE, 0, sizeof(float) * batch * 1000, layer_softmax_host, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueReadBuffer(queue, layer_label, CL_FALSE, 0, sizeof(int) * batch, layer_label_host, 0, NULL, NULL);
    CHECK_ERROR(err);

    clFinish(queue);
    for (int j = 0; j < batch; ++j) {
      int l = layer_label_host[j];
      labels[i + j] = l;
      confidences[i + j] = layer_softmax_host[j * 1000 + l];
    }
  }

  clReleaseMemObject(c1_1);
  clReleaseMemObject(c1_2);
  clReleaseMemObject(p1);
  clReleaseMemObject(c2_1);
  clReleaseMemObject(c2_2);
  clReleaseMemObject(p2);
  clReleaseMemObject(c3_1);
  clReleaseMemObject(c3_2);
  clReleaseMemObject(c3_3);
  clReleaseMemObject(p3);
  clReleaseMemObject(c4_1);
  clReleaseMemObject(c4_2);
  clReleaseMemObject(c4_3);
  clReleaseMemObject(p4);
  clReleaseMemObject(c5_1);
  clReleaseMemObject(c5_2);
  clReleaseMemObject(c5_3);
  clReleaseMemObject(p5);
  clReleaseMemObject(fc1);
  clReleaseMemObject(fc2);
  clReleaseMemObject(fc3);
  clReleaseMemObject(layer_softmax);
  clReleaseMemObject(layer_label);
  free(layer_softmax_host);
  free(layer_label_host);

  clReleaseKernel(kernel_conv);
  clReleaseKernel(kernel_convInputPad);
  clReleaseKernel(kernel_convWeightPad);
  clReleaseKernel(kernel_convOutputUnpad);
  clReleaseKernel(kernel_pool);
  clReleaseKernel(kernel_fc);
  clReleaseKernel(kernel_softmax);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}
