#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "network.h"
#include "class_name.h"

void vggnet(float * images, float * network, int * labels, float * confidences, int num_images);

int timespec_subtract(struct timespec*, struct timespec*, struct timespec*);

int main(int argc, char** argv) {
  float *images, *network, *confidences;
  int *labels;
  int num_images, i;
  FILE *io_file;
  int sizes[32];
  char image_files[1024][1000];
  struct timespec start, end, spent;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <image list>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  io_file = fopen(argv[1], "r");
  fscanf(io_file, "%d\n", &num_images);

  for(i = 0; i < num_images; i++)
  {
    fscanf(io_file, "%s", image_files[i]); 
  }
  fclose(io_file);

  int vggnet_size = 0;
  for(i = 0; i < 32; i++)
  {
    char filename[100];
    memset(filename, 0, 100);
    strcat(filename, "network/");
    strcat(filename, file_list[i]);
    io_file = fopen(filename, "rb");
    fseek(io_file, 0, SEEK_END); 
    sizes[i] = ftell(io_file);
    vggnet_size += sizes[i];
    fclose(io_file);
  }

  images = (float *)malloc(sizeof(float) * 224 * 224 * 3 * num_images);
  network = (float *)malloc(sizeof(float) * vggnet_size); 
  labels = (int *)malloc(sizeof(int) * num_images);
  confidences = (float *)malloc(sizeof(float) * num_images);

  int vggnet_idx = 0;
  for(i = 0; i < 32; i++)
  {
    char filename[100];
    memset(filename, 0, 100);
    strcat(filename, "network/");
    strcat(filename, file_list[i]);
    io_file = fopen(filename, "rb");
    fread(network + vggnet_idx, 1, sizes[i], io_file);
    fclose(io_file);
    vggnet_idx += sizes[i]/sizeof(float);
  }

  for(i = 0; i < num_images; i++)
  {
    io_file = fopen(image_files[i], "rb");
    if(!io_file)
    {
      printf("%s does not exist!\n", image_files[i]);
    }
    fread(images + (224 * 224 * 3) * i, 4,  224 * 224 * 3, io_file);
    fclose(io_file);
  }

  clock_gettime(CLOCK_MONOTONIC, &start);
  vggnet(images, network, labels, confidences, num_images);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);

  for(i = 0; i < num_images; i++)
  {
    printf("%s :%s : %.3f\n", image_files[i], class_name[labels[i]], confidences[i]);
  }

  printf("Elapsed time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
  return 0;
}

int timespec_subtract(struct timespec* result, struct timespec *x, struct timespec *y) {
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
