#include "util.h"
#include <time.h>
#include "math.h"

uint64_t getTimeMicroseconds64() {
  uint64_t nTime;
  struct timespec tSpec;

  clock_gettime(CLOCK_REALTIME, &tSpec);

  nTime = (uint64_t)tSpec.tv_sec * 1000000 + (uint64_t)tSpec.tv_nsec / 1000;
  return nTime;
}

float* transpose(float* weight, int h, int w) {
  float* new_weight = (float*)malloc(w * h * 4);
  int i, j;
  for (i = 0; i < w; ++i) {
    for (j = 0; j < h; ++j) {
      new_weight[j * w + i] = weight[i * h + j];
    }
  }

  free(weight);
  return new_weight;
}

float* get_parameter(const char* filename, int size) {
  float* parameter = (float*)malloc(size * 4);
  if (!parameter) {
    printf("Bad Malloc\n");
    exit(0);
  }
  FILE* ptr = fopen(filename, "rb");

  if (!ptr) {
    printf("Bad file path: %p, %s\n", ptr, strerror(errno));
    exit(0);
  }
  fread(parameter, size * 4, 1, ptr);

  fclose(ptr);
  return parameter;
}

float output_checker(float* A, float* B, int len, int channel, int shift) {\
  int curr_err, prev_err;
  curr_err=0;
  prev_err=0;
  int consecutive_count = 0;
  int max_consecutive_count = 0;
  int end_max_index = 0;
  int error_cnt = 0, i, j, k;
  float max_error = 0;
  for (i = 0; i < len; i++) {
    for (j = 0; j < len; j++) {
      
      for (k = 0; k < channel; k++) {
        prev_err = curr_err;
        curr_err = 0;
        float diff = fabs(
            A[((i + shift) * (len + 2 * shift) + j + shift) * channel + k] -
            B[(i * len + j) * channel + k]);
		        //printf ("WINOGRAD: %f, CUDNN: %f \n",A[((i + shift) * (len + 2 * shift) + j + shift) * channel + k], B[(i * len + j) * channel + k]);
            
        if (diff > 1e-5) {
	  //printf("%f\n",A[((i + shift) * (len + 2 * shift) + j + shift) * channel + k]);
          error_cnt++;
          curr_err = 1;
          if (prev_err == curr_err) {
            consecutive_count =  consecutive_count +1;
            if (consecutive_count > max_consecutive_count) {
              max_consecutive_count = consecutive_count;
              end_max_index = (i * len + j) * channel + k;
              //printf("%d\n", end_max_index);
            }
          }
          
        }
        //lol, this was in the for
        consecutive_count = consecutive_count * curr_err;
        //printf ("%d\n" ,consecutive_count);
        if (diff > max_error)
          max_error = diff;
      }
    }
  }
  printf("[max_error: %f][error_cnt: %d][max_consecutive_count: %d AND %d]\n", max_error, error_cnt, max_consecutive_count, end_max_index);
}
