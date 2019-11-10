#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Kernel128_one.h"
#include "Kernel128_winograd.h"
#include "Kernel128_winograd_variable_feat_map.h"
#include "official.h"
//#include "Kernel128_winograd_varfeatmap_varchannel_fixedTHfixedCH.h"
#include "Kernel256_one.h"
#include "Kernel_var_channelsize.h"
#include "Kernel256_winograd.h"
#include "cioflan.h"
#include "util.h"

int main(int argc, char** argv) {
  cudaSetDevice(0);

  int mode = 0;

  for (i = 0; i < nTest; i++) {
    //printf("---- Iter: %d ----\n", i);
    int res = -1;
    switch (mode) {
      case 0:
        res = ();
        break;
      case 1:
        res = kernel_256();
        break;
      case 2:
        res = kernel_128_1_in();
        break;
      case 3:
        res = kernel_128_1_out();
        break;
      case 4:
        res = kernel_256_1_in();
        break;
      case 5:
        res = kernel_256_1_out();
        break;
      case 6:
        res = kernel_128_varfeat();
        break;
      case 7:
	res = kernel_var_channelsize();
        break;
      case 8:
        res = kernel_128_cioflan();
        break;
      case 9: 
        //res = kernel_128_varfeat_varchan(channels,featsize);
        break;
      case 10:
        res = winograd_convolution();
	break;
    }
  }
  printf(
      "Average Total Time: [%d us], \n",
      sum / (nTest - 2),

  return 0;
}
