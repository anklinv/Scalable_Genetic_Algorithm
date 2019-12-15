#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "tags.h"

typedef uint32_t tag_t;
typedef uint32_t tag_value_t;

#define LOG(EVENT, VALUE) log_tag_value(LOGGING_TAG_VAL_##EVENT, VALUE)
#define LOG_WC(EVENT) log_wall_clock(LOGGING_TAG_WC_##EVENT)
#define LOG_CC(EVENT) log_cpu_time(LOGGING_TAG_CC_##EVENT)

char *filename_prefix;
FILE *fitness_file;
FILE *tag_file;

// see http://www.cs.tufts.edu/comp/111/examples/Time/clock_gettime.c
// struct timespec {
//    time_t   tv_sec;        /* seconds */
//    long     tv_nsec;       /* nanoseconds */
// };
struct timespec start, curr_time, curr_cpu_time;

/*/// Generate a Logger object to collect statistics and debug data
/// \param f base name for all files
void setup_logger(char *f);*/

/// Generate a Logger object to collect statistics and debug data
/// \param rank rank of the current process
/// \param time used in specifying file names
void setup_logger(char *dir, int rank);

/// Open logging files
void open_logger();

/// Close logging files
void close_logger();

/// Log a small value quickly
/// \param tag defined in tags.hpp
/// \param value any 32-bit piece of data
void log_tag_value(tag_t t, tag_value_t v);

/// Log wall clock time
/// \param tag defined in tags.hpp
void log_wall_clock(tag_t t);

/// Log CPU time
/// \param tag defined in tags.hpp
void log_cpu_time(tag_t t);
