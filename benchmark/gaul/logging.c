#include <string.h>
#include "logging.h"

/*/// Generate a Logger object to collect statistics and debug data
/// \param f base name for all files
void setup_logger(char *f) {
    filename_prefix = (char*)malloc((strlen(f)+1)*sizeof(filename_prefix));
    strcpy(filename_prefix, f)
}*/

/// Generate a Logger object to collect statistics and debug data
/// \param rank rank of the current process
/// \param time used in specifying file names
void setup_logger(char *dir, int rank) {
    char time_buff[21];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    snprintf(time_buff, sizeof(time_buff), "%d-%d-%d_%d-%d-%d_",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec);
    char rank_buff[5];
    sprintf(rank_buff, "%04d", rank);
    filename_prefix =
        (char*)malloc((strlen(dir)+strlen(time_buff)+strlen(rank_buff)+1)
        *sizeof(filename_prefix));
    strcpy(filename_prefix, dir); // cpy
    strcat(filename_prefix, time_buff);
    strcat(filename_prefix, rank_buff);
}

void open_logger() {
    char *postfix = "_tags.bin";
    char filename[strlen(filename_prefix)+strlen(postfix)+1];
    strcpy(filename, filename_prefix);
    strcat(filename, postfix);
    tag_file = fopen(filename, "ab");
    log_tag_value(LOGGING_TAG_VERSION, LOGGING_VERSION);
    log_tag_value(LOGGING_TAG_CLOCKS_PER_SEC, CLOCKS_PER_SEC);
    clock_gettime(CLOCK_REALTIME, &start);
    LOG_WC(LOGGING_BEGIN);
    LOG_CC(LOGGING_BEGIN);
}

void close_logger() {
    LOG_WC(LOGGING_END);
    LOG_CC(LOGGING_END);
    if(tag_file != NULL) {
        fclose(tag_file);
    }
    free(filename_prefix);
}

void log_tag_value(tag_t t, tag_value_t v) {
    fwrite((const char *)(&t), sizeof(t), 1, tag_file);
    fwrite((const char *)(&v), sizeof(v), 1, tag_file);
}

void log_wall_clock(tag_t t) {
    clock_gettime(CLOCK_REALTIME, &curr_time);
    long delta = (curr_time.tv_sec - start.tv_sec)*1e6 +
        ((curr_time.tv_nsec - start.tv_nsec) / ((long)1e3));
    //printf("WCT is %d \n", delta);
    log_tag_value(
        t,
        delta
    ); // convert from ns to microseconds
}

void log_cpu_time(tag_t t) {
    log_tag_value(t, clock());
}
