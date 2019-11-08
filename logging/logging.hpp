#ifndef DPHPC_PROJECT_LOGGING_HPP
#define DPHPC_PROJECT_LOGGING_HPP

#include <cstdint>
#include <fstream>
#include <vector>

#include "tags.hpp"

typedef uint32_t tag_t;
typedef uint32_t tag_value_t;

class Logger {
public:
    /// Generate a Logger object to collect statistics and debug data
    /// \param filename_prefix base name for all files
    Logger(std::string filename_prefix);

    /// Generate a Logger object to collect statistics and debug data
    /// \param rank rank of the current process
    /// \param time used in specifying file names
    Logger(int rank, time_t *time = nullptr);

    /// Open logging files
    void open();

    /// Close logging files
    void close();

    /// Store best fitness value after given number of epochs
    /// \param epoch number of finished epochs (0 means initialised)
    /// \param fitness vector of fitness values
    void log_best_fitness_per_epoch(
        int epoch,
        std::vector<double> fitness = std::vector<double>()
    );

    /// Log a small value quickly
    /// \param tag defined in tags.hpp
    /// \param value any 32-bit piece of data
    void log_tag_value(tag_t tag, tag_value_t value);

    /// Log wall clock time
    /// \param tag defined in tags.hpp
    void log_wall_clock(tag_t tag);

    /// Log CPU time
    /// \param tag defined in tags.hpp
    void log_cpu_time(tag_t tag);

protected:
    std::string filename_prefix;
    std::ofstream fitness_file;
    std::ofstream tag_file;

    std::chrono::high_resolution_clock clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

#endif /* DPHPC_PROJECT_LOGGING_HPP */
