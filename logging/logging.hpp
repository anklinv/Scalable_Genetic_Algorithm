#ifndef DPHPC_PROJECT_LOGGING_HPP
#define DPHPC_PROJECT_LOGGING_HPP

#include <fstream>
#include <vector>

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

protected:
    std::string filename_prefix;
    std::ofstream fitness_file;
};

#endif /* DPHPC_PROJECT_LOGGING_HPP */
