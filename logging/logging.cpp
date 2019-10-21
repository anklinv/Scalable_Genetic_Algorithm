#include <sstream>
#include <iomanip>

#include "logging.hpp"

#define LOGGING_DIRECTORY "logs/"

Logger::Logger(std::string filename_prefix) {
    this->filename_prefix = filename_prefix;
}

Logger::Logger(int rank, time_t *time) {
    std::stringstream ss;
    ss << LOGGING_DIRECTORY;
    auto t = std::time(time);
    auto tm = *std::localtime(&t);
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << "_";
    ss << std::setw(4) << std::setfill('0') << rank;
    this->filename_prefix = ss.str();
}

void Logger::open() {
    this->fitness_file.open(this->filename_prefix + "_fitness.csv");
    // this->fitness_file << "epoch, fitness" << std::endl;
}

void Logger::close() {
    this->fitness_file.close();
}

void Logger::log_best_fitness_per_epoch(int epoch, double fitness) {
    this->fitness_file << epoch << ", " << fitness << std::endl;
}
