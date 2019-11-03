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

void Logger::open_timing() {
    this->timing_file.open(this->filename_prefix + "_timing.csv");
}

void Logger::close() {
    this->fitness_file.close();
}

void Logger::close_timing() {
    this->timing_file.close();
}

void Logger::log_best_fitness_per_epoch(int epoch, std::vector<double> fitness) {
    this->fitness_file << epoch;
    for (auto f : fitness) this->fitness_file << "," << f;
    this->fitness_file << std::endl;
}

void Logger::log_timing_per_epoch(int epoch, double rank_dur, double breed, double mutate) {
    this->timing_file << epoch << "," << rank_dur << "," << breed << "," << mutate << std::endl;
}
