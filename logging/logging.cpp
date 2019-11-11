#include <ctime>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "logging.hpp"
#include "tags.hpp"

Logger::Logger(std::string filename_prefix) {
    this->filename_prefix = filename_prefix;
}

Logger::Logger(std::string dir, int rank, time_t *time) {
    std::stringstream ss;
    ss << dir;
    auto t = std::time(time);
    auto tm = *std::localtime(&t);
    ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << "_";
    ss << std::setw(4) << std::setfill('0') << rank;
    this->filename_prefix = ss.str();
}

void Logger::open() {
    this->fitness_file.open(this->filename_prefix + "_fitness.csv");
    // this->fitness_file << "epoch, fitness" << std::endl;
    this->tag_file.open(this->filename_prefix + "_tags.bin", std::ios::out | std::ios::binary);
    log_tag_value(LOGGING_TAG_VERSION, LOGGING_VERSION);
    log_tag_value(LOGGING_TAG_CLOCKS_PER_SEC, CLOCKS_PER_SEC);

    this->clock = std::chrono::high_resolution_clock();
    this->start = this->clock.now();
    LOG_WC(LOGGING_BEGIN);
    LOG_CC(LOGGING_BEGIN);
}

void Logger::close() {
    this->fitness_file.close();
    LOG_WC(LOGGING_END);
    LOG_CC(LOGGING_END);
    this->tag_file.close();
}

void Logger::log_all_fitness_per_epoch(int epoch, const std::vector<double>& fitness) {
    this->fitness_file << epoch;
    for (auto f : fitness) {
        this->fitness_file << "," << f;
    }
    this->fitness_file << std::endl;
}

void Logger::log_best_fitness_per_epoch(int epoch, double fitness) {
    this->fitness_file << epoch;
    this->fitness_file << "," << fitness;
    this->fitness_file << std::endl;
}

void Logger::log_tag_value(tag_t tag, tag_value_t value) {
    this->tag_file.write((const char *)(&tag), sizeof(tag));
    this->tag_file.write((const char *)(&value), sizeof(value));
}

void Logger::log_wall_clock(tag_t tag) {
    log_tag_value(
        tag,
        std::chrono::duration_cast<std::chrono::microseconds>(
            this->clock.now() - this->start
        ).count()
    );
}

void Logger::log_cpu_time(tag_t tag) {
    log_tag_value(tag, std::clock());
}
