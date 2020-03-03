#include "utils.hpp"
GPUTimers gtimers;
using std::cout;
using std::endl;
std::mutex GPUTimers::timers_mutex;

void GPUTimers::addDataStr(const std::string& str) {
    std::istringstream f(str);
    std::string line;
    while (std::getline(f, line)) {
        size_t i = line.find(':');
        std::string name = line.substr(0, i);
        auto t = gtimers.getTimer(name);
        t->fromString(line.substr(i + 1),epoch);
    }
}
// read "begin:end" and add it to vector
void TimerPlus::fromString(const std::string& str, unsigned epoch) {
    if(begin_vec.size()>=epoch)
        return;
    unsigned i = str.find(',');
    long long begin_ = std::stoll(str.substr(0, i));
    long long end_ = std::stoll(str.substr(i + 1));
    std::lock_guard<std::mutex> guard(timer_mutex);
    begin_vec.push_back(begin_);
    end_vec.push_back(end_);
}

// void TimerPlus::report() {
//     mili_duration max_d = mili_duration::zero();
//     mili_duration avg_d = mili_duration::zero();
//     mili_duration total_d = mili_duration::zero();
//     for (size_t i = 0; i < durations.size(); ++i) {
//         total_d += durations[i];
//         max_d = max(max_d, durations[i]);
//     }
//     avg_d = total_d / durations.size();
//     std::cout << name + "Timer : \n";
//     std::cout << "Max: " << max_d.count() << "ms \n";
//     std::cout << "Avg: " << avg_d.count() << "ms \n";
//     std::cout << "Tot: " << total_d.count() << "ms \n";
// }