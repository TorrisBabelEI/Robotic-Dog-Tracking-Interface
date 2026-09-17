#include "go1_operator_support.hpp"

#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

namespace {
volatile std::sig_atomic_t stopRequested = 0;
void stopSignal(int) { stopRequested = 1; }
}

int main() {
  go1::OperatorSupportServer server;
  std::string error;
  if (!server.start(go1::kOperatorSupportPort, &error)) {
    std::cerr << "Loopback support probe failed: " << error << '\n';
    return 2;
  }
  std::signal(SIGINT, stopSignal);
  std::signal(SIGTERM, stopSignal);
  std::cout << "C++ offline support probe listening on 127.0.0.1:"
            << server.port() << "; no Go1 SDK or motor commands\n";
  bool previous = false;
  while (!stopRequested) {
    const bool active = server.active(go1::operatorSupportNowNs());
    if (active != previous) {
      previous = active;
      std::cout << "support=" << (active ? "true" : "false") << '\n';
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  server.stop();
  std::cout << "Probe stopped; support=false\n";
  return 0;
}
