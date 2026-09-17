#include "../src/go1_operator_support.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool okay, const char *message) {
  if (!okay) throw std::runtime_error(message);
}
}

int main() {
  try {
    go1::OperatorSupportLease lease;
    const int64_t first = 1000000000LL;
    require(lease.receive("H 1", first) && lease.active(first),
            "first heartbeat was not accepted");
    require(lease.active(first + go1::kOperatorSupportLeaseNs),
            "heartbeat expired before the 100 ms boundary");
    require(!lease.active(first + go1::kOperatorSupportLeaseNs + 1),
            "stale heartbeat remained active");
    require(lease.receive("H 2", first + 200000000LL),
            "fresh heartbeat after a gap was not accepted");
    require(lease.receive("R 3", first + 201000000LL) &&
                !lease.active(first + 201000000LL),
            "release did not clear support immediately");
    require(lease.receive("H 4", first + 202000000LL),
            "new pulse was not accepted");
    require(!lease.receive("H 4", first + 203000000LL) &&
                !lease.active(first + 203000000LL),
            "duplicate sequence did not fail closed");
    for (const std::string &bad : {
             "H -1", "H 0", "H 9223372036854775808", "H 1 extra",
             "X 1", "H 1\r", "H ", ""}) {
      lease.reset();
      require(lease.receive("H 1", first), "fixture heartbeat failed");
      require(!lease.receive(bad, first + 1) &&
                  !lease.active(first + 1),
              "malformed frame did not fail closed");
    }
    lease.receive("H 1", first);
    lease.reset();
    require(!lease.active(first), "disconnect reset did not clear support");
    std::cout << "[PASS] C++ operator-support lease, release, replay and "
                 "malformed-frame gates\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
