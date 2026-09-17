#pragma once

// A narrowly scoped operator-attestation channel. It is not a contact sensor
// and does not authorize hardware motion by itself. Only loopback TCP is
// accepted; the laptop can reach it later through an authenticated SSH tunnel.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>

#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

namespace go1 {

constexpr int64_t kOperatorSupportLeaseNs = 100000000LL;
constexpr uint16_t kOperatorSupportPort = 18092;

inline int64_t operatorSupportNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

class OperatorSupportLease {
public:
  bool receive(const std::string &frame, int64_t nowNs) {
    if (frame.size() < 3 || frame.size() > 32 || frame[1] != ' ' ||
        (frame[0] != 'H' && frame[0] != 'R')) {
      reset();
      return false;
    }
    uint64_t sequence = 0;
    for (std::size_t i = 2; i < frame.size(); ++i) {
      const char digit = frame[i];
      if (digit < '0' || digit > '9' ||
          sequence > (static_cast<uint64_t>(
                          std::numeric_limits<int64_t>::max()) -
                      static_cast<uint64_t>(digit - '0')) / 10) {
        reset();
        return false;
      }
      sequence = sequence * 10 + static_cast<uint64_t>(digit - '0');
    }
    if (sequence == 0 || sequence <= lastSequence_) {
      reset();
      return false;
    }
    lastSequence_ = sequence;
    if (frame[0] == 'R') {
      held_ = false;
      lastHoldNs_ = 0;
    } else {
      held_ = true;
      lastHoldNs_ = nowNs;
    }
    return true;
  }

  bool active(int64_t nowNs) const {
    return held_ && lastHoldNs_ > 0 && nowNs >= lastHoldNs_ &&
           nowNs - lastHoldNs_ <= kOperatorSupportLeaseNs;
  }
  bool held() const { return held_; }
  int64_t lastHoldNs() const { return lastHoldNs_; }
  void reset() {
    lastSequence_ = 0;
    lastHoldNs_ = 0;
    held_ = false;
  }

private:
  uint64_t lastSequence_ = 0;
  int64_t lastHoldNs_ = 0;
  bool held_ = false;
};

class OperatorSupportServer {
public:
  OperatorSupportServer() = default;
  ~OperatorSupportServer() { stop(); }
  OperatorSupportServer(const OperatorSupportServer &) = delete;
  OperatorSupportServer &operator=(const OperatorSupportServer &) = delete;

  bool start(uint16_t port, std::string *error = nullptr) {
    if (running_.load()) return false;
    listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener_ < 0) return fail("socket", error);
    if (listener_ >= FD_SETSIZE) {
      errno = EMFILE;
      return fail("select descriptor range", error);
    }
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    if (::bind(listener_, reinterpret_cast<sockaddr *>(&address),
               sizeof(address)) < 0) return fail("bind loopback TCP", error);
    if (::listen(listener_, 1) < 0) return fail("listen", error);
    sockaddr_in bound = {};
    socklen_t length = sizeof(bound);
    if (::getsockname(listener_, reinterpret_cast<sockaddr *>(&bound),
                      &length) < 0) return fail("getsockname", error);
    port_ = ntohs(bound.sin_port);
    running_.store(true);
    worker_ = std::thread(&OperatorSupportServer::run, this);
    return true;
  }

  void stop() {
    running_.store(false);
    if (worker_.joinable()) worker_.join();
    if (listener_ >= 0) {
      ::close(listener_);
      listener_ = -1;
    }
    held_.store(false);
    lastHoldNs_.store(0);
  }

  uint16_t port() const { return port_; }
  bool active(int64_t nowNs) const {
    if (!held_.load(std::memory_order_acquire)) return false;
    const int64_t last = lastHoldNs_.load(std::memory_order_relaxed);
    return last > 0 && nowNs >= last &&
           nowNs - last <= kOperatorSupportLeaseNs;
  }

private:
  bool fail(const char *operation, std::string *error) {
    if (error) *error = std::string(operation) + ": " + std::to_string(errno);
    if (listener_ >= 0) {
      ::close(listener_);
      listener_ = -1;
    }
    return false;
  }

  void publish(const OperatorSupportLease &lease) {
    if (lease.held()) {
      lastHoldNs_.store(lease.lastHoldNs(), std::memory_order_relaxed);
      held_.store(true, std::memory_order_release);
    } else {
      held_.store(false, std::memory_order_release);
      lastHoldNs_.store(0, std::memory_order_relaxed);
    }
  }

  void run() {
    int client = -1;
    OperatorSupportLease lease;
    std::string pending;
    auto disconnect = [&]() {
      if (client >= 0) ::close(client);
      client = -1;
      pending.clear();
      lease.reset();
      publish(lease);
    };
    while (running_.load()) {
      fd_set readSet;
      FD_ZERO(&readSet);
      FD_SET(listener_, &readSet);
      if (client >= 0) FD_SET(client, &readSet);
      timeval timeout = {0, 20000};
      const int ready = ::select(std::max(listener_, client) + 1, &readSet,
                                 nullptr, nullptr, &timeout);
      if (ready < 0) {
        if (errno != EINTR) {
          disconnect();
          running_.store(false);
        }
        continue;
      }
      if (FD_ISSET(listener_, &readSet)) {
        const int candidate = ::accept(listener_, nullptr, nullptr);
        if (candidate >= 0) {
          if (client >= 0 || candidate >= FD_SETSIZE) ::close(candidate);
          else {
            client = candidate;
            lease.reset();
            publish(lease);
          }
        }
      }
      if (client < 0 || !FD_ISSET(client, &readSet)) continue;
      char bytes[512];
      const ssize_t size = ::recv(client, bytes, sizeof(bytes), 0);
      if (size <= 0) {
        disconnect();
        continue;
      }
      bool invalid = false;
      for (ssize_t i = 0; i < size; ++i) {
        if (bytes[i] == '\n') {
          const int64_t now = operatorSupportNowNs();
          if (!lease.receive(pending, now)) invalid = true;
          publish(lease);
          pending.clear();
          if (invalid) break;
        } else if (pending.size() >= 32) {
          invalid = true;
          break;
        } else {
          pending.push_back(bytes[i]);
        }
      }
      if (invalid) disconnect();
    }
    disconnect();
  }

  int listener_ = -1;
  uint16_t port_ = 0;
  std::atomic<bool> running_{false}, held_{false};
  std::atomic<int64_t> lastHoldNs_{0};
  std::thread worker_;
};

} // namespace go1
