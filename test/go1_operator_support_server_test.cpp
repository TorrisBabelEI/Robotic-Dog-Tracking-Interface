#include "../src/go1_operator_support.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool okay, const char *message) {
  if (!okay) throw std::runtime_error(message);
}
bool waitFor(go1::OperatorSupportServer &server, bool wanted) {
  for (int i = 0; i < 100; ++i) {
    if (server.active(go1::operatorSupportNowNs()) == wanted) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}
void sendFrame(int socketFd, const char *frame) {
  const std::string text(frame);
  require(::send(socketFd, text.data(), text.size(), 0) ==
              static_cast<ssize_t>(text.size()),
          "test sender could not send a complete frame");
}
}

int main() {
  go1::OperatorSupportServer server;
  std::string error;
  try {
    require(server.start(0, &error), "loopback server did not start");
    require(server.port() != 0, "server did not publish assigned port");
    int client = ::socket(AF_INET, SOCK_STREAM, 0);
    require(client >= 0, "test client socket failed");
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(server.port());
    require(::connect(client, reinterpret_cast<sockaddr *>(&address),
                      sizeof(address)) == 0, "test client connect failed");
    sendFrame(client, "H 1\n");
    require(waitFor(server, true), "hold heartbeat was not received");
    std::this_thread::sleep_for(std::chrono::milliseconds(125));
    require(!server.active(go1::operatorSupportNowNs()),
            "silence did not expire the 100 ms lease");
    sendFrame(client, "H 2\n");
    require(waitFor(server, true), "fresh heartbeat was not received");
    sendFrame(client, "R 3\n");
    require(waitFor(server, false), "release was not observed");
    sendFrame(client, "H 4\n");
    require(waitFor(server, true), "second pulse was not received");
    ::close(client);
    require(waitFor(server, false), "disconnect did not clear support");
    server.stop();
    std::cout << "[PASS] loopback hold, lease expiry, release and disconnect\n";
    return 0;
  } catch (const std::exception &failure) {
    server.stop();
    std::cerr << "[FAIL] " << failure.what();
    if (!error.empty()) std::cerr << ": " << error;
    std::cerr << '\n';
    return 1;
  }
}
