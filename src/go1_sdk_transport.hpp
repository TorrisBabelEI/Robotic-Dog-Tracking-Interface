#pragma once
#include "go1_sdk_receive.hpp"
namespace go1 {
// Return the SDK's raw byte count. The caller normalizes exact length only.
// Failed staging must never retransmit a retained, potentially stale command.
template<class Udp, class Command>
int stageAndSend(Udp& udp, Command& command) {
  return udp.SetSend(command) < 0 ? -1 : udp.Send();
}
}
