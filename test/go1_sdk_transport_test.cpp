#include "go1_sdk_transport.hpp"
#include <stdexcept>
#include <iostream>
#include <limits>
void require(bool ok) { if(!ok) throw std::runtime_error("SDK transport regression"); }
struct MockUdp {
  int staged=0, result=614, sends=0;
  int SetSend(const int&) { return staged; }
  int Send() { ++sends; return result; }
};
int main() {
  using go1::SdkReceiveCounters;
  const SdkReceiveCounters before{10,2,3};
  require(go1::validatedSdkReceive(820,before,{11,2,3}));
  require(go1::validatedSdkReceive(0,before,{11,2,3}));
  require(!go1::validatedSdkReceive(-1,before,{11,2,3}));
  require(!go1::validatedSdkReceive(820,before,before));
  require(!go1::validatedSdkReceive(820,before,{11,3,3}));
  require(!go1::validatedSdkReceive(820,before,{11,2,4}));
  require(!go1::validatedSdkReceive(820,before,{0,2,3}));
  require(!go1::validatedSdkReceive(820,{std::numeric_limits<uint64_t>::max(),0,0},{0,0,0}));
  struct Counters { unsigned RecvCount=1,FlagError=2,RecvCRCError=3,SendCount=99; } c;
  const auto snapshot=SdkReceiveCounters::snapshot(c);
  require(snapshot.received==1 && snapshot.flagErrors==2 && snapshot.crcErrors==3);
  MockUdp udp; int command=0;
  udp.staged=-1; require(go1::stageAndSend(udp,command)==-1 && udp.sends==0);
  udp.staged=0; require(go1::stageAndSend(udp,command)==614 && udp.sends==1);
  udp.result=613; require(go1::stageAndSend(udp,command)==613 && udp.sends==2);
  udp.result=-1; require(go1::stageAndSend(udp,command)==-1 && udp.sends==3);
  std::cout << "PASS receive counter/CRC/error/reset checks and guarded send; no SDK transport constructed\n";
}
