#pragma once
// Transport-free candidate. Physical RF qualification must come from outside
// this adapter; fresh LowState packets cannot supply that proof.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include "go1_policy_session.hpp"
namespace go1 {
class OperatorAdapter {
 public:
  struct Result {bool valid=false,held=false,stop=false; int64_t stamp=0;};
  Result update(const std::array<uint8_t,40>& b,int64_t sample,int64_t now,
                bool radioQualified) {
    Result r;lastValid_=false;
    const uint16_t buttons=uint16_t(b[2])|(uint16_t(b[3])<<8);
    lastButtons_=buttons;
    r.stop=(buttons&0x220)==0x220; // parent L2+B; not factory L2+A
    bool axes=true;
    for(int i=0;i<5;++i){float v;std::memcpy(&v,b.data()+4+4*i,4);
      axes=axes && std::isfinite(v) && std::abs(v)<=1.5f;}
    const bool fresh=sample>0 && sample<=now && now-sample<=50000000;
    const bool ordered=now>lastNow_ && sample>lastSample_;
    lastNow_=now;lastSample_=sample;
    if(r.stop){latch("operator_stop");return r;}
    if(b[0]!=0x55 || b[1]!=0x51 || !axes || !fresh || !ordered || !radioQualified){
      latch("operator_invalid_stale_or_unqualified");return r;
    }
    r.valid=true;r.stamp=sample;lastValid_=true;
    if(!fault_.empty())return r;
    if(!(buttons&2)) {if(enabled_)latch("enable_released");else released_=true;return r;}
    enabled_=released_;r.held=enabled_;return r;
  }
  bool reset(int64_t now){
    if(!lastValid_ || (lastButtons_&2) || now<lastNow_ ||
       now<lastSample_ || now-lastSample_>50000000)return false;
    fault_.clear();released_=enabled_=false;return true;
  }
  void apply(const Result& r,PolicyEvidence& e) const {
    e.operatorNs=r.stamp;e.operatorVerified=r.valid && fault_.empty();
    e.held=r.held && fault_.empty();e.stop=r.stop;
  }
  const std::string& fault()const{return fault_;}
 private:
  void latch(const char* why){if(fault_.empty())fault_=why;released_=enabled_=false;}
  bool lastValid_=false;uint16_t lastButtons_=0;
  int64_t lastNow_=0,lastSample_=0;bool released_=false,enabled_=false;std::string fault_;
};
}
