#pragma once
// Transport-free acquisition-to-control bridge. No motor or network interface.
// Timestamps must come from one monotonic clock; do not restamp old radio data.
#include "go1_policy_operator.hpp"
namespace go1 {
class ControlEvidence {
 public:
  // Invoke only on an acquired radio frame. Default qualification fails closed.
  void acquireRadio(const std::array<uint8_t,40>& bytes, int64_t sampleNs,
                    int64_t receiveNs, bool radioQualified=false) {
    radio_=adapter_.update(bytes,sampleNs,receiveNs,radioQualified);
  }
  // Invoke every control tick, including ticks with no new radio frame.
  PolicyEvidence compose(int64_t nowNs,int64_t stateNs,uint64_t sequence,
                         int64_t estimateNs,bool estimateValid,bool owner) const {
    PolicyEvidence e;e.nowNs=nowNs;e.stateNs=stateNs;e.stateSequence=sequence;
    e.estimateNs=estimateNs;e.estimateValid=estimateValid;e.owner=owner;
    adapter_.apply(radio_,e);return e;
  }
  // This resets the input adapter only, not the parent core or worker session.
  // Parent recovery still needs a separately reviewed explicit lifecycle.
  bool resetOperator(int64_t nowNs) {
    if(!adapter_.reset(nowNs))return false;
    radio_={};return true;
  }
  const std::string& operatorFault() const{return adapter_.fault();}
 private:
  OperatorAdapter adapter_;
  OperatorAdapter::Result radio_{};
};
}
