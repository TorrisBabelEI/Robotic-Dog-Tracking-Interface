#pragma once
// SDK-independent policy lifecycle used by the parent core and offline tests.
// A policy worker proposes actions only; it never owns a motor transport.
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <string>
namespace go1 {
struct PolicyEvidence {
  int64_t nowNs=0,stateNs=0,operatorNs=0,estimateNs=0;
  uint64_t stateSequence=0;
  bool owner=false,operatorVerified=false,held=false,stop=false,estimateValid=false;
  bool inferenceSampleReady=true; // Request only on a new acquisition; never retimestamp cached data.
};
struct PolicyRequest {
  uint64_t generation=0,id=0,stateSequence=0;
  int64_t sampleNs=0,requestNs=0;
  std::array<double,12> previousAction{{}};
};
class PolicySession {
public:
  static constexpr int64_t kPeriodNs=20000000LL;
  static constexpr int64_t kStateTimeoutNs=50000000LL;
  static constexpr int64_t kActionTimeoutNs=60000000LL;
  static constexpr int64_t kOperatorTimeoutNs=50000000LL;
  const std::string& fault() const {return fault_;}
  bool running() const {return running_;}
  bool pending() const {return pending_;}
  bool appliedValid() const {return appliedValid_;}
  const std::array<double,12>& previousAction() const {return applied_;}
  const PolicyRequest& request() const {return request_;}
  void stop(const std::string& reason) {if(fault_.empty())fault_=reason;running_=false;pending_=false;}
  bool reset(bool released) {
    if(!released)return false;
    ++generation_;fault_.clear();released_=false;running_=pending_=haveAction_=appliedValid_=false;
    lastTickNs_=startNs_=nextRequestNs_=lastActionNs_=0;requestCounter_=0;return true;
  }
  // Call after the final limit/watchdog stage and successful transport send.
  // A send acknowledgement is not evidence of measured actuator tracking.
  void acknowledge(const std::array<double,12>& target,bool positionCommand,bool sent) {
    if(!sent){appliedValid_=false;stop("policy_send_failure");return;}
    if(!positionCommand){appliedValid_=false;return;}
    for(unsigned i=0;i<12;++i) {
      if(!std::isfinite(target[i])){appliedValid_=false;stop("nonfinite_sent_target");return;}
    }
    lastSent_=target;appliedValid_=true;
    for(unsigned i=0;i<12;++i)applied_[i]=(target[i]-nominal(i))/.25;
  }
  bool tick(const PolicyEvidence& e) {
    if(!fault_.empty())return false;
    if(lastTickNs_ && (e.nowNs<=lastTickNs_ || e.nowNs-lastTickNs_>20000000LL)) {
      stop("policy_control_deadline");return false;
    }
    lastTickNs_=e.nowNs;
    auto fresh=[&](int64_t t,int64_t limit){return t>0 && e.nowNs>=t && e.nowNs-t<=limit;};
    if(!e.owner){stop("policy_owner_lost");return false;}
    if(e.stop){stop("policy_operator_stop");return false;}
    if(!e.operatorVerified || !fresh(e.operatorNs,kOperatorTimeoutNs)) {
      stop("policy_operator_input_stale_or_unverified");return false;
    }
    if(!e.held) {
      if(running_){stop("policy_enable_released");return false;}
      released_=true;return false;
    }
    if(!released_)return false; // A held button at startup never arms.
    if(!fresh(e.stateNs,kStateTimeoutNs) || !e.estimateValid || !fresh(e.estimateNs,kStateTimeoutNs)) {
      stop("policy_state_or_estimate_stale");return false;
    }
    if(!appliedValid_){stop("policy_applied_command_unknown");return false;}
    if(!running_) {
      running_=true;startNs_=nextRequestNs_=lastActionNs_=e.nowNs;seed_=lastSent_;
      for(double a:applied_)if(std::abs(a)>1.00001){stop("policy_start_pose_outside_action_range");return false;}
    }
    if(e.nowNs-lastActionNs_>kActionTimeoutNs){stop("policy_action_timeout");return false;}
    if(pending_ && e.nowNs-request_.requestNs>kStateTimeoutNs){stop("policy_inference_timeout");return false;}
    if(!pending_ && e.nowNs>=nextRequestNs_ && e.inferenceSampleReady) {
      request_={generation_,++requestCounter_,e.stateSequence,e.stateNs,e.nowNs,applied_};
      pending_=true;
      nextRequestNs_+=((e.nowNs-nextRequestNs_)/kPeriodNs+1)*kPeriodNs;
      return true;
    }
    return false;
  }
  bool submit(uint64_t generation,uint64_t id,int64_t sampleNs,
              int64_t nowNs,const std::array<double,12>& action) {
    if(!running_ || !pending_ || !fault_.empty())return false;
    if(generation!=request_.generation || id!=request_.id || sampleNs!=request_.sampleNs ||
       nowNs<request_.requestNs || nowNs-sampleNs>kStateTimeoutNs) {
      stop("policy_reply_identity_or_age");return false;
    }
    for(double a:action)if(!std::isfinite(a) || std::abs(a)>1.00001){stop("policy_invalid_action");return false;}
    action_=action;haveAction_=true;pending_=false;lastActionNs_=nowNs;return true;
  }
  std::array<double,12> target(int64_t nowNs) const {
    auto q=lastSent_;
    if(!running_ || !haveAction_ || !fault_.empty())return q;
    const double blend=std::min(1.,std::max(0.,(nowNs-startNs_)/2e9));
    for(unsigned i=0;i<12;++i) {
      const double desired=(1-blend)*seed_[i]+blend*(nominal(i)+.25*action_[i]);
      // Conservative parent-compatible 1 rad/s target slew, 500 Hz command loop.
      q[i]+=std::max(-.002,std::min(.002,desired-q[i]));
    }
    return q;
  }
private:
  static double nominal(unsigned i){return i%3==0?0.:i%3==1?.9:-1.8;}
  std::string fault_;
  bool released_=false,running_=false,pending_=false,haveAction_=false,appliedValid_=false;
  uint64_t generation_=1,requestCounter_=0;
  int64_t lastTickNs_=0,startNs_=0,nextRequestNs_=0,lastActionNs_=0;
  PolicyRequest request_{};
  std::array<double,12> action_{{}},applied_{{}},lastSent_{{}},seed_{{}};
};
} // namespace go1
