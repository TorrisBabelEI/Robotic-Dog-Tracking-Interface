#pragma once
// Controller composition only: no SDK transport or motor sender. Hardware entry
// remains locked separately. Qualification is evidence supplied by caller, not
// inferred from process readiness or successful mock tests.
#include "go1_policy_process.hpp"
#include "go1_policy_dispatch.hpp"
#include "go1_policy_evidence.hpp"
#include "go1_sdk_policy_input.hpp"
#include <memory>
namespace go1 {
struct PolicyLaunchSpec {
  std::string python,script,config,bundle,log,releaseManifest;
};
struct PolicyQualification {
  bool release=false,sensors=false,radio=false,exclusiveOwner=false;
  bool complete()const{return release&&sensors&&radio&&exclusiveOwner;}
};
class PolicyRuntime {
public:
  PolicyRuntime()=default;
  PolicyRuntime(const PolicyRuntime&)=delete;
  PolicyRuntime& operator=(const PolicyRuntime&)=delete;
  ~PolicyRuntime(){stop();}
  void start(const PolicyLaunchSpec& spec,const PolicyQualification& qualification) {
    if(started_)throw std::runtime_error("policy lifecycle cannot automatically restart");
    if(!qualification.complete())throw std::runtime_error("policy release/sensor/operator/ownership evidence missing");
    started_=true;
    process_.start(spec.python,spec.script,spec.config,spec.bundle,spec.log,"native",20000,spec.releaseManifest);
    worker_.reset(new PolicyWorker());
    if(!worker_->attach(process_.fd())){stop();throw std::runtime_error("policy worker attach failed");}
  }
  // For transport-free synthetic tests only; hardware runner never calls this.
  bool attachReviewEndpoint(int fd,int64_t estimatorPeriodNs=8000000) {
    if(started_)return false;
    if(!dispatch_.configureEstimatorPeriodNs(estimatorPeriodNs))return false;
    started_=true;worker_.reset(new PolicyWorker());
    if(!worker_->attach(fd)){worker_.reset();return false;}return true;
  }
  uint64_t reviewReplies()const{return worker_?worker_->replies():0;}
  uint64_t reviewActions()const{return worker_?worker_->actions():0;}
  uint64_t reviewMaxAgeNs()const{return worker_?worker_->maxAgeNs():0;}
  bool stop() {
    worker_.reset(); // Close duplicate IPC before waiting for child exit.
    return process_.stop();
  }
  template<class Core> void before(Core& core,const UNITREE_LEGGED_SDK::LowState& state,
      bool hasState,bool fresh,int64_t stateNs,uint64_t sequence,int64_t now,
      const PolicyQualification& qualification) {
    fresh_=false;
    if(!qualification.complete()){core.forceHardFault("policy_qualification_lost",now);return;}
    if(!worker_){core.forceHardFault("policy_worker_not_started",now);return;}
    if(!hasState || stateNs<=0 || stateNs>now || now-stateNs>50000000LL){
      core.forceHardFault("policy_acquisition_invalid_or_stale",now);return;}
    if(fresh){
      if(stateNs<=lastNs_ || sequence<=lastSequence_){core.forceHardFault("policy_acquisition_order",now);return;}
      try{sensor_=sdkPolicySensors(state,true);}
      catch(const std::exception&){core.forceHardFault("policy_sdk_input_invalid",now);return;}
      input_.acquireRadio(state.wirelessRemote,stateNs,now,qualification.radio);
      lastNs_=stateNs;lastSequence_=sequence;fresh_=true;
    }else if(stateNs!=lastNs_ || sequence!=lastSequence_){
      core.forceHardFault("policy_acquisition_identity_mismatch",now);return;
    }
    worker_->poll(now,core.policySession());
    auto e=input_.compose(now,stateNs,sequence,worker_->estimateNs(),worker_->estimateValid(),qualification.exclusiveOwner);
    e.inferenceSampleReady=fresh_;core.setPolicyEvidence(e);
    if(!worker_->fault().empty())core.forceHardFault(worker_->fault(),now);
  }
  template<class Core> bool after(Core& core,int64_t now) {
    if(core.failed())return false;
    if(!worker_){core.forceHardFault("policy_worker_not_started",now);return false;}
    dispatch_.dispatch(*worker_,core.policySession(),now,lastNs_,sensor_.data(),fresh_);
    const auto& fault=dispatch_.fault().empty()?worker_->fault():dispatch_.fault();
    if(!fault.empty()){core.forceHardFault(fault,now);return false;}
    return true;
  }
private:
  PolicyProcess process_;std::unique_ptr<PolicyWorker> worker_;
  ControlEvidence input_;PolicyDispatch dispatch_;std::array<double,58> sensor_{};
  bool started_=false,fresh_=false;int64_t lastNs_=0;uint64_t lastSequence_=0;
};
}
