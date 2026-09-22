#pragma once
// Nonblocking acquisition-to-worker dispatch. Cache the exact inference sample
// under backpressure; never relabel newer sensors with an old request identity.
#include "go1_policy_worker.hpp"
namespace go1 {
class PolicyDispatch {
 public:
  // Configure before first dispatch only. Existing hardware default stays125Hz.
  // A faster profile needs independent target throughput qualification.
  bool configureEstimatorPeriodNs(int64_t period) {
    if(lastQueuedNs_ || cached_ || !fault_.empty() ||
       (period!=2000000 && period!=4000000 && period!=8000000))return false;
    estimatorPeriodNs_=period;return true;
  }
  const std::string& fault()const{return fault_;}
  bool dispatch(PolicyWorker& worker,PolicySession& session,int64_t now,
                int64_t stateNs,const double* sensor,bool fresh) {
    if(!fault_.empty()||!worker.attached())return false;
    if(fresh && (stateNs<=0 || stateNs>now || stateNs<=lastQueuedNs_)){
      fault_="policy_dispatch_acquisition_order";return false;
    }
    if(cached_ && (!session.pending() || session.request().id!=job_.h.id ||
                   session.request().generation!=job_.h.generation))cached_=false;
    const bool request=session.pending() &&
      (session.request().id!=queuedId_||session.request().generation!=queuedGeneration_);
    if(request&&!cached_){
      const auto& r=session.request();
      if(!fresh || r.sampleNs!=stateNs){fault_="policy_request_acquisition_mismatch";return false;}
      job_=WorkerJob{};job_.h.flags=1;job_.h.stateNs=stateNs;
      job_.h.generation=r.generation;job_.h.id=r.id;job_.h.sampleNs=r.sampleNs;job_.h.requestNs=r.requestNs;
      std::copy(sensor,sensor+58,job_.sensor);std::copy(r.previousAction.begin(),r.previousAction.end(),job_.previous);
      cached_=true;
    }
    if(!cached_){
      if(!fresh || (lastQueuedNs_ && stateNs-lastQueuedNs_<estimatorPeriodNs_))return false;
      job_=WorkerJob{};job_.h.stateNs=stateNs;std::copy(sensor,sensor+58,job_.sensor);
    }
    if(now<static_cast<int64_t>(job_.h.stateNs) || now-static_cast<int64_t>(job_.h.stateNs)>50000000 ||
       static_cast<int64_t>(job_.h.stateNs)<=lastQueuedNs_){fault_="policy_dispatch_sample_age_or_order";return false;}
    if(!worker.queue(job_))return false; // cached inference retained, bounded by age/session timeout
    lastQueuedNs_=job_.h.stateNs;
    if(cached_){queuedId_=job_.h.id;queuedGeneration_=job_.h.generation;cached_=false;}
    return true;
  }
 private:
  bool cached_=false;WorkerJob job_{};int64_t lastQueuedNs_=0,estimatorPeriodNs_=8000000;
  uint64_t queuedId_=0,queuedGeneration_=0;std::string fault_;
};
}
