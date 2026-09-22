// Exposes the actual parent ExperimentCore for simulator integration. No SDK.
#define GO1_CORE_TEST
#define GO1_WITH_POLICY 1
#include "go1_lowlevel_experiment.cpp"
#include "go1_policy_worker.hpp"
#include "go1_policy_dispatch.hpp"
#include "go1_policy_evidence.hpp"
namespace {
Options offlineOptions(){Options o;o.mode=ExperimentMode::WalkingPolicy;o.dryRun=true;return o;}
struct OfflinePolicyCore {
  ExperimentCore core{offlineOptions()};
  uint64_t stateSequence=0;
  int64_t lastNs=0,lastAcquisitionNs=0;
  go1::ControlEvidence input;bool useOperatorAdapter=false;
  go1::PolicyWorker worker;go1::PolicyDispatch dispatch;
  OfflinePolicyCore(){
    std::array<float,12> q;
    for(unsigned i=0;i<12;++i)q[i]=i%3==0?0:i%3==1?.9F:-1.8F;
    core.seedGroundPose(q);
  }
};
}
extern "C" {
int go1_policy_operator_frame(void* p,const uint8_t* bytes,int64_t sampleNs,int64_t receiveNs,int qualified) {
  if(!p || !bytes)return 0;
  auto& o=*static_cast<OfflinePolicyCore*>(p);std::array<uint8_t,40> b{};
  std::copy(bytes,bytes+40,b.begin());o.useOperatorAdapter=true;
  o.input.acquireRadio(b,sampleNs,receiveNs,qualified!=0);return 1;
}
void* go1_policy_create(){try{return new OfflinePolicyCore();}catch(...){return nullptr;}}
void go1_policy_seed(void* p,const double* values){
  if(!p)return;std::array<float,12> q;for(unsigned i=0;i<12;++i)q[i]=values[i];
  static_cast<OfflinePolicyCore*>(p)->core.seedGroundPose(q);
}
int go1_policy_seed_support(void* p,const double* torque,const double* velocity) {
  if(!p)return 0;
  std::array<float,12> tau{},dq{};
  for(unsigned i=0;i<12;++i){tau[i]=torque[i];dq[i]=velocity[i];}
  return static_cast<OfflinePolicyCore*>(p)->core.seedOfflinePolicySupport(tau,dq);
}
int go1_policy_estimator_period(void* p,int64_t periodNs) {
  if(!p)return 0;
  auto& o=*static_cast<OfflinePolicyCore*>(p);
  if(o.lastNs || o.worker.attached())return 0;
  return o.dispatch.configureEstimatorPeriodNs(periodNs);
}
int go1_policy_worker_attach(void* p,int fd){return p && static_cast<OfflinePolicyCore*>(p)->worker.attach(fd);}
void go1_policy_worker_stats(void* p,double* out){
  if(!p)return;auto& w=static_cast<OfflinePolicyCore*>(p)->worker;
  out[0]=w.estimateValid();out[1]=w.estimateNs();std::copy(w.velocity().begin(),w.velocity().end(),out+2);
  out[5]=w.replies();out[6]=w.actions();out[7]=w.maxAgeNs();
}
void go1_policy_destroy(void* p){delete static_cast<OfflinePolicyCore*>(p);}
int go1_policy_advance(void* p,int64_t now,int64_t stateNs,uint32_t tick,
                       const double* sensor,int64_t operatorNs,int held,int stop,
                       int64_t estimateNs,int estimateValid,int owner,double* command) {
  if(!p)return -1;
  auto& o=*static_cast<OfflinePolicyCore*>(p);Feedback f;f.levelFlag=kLowLevel;f.tickMs=tick;
  for(unsigned i=0;i<12;++i){f.joint[i].q=sensor[i];f.joint[i].dq=sensor[12+i];f.joint[i].tauEst=sensor[37+i];f.joint[i].temperature=30;f.joint[i].mode=kServoMode;}
  for(unsigned i=0;i<3;++i){f.gyro[i]=sensor[24+i];f.accel[i]=sensor[27+i];f.rpy[i]=sensor[30+i];}
  for(unsigned i=0;i<4;++i)f.footForce[i]=sensor[33+i];
  f.remote.valid=true;f.remote.buttons=stop?(kRemoteL2Mask|kRemoteBMask):0;
  go1::PolicyEvidence e;e.nowNs=now;e.stateNs=stateNs;e.operatorNs=operatorNs;e.estimateNs=estimateNs;
  const bool fresh=stateNs!=o.lastAcquisitionNs;
  if(fresh){
    if(stateNs<o.lastAcquisitionNs)o.core.forceHardFault("policy_acquisition_clock_backwards",now);
    o.lastAcquisitionNs=stateNs;++o.stateSequence;
  }
  e.stateSequence=o.stateSequence;e.owner=owner;e.operatorVerified=true;e.held=held;e.stop=stop;e.estimateValid=estimateValid;
  if(o.useOperatorAdapter){
    e=o.input.compose(now,stateNs,o.stateSequence,estimateNs,estimateValid,owner);
    e.stop=e.stop || stop; // Additional explicit mock stop cannot be masked.
  }
  e.inferenceSampleReady=fresh;
  if(o.worker.attached()) {
    o.worker.poll(now,o.core.policySession());
    e.estimateNs=o.worker.estimateNs();e.estimateValid=o.worker.estimateValid();
    if(!o.worker.fault().empty())o.core.forceHardFault(o.worker.fault(),now);
  }
  o.core.setPolicyEvidence(e);
  const double loopUs=o.lastNs?(now-o.lastNs)/1000.:2000.;o.lastNs=now;
  const bool alive=now>=stateNs && now-stateNs<=50000000;
  auto c=o.core.step(f,true,fresh,alive,0,0,loopUs,now,false);
  auto& session=o.core.policySession();
  if(o.worker.attached() && !o.core.failed()) {
    o.dispatch.dispatch(o.worker,session,now,stateNs,sensor,fresh);
    const auto& fault=o.dispatch.fault().empty()?o.worker.fault():o.dispatch.fault();
    if(!fault.empty()){o.core.forceHardFault(fault,now);c=o.core.emergencyDampingCommand();}
  }
  for(unsigned i=0;i<12;++i){command[i]=c.joint[i].q;command[12+i]=c.joint[i].dq;command[24+i]=c.joint[i].kp;command[36+i]=c.joint[i].kd;command[48+i]=c.joint[i].tauFf;}
  return o.core.failed()?-1:o.core.phase()==Phase::WalkingPolicy?1:0;
}
int go1_policy_request(void* p,uint64_t* info,double* previous) {
  if(!p)return 0;
  auto& s=static_cast<OfflinePolicyCore*>(p)->core.policySession();
  if(!s.pending())return 0;
  const auto& r=s.request();info[0]=r.generation;info[1]=r.id;info[2]=r.stateSequence;info[3]=r.sampleNs;info[4]=r.requestNs;
  for(unsigned i=0;i<12;++i)previous[i]=r.previousAction[i];return 1;
}
int go1_policy_submit(void* p,uint64_t generation,uint64_t id,int64_t sampleNs,int64_t now,const double* action) {
  if(!p)return 0;std::array<double,12>a;std::copy(action,action+12,a.begin());
  return static_cast<OfflinePolicyCore*>(p)->core.policySession().submit(generation,id,sampleNs,now,a);
}
void go1_policy_ack(void* p,const double* q,int position,int success) {
  if(!p)return;std::array<double,12>a;std::copy(q,q+12,a.begin());
  static_cast<OfflinePolicyCore*>(p)->core.policySession().acknowledge(a,position,success);
}
const char* go1_policy_fault(void* p){return p?static_cast<OfflinePolicyCore*>(p)->core.faultReason().c_str():"null core";}
}
