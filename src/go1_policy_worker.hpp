#pragma once
// Bounded local worker IPC. No SDK, network listener, or motor ownership here.
// Same-host little-endian IEEE-754 protocol; private inherited SOCK_SEQPACKET fd.
#include "go1_policy_session.hpp"
#include <sys/socket.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <deque>
#include <cerrno>
#include <iterator>
#include <limits>
namespace go1 {
struct WorkerHeader {
  char magic[8]={'G','O','1','W','K','R','0','1'};
  uint32_t kind=1,flags=0;
  uint64_t sequence=0,stateNs=0,generation=0,id=0,sampleNs=0,requestNs=0;
};
struct WorkerJob {WorkerHeader h; double sensor[58]{},command[3]{.5,0,0},previous[12]{};};
struct WorkerReply {WorkerHeader h; double velocity[3]{},action[12]{};};
static_assert(sizeof(WorkerHeader)==64 && sizeof(WorkerJob)==648 && sizeof(WorkerReply)==184,"IPC layout");
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,"IPC requires binary64");
class PolicyWorker {
 public:
  ~PolicyWorker(){if(fd_>=0)::close(fd_);}
  PolicyWorker()=default;PolicyWorker(const PolicyWorker&)=delete;PolicyWorker& operator=(const PolicyWorker&)=delete;
  bool attach(int fd) {
    if(fd_>=0)return false;
    const uint16_t endian=1;if(*reinterpret_cast<const char*>(&endian)!=1)return false;
    int type=0;socklen_t n=sizeof(type);
    if(getsockopt(fd,SOL_SOCKET,SO_TYPE,&type,&n)||type!=SOCK_SEQPACKET)return false;
    fd_=fcntl(fd,F_DUPFD_CLOEXEC,3);return fd_>=0;
  }
  bool attached() const{return fd_>=0;}
  const std::string& fault()const{return fault_;}
  int64_t estimateNs()const{return estimateNs_;}
  const std::array<double,3>& velocity()const{return velocity_;}
  uint64_t replies()const{return replies_;}uint64_t actions()const{return actions_;}
  uint64_t maxAgeNs()const{return maxAgeNs_;}
  bool estimateValid()const{return estimateValid_ && fault_.empty();}
  bool queue(WorkerJob job) {
    if(fd_<0 || !fault_.empty())return false;
    job.h.sequence=++nextSequence_;
    const auto n=send(fd_,&job,sizeof(job),MSG_DONTWAIT|MSG_NOSIGNAL);
    if(n<0 && (errno==EAGAIN || errno==EWOULDBLOCK))return false;
    if(n!=sizeof(job)){fault_="policy_worker_send_failed";return false;}
    sent_.push_back(job.h);
    if(sent_.size()>64){fault_="policy_worker_backlog";return false;}
    return true;
  }
  void poll(int64_t now,PolicySession& session) {
    if(fd_<0 || !fault_.empty())return;
    // Enforce the same50ms source-age limit even when the worker sends nothing.
    // Otherwise startup/hold could wait for64 queued jobs before detecting loss.
    if(!sent_.empty() && (now<0 || sent_.front().stateNs>static_cast<uint64_t>(now) ||
       static_cast<uint64_t>(now)-sent_.front().stateNs>50000000ULL)) {
      fault_="policy_worker_response_timeout";return;
    }
    for(unsigned count=0;count<8;++count){
      WorkerReply reply{};const auto n=recv(fd_,&reply,sizeof(reply),MSG_DONTWAIT|MSG_TRUNC);
      if(n<0 && (errno==EAGAIN || errno==EWOULDBLOCK))return;
      if(n==0 || (n<0 && (errno==ECONNRESET || errno==ENOTCONN))){fault_="policy_worker_disconnected";return;}
      if(n<0){fault_="policy_worker_receive_failed";return;}
      if(n!=sizeof(reply)||std::memcmp(reply.h.magic,"GO1WKR01",8)||reply.h.kind!=2 || (reply.h.flags&~6u)){
        fault_="policy_worker_bad_frame";return;
      }
      const auto& h=reply.h;
      auto found=std::find_if(sent_.begin(),sent_.end(),[&](const WorkerHeader& x){return x.sequence==h.sequence;});
      if(found==sent_.end() || h.sequence<=lastReceived_ || h.stateNs!=found->stateNs ||
         h.generation!=found->generation || h.id!=found->id || h.sampleNs!=found->sampleNs || h.requestNs!=found->requestNs ||
         h.stateNs==0 || h.stateNs>static_cast<uint64_t>(now) || static_cast<uint64_t>(now)-h.stateNs>50000000ULL){
        fault_="policy_worker_identity_or_age";return;
      }
      if((h.flags&4u) && (!(found->flags&1u) || !(h.flags&2u))){fault_="policy_worker_unsolicited_action";return;}
      ++replies_;maxAgeNs_=std::max(maxAgeNs_,static_cast<uint64_t>(now)-h.stateNs);
      lastReceived_=h.sequence;sent_.erase(sent_.begin(),std::next(found));
      estimateValid_=h.flags&2u;estimateNs_=h.stateNs;
      if(estimateValid_){
        double norm=0;for(double v:reply.velocity){if(!std::isfinite(v)){fault_="policy_worker_nonfinite_velocity";return;}norm+=v*v;}
        std::copy(reply.velocity,reply.velocity+3,velocity_.begin());
        if(norm>4.){fault_="policy_worker_velocity_bound";return;}
      }
      if(h.flags&4u){
        ++actions_;
        std::array<double,12>a;std::copy(reply.action,reply.action+12,a.begin());
        if(!session.submit(h.generation,h.id,h.sampleNs,now,a)){fault_="policy_worker_reply_rejected";return;}
      }
    }
  }
 private:
  int fd_=-1;uint64_t nextSequence_=0,lastReceived_=0;int64_t estimateNs_=0;bool estimateValid_=false;
  std::array<double,3> velocity_{};uint64_t replies_=0,actions_=0,maxAgeNs_=0;
  std::string fault_;std::deque<WorkerHeader> sent_;
};
}
