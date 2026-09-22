#include "../src/go1_policy_worker.hpp"
#include <iostream>
#include <stdexcept>
void check(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(){try{
 for(int fault=0;fault<7;++fault){
  int fd[2];check(socketpair(AF_UNIX,SOCK_SEQPACKET,0,fd)==0,"socketpair");
  go1::PolicyWorker worker;go1::PolicySession session;check(worker.attach(fd[0]),"attach");close(fd[0]);
  go1::WorkerJob job;job.h.stateNs=1000000000;check(worker.queue(job),"queue");
  check(recv(fd[1],&job,sizeof(job),0)==sizeof(job),"receive job");
  go1::WorkerReply r;r.h=job.h;r.h.kind=2;r.h.flags=2;
  if(fault==1)r.h.sequence+=1;
  if(fault==2)r.velocity[1]=std::numeric_limits<double>::quiet_NaN();
  if(fault==3)r.h.flags|=4;
  if(fault==4)r.h.magic[0]='X';
  if(fault==5){close(fd[1]);worker.poll(1002000000,session);check(!worker.fault().empty(),"disconnect rejected");continue;}
  const auto len=fault==6?sizeof(r)-1:sizeof(r);send(fd[1],&r,len,0);worker.poll(1002000000,session);
  check(fault==0?worker.fault().empty():!worker.fault().empty(),"reply check");
  if(fault==0){check(worker.estimateValid()&&worker.replies()==1,"valid measurement");send(fd[1],&r,sizeof(r),0);worker.poll(1004000000,session);check(!worker.fault().empty(),"replay rejected");}
  close(fd[1]);
 }
 {int fd[2];socketpair(AF_UNIX,SOCK_SEQPACKET,0,fd);go1::PolicyWorker w;go1::PolicySession s;w.attach(fd[0]);close(fd[0]);
  go1::WorkerJob j;j.h.stateNs=1000000000;w.queue(j);recv(fd[1],&j,sizeof(j),0);go1::WorkerReply r;r.h=j.h;r.h.kind=2;send(fd[1],&r,sizeof(r),0);w.poll(1050000001,s);check(!w.fault().empty(),"stale rejected");close(fd[1]);}
 {int fd[2];check(socketpair(AF_UNIX,SOCK_SEQPACKET,0,fd)==0,"reset socketpair");
  go1::PolicyWorker w;go1::PolicySession s;check(w.attach(fd[0]),"reset attach");close(fd[0]);
  go1::WorkerJob j;j.h.stateNs=1000000000;check(w.queue(j),"reset queue");
  close(fd[1]); // Unread queued job can make recv fail with ECONNRESET, not EOF.
  w.poll(1002000000,s);check(w.fault()=="policy_worker_disconnected","reset classified as disconnect");}
 std::cout<<"worker protocol identity, finite values, replay, age, disconnect, size checks passed\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
