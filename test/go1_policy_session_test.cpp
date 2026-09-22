#include "../src/go1_policy_session.hpp"
#include "../src/go1_command_owner.hpp"
#include <iostream>
#include <limits>
#include <functional>
#include <unistd.h>
void require(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
std::array<double,12> q0(){std::array<double,12> q;for(unsigned i=0;i<12;++i)q[i]=i%3==0?0:i%3==1?.9:-1.8;return q;}
struct Fixture {
 go1::PolicySession p;go1::PolicyEvidence e;
 Fixture(){e.nowNs=1000000000;e.owner=e.operatorVerified=e.estimateValid=true;p.acknowledge(q0(),true,true);tick(false);}
 bool tick(bool held=true){e.nowNs+=2000000;e.stateNs=e.operatorNs=e.estimateNs=e.nowNs;++e.stateSequence;e.held=held;return p.tick(e);}
 void reply(double a=.2){auto r=p.request();std::array<double,12> v;v.fill(a);require(p.submit(r.generation,r.id,r.sampleNs,e.nowNs,v),"reply accepted");}
};
int main(){try{
 {Fixture f;require(f.tick(),"release then press requests policy");f.reply();unsigned requests=1;
  for(int i=0;i<500;++i){if(f.tick()){++requests;f.reply();}auto q=f.p.target(f.e.nowNs);f.p.acknowledge(q,true,true);}
  require(requests==51,"50 Hz phase-locked policy schedule on 500 Hz loop");
  auto sent=q0();sent[0]=.1;f.p.acknowledge(sent,true,true);require(std::abs(f.p.previousAction()[0]-.4)<1e-12,"previous action derives from final sent target");
  f.tick(false);require(!f.p.fault().empty(),"release faults immediately");require(!f.p.reset(false),"held cannot clear latch");require(f.p.reset(true),"explicit released reset");
  require(!f.tick(),"reset requires a new release observation");}
 {go1::PolicySession p;go1::PolicyEvidence e;e.nowNs=e.stateNs=e.operatorNs=e.estimateNs=1000000000;e.owner=e.operatorVerified=e.estimateValid=e.held=true;
  p.acknowledge(q0(),true,true);require(!p.tick(e)&&!p.running(),"held-at-start cannot arm");}
 for(int fault=0;fault<6;++fault){Fixture f;f.tick();f.reply();f.e.nowNs+=2000000;
  if(fault==0)f.e.stop=true;if(fault==1)f.e.owner=false;if(fault==2)f.e.operatorNs-=100000000;
  if(fault==3)f.e.stateNs-=100000000;if(fault==4)f.e.estimateValid=false;if(fault==5)f.e.nowNs+=20000000;
  f.p.tick(f.e);require(!f.p.fault().empty(),"loss/stop/deadline latches fault");}
 {Fixture f;f.tick();for(int i=0;i<30;++i)f.tick();require(!f.p.fault().empty(),"worker loss times out");}
 {Fixture f;f.tick();auto r=f.p.request();std::array<double,12>a{{}};require(!f.p.submit(r.generation+1,r.id,r.sampleNs,f.e.nowNs,a),"old session rejected");}
 {Fixture f;f.tick();auto r=f.p.request();std::array<double,12>a{{}};a[2]=std::numeric_limits<double>::quiet_NaN();require(!f.p.submit(r.generation,r.id,r.sampleNs,f.e.nowNs,a),"NaN rejected");}
 {Fixture f;f.tick();f.p.acknowledge(q0(),true,false);require(!f.p.appliedValid()&&!f.p.fault().empty(),"send failure invalidates previous action");}
 {const std::string path="/tmp/go1-policy-owner-test-"+std::to_string(getpid());
  {go1::CommandOwner first(path);bool denied=false;try{go1::CommandOwner second(path);}catch(...){denied=true;}require(denied,"duplicate motor owner denied");}
  {go1::CommandOwner recovered(path);}unlink(path.c_str());}
 std::cout<<"policy lifecycle, timing, sent-action, stop, loss, recovery and owner tests passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
