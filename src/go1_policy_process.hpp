#pragma once
// Private compute-child lifecycle. No SDK, robot sockets, ownership or arming.
// Start before control threads. Destroy PolicyWorker's duplicated fd before stop.
#include <sys/socket.h>
#include <sys/wait.h>
#include <spawn.h>
#include <dirent.h>
#include <cstdlib>
#include <poll.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
extern char **environ;
namespace go1 {
class PolicyProcess {
 public:
  PolicyProcess()=default;
  PolicyProcess(const PolicyProcess&)=delete;
  PolicyProcess& operator=(const PolicyProcess&)=delete;
  ~PolicyProcess(){stop();}
  int fd()const{return fd_;}
  pid_t pid()const{return pid_;}
  void start(const std::string& python,const std::string& script,
             const std::string& config,const std::string& bundle,
             const std::string& log,const std::string& backend="native",int timeoutMs=20000, const std::string& releaseManifest="") {
    if(pid_>0||fd_>=0)throw std::runtime_error("compute worker already started");
    if(timeoutMs<=0||timeoutMs>20000)throw std::runtime_error("invalid compute startup timeout");
    if(backend!="native"&&backend!="numpy"&&backend!="torch")throw std::runtime_error("unknown actor backend");
    for(const auto* path:{&python,&script,&config,&bundle,&log})
      if(path->empty()||path->front()!='/')throw std::runtime_error("compute paths must be absolute");
    int child=-1,logfd=-1;posix_spawn_file_actions_t actions;bool initialized=false;
    try{
      int sockets[2];if(socketpair(AF_UNIX,SOCK_SEQPACKET|SOCK_CLOEXEC,0,sockets))fail("socketpair");
      fd_=sockets[0];child=sockets[1];
      logfd=open(log.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);
      if(logfd<0)fail("create compute log");
      if(posix_spawn_file_actions_init(&actions))fail("spawn actions");
      initialized=true;
      // Copy endpoints to reserved child numbers, then close originals. Sources
      // are duplicated above 200 first so stdio/fd198 cannot alias a source.
      int childCopy=fcntl(child,F_DUPFD_CLOEXEC,200);if(childCopy<0)fail("duplicate child IPC");
      close(child);child=childCopy;
      int logCopy=fcntl(logfd,F_DUPFD_CLOEXEC,200);if(logCopy<0)fail("duplicate child log");
      close(logfd);logfd=logCopy;
      auto action=[&](int e){if(e)throw std::runtime_error(std::string("spawn action: ")+std::strerror(e));};
      action(posix_spawn_file_actions_adddup2(&actions,child,198));
      action(posix_spawn_file_actions_adddup2(&actions,logfd,STDOUT_FILENO));
      action(posix_spawn_file_actions_adddup2(&actions,logfd,STDERR_FILENO));
      action(posix_spawn_file_actions_addopen(&actions,STDIN_FILENO,"/dev/null",O_RDONLY,0));
      action(posix_spawn_file_actions_addclose(&actions,child));
      action(posix_spawn_file_actions_addclose(&actions,logfd));
      // Do not inherit unrelated SDK/lock/file descriptors even if their
      // owners omitted CLOEXEC. Caller starts this before other threads.
      DIR* directory=opendir("/proc/self/fd");if(!directory)fail("descriptor inventory");
      try{
        while(auto* item=readdir(directory)){
          char* end=nullptr;const long number=std::strtol(item->d_name,&end,10);
          if(!end||*end||number<3||number==198||number==child||number==logfd||number==dirfd(directory))continue;
          action(posix_spawn_file_actions_addclose(&actions,static_cast<int>(number)));
        }
      }catch(...){closedir(directory);throw;}
      closedir(directory);
      std::vector<std::string> values{python,script,"--fd","198","--config",config,"--bundle",bundle,"--policy-backend",backend};
      if(!releaseManifest.empty()){
        if(releaseManifest.front()!='/')throw std::runtime_error("release manifest must be absolute");
        values.push_back("--release-manifest");values.push_back(releaseManifest);
      }
      std::vector<char*> argv;for(auto& v:values)argv.push_back(&v[0]);argv.push_back(nullptr);
      std::vector<std::string> environment;
      for(char** e=environ;*e;++e){std::string v=*e;if(v.compare(0,21,"OPENBLAS_NUM_THREADS=")&&v.compare(0,16,"OMP_NUM_THREADS="))environment.push_back(v);}
      environment.push_back("OPENBLAS_NUM_THREADS=1");environment.push_back("OMP_NUM_THREADS=1");
      std::vector<char*> env;for(auto& v:environment)env.push_back(&v[0]);env.push_back(nullptr);
      const int result=posix_spawn(&pid_,python.c_str(),&actions,nullptr,argv.data(),env.data());
      posix_spawn_file_actions_destroy(&actions);initialized=false;
      close(child);child=-1;close(logfd);logfd=-1;
      if(result){pid_=-1;throw std::runtime_error(std::string("compute spawn: ")+std::strerror(result));}
      const auto deadline=Clock::now()+std::chrono::milliseconds(timeoutMs);
      while(Clock::now()<deadline){
        pollfd p{fd_,POLLIN,0};const int ready=poll(&p,1,10);
        if(ready<0&&errno==EINTR)continue;
        if(ready<0)fail("compute readiness poll");
        if(!ready)continue;
        char buffer[32];const auto n=recv(fd_,buffer,sizeof(buffer),MSG_DONTWAIT|MSG_TRUNC);
        if(n<0&&(errno==EAGAIN||errno==EINTR))continue;
        if(n!=9||std::memcmp(buffer,"GO1READY1",9))throw std::runtime_error("invalid compute readiness");
        return;
      }
      throw std::runtime_error("compute readiness timeout");
    }catch(...){if(initialized)posix_spawn_file_actions_destroy(&actions);if(child>=0)close(child);if(logfd>=0)close(logfd);stop();throw;}
  }
  bool stop() noexcept {
    if(fd_>=0){close(fd_);fd_=-1;}
    if(waitFor(100))return true;
    kill(pid_,SIGTERM);if(waitFor(200))return true;
    kill(pid_,SIGKILL);return waitFor(1000);
  }
 private:
  using Clock=std::chrono::steady_clock;
  [[noreturn]] static void fail(const char* operation){throw std::runtime_error(std::string(operation)+": "+std::strerror(errno));}
  bool waitFor(int ms) noexcept {
    const auto deadline=Clock::now()+std::chrono::milliseconds(ms);
    while(pid_>0){int status=0;const auto got=waitpid(pid_,&status,WNOHANG);
      if(got==pid_||(got<0&&errno==ECHILD)){pid_=-1;return true;}
      if(Clock::now()>=deadline)return false;
      poll(nullptr,0,2);
    }return true;
  }
  int fd_=-1;pid_t pid_=-1;
};
}
