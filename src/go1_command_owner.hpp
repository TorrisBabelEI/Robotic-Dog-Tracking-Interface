#pragma once
// Advisory ownership for cooperating parent runners on one Linux host.
// Does not stop or prove ownership against the independent factory controller.
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#endif
namespace go1 {
class CommandOwner {
public:
  explicit CommandOwner(const std::string& path="/tmp/go1-lowlevel-owner.lock") {
#ifdef __linux__
    fd_=::open(path.c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd_<0)throw std::runtime_error("Cannot open Go1 command-owner lock; no UDP opened");
    if(::flock(fd_,LOCK_EX|LOCK_NB)!=0){::close(fd_);fd_=-1;throw std::runtime_error("Go1 command owner already exists; no UDP opened");}
#else
    throw std::runtime_error("Go1 ownership lock requires Linux");
#endif
  }
  ~CommandOwner(){
#ifdef __linux__
    if(fd_>=0){::flock(fd_,LOCK_UN);::close(fd_);}
#endif
  }
  CommandOwner(const CommandOwner&)=delete;
  CommandOwner& operator=(const CommandOwner&)=delete;
private:int fd_=-1;
};
}
