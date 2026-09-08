#pragma once

#include <cerrno>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace go1 {
// Called before arming or creating the hardware runner. Confirmation itself
// never opens or truncates the existing log; writeLog replaces it at run end.
inline bool confirmLogOverwrite(const std::string &path, std::istream &input,
                                std::ostream &output) {
  struct stat info = {};
  if (::lstat(path.c_str(), &info) != 0) {
    if (errno == ENOENT) return true;
    throw std::runtime_error("cannot inspect log path: " + path + ": " +
                             std::strerror(errno));
  }
  if (!S_ISREG(info.st_mode))
    throw std::runtime_error("log path must be a regular file, not a directory "
                             "or symbolic link: " + path);
  output << "Log already exists: " << path
         << "\nReplace it with this run's log when the run finishes? [y/N]: "
         << std::flush;
  std::string answer;
  if (!std::getline(input, answer) ||
      (answer != "y" && answer != "Y" && answer != "yes" && answer != "YES")) {
    output << "Cancelled; existing log preserved. No motor commands sent.\n";
    return false;
  }
  output << "Overwrite confirmed. The existing log is kept until this run "
            "writes its results.\n";
  return true;
}
} // namespace go1
