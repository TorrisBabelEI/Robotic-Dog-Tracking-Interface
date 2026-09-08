#include "../src/go1_log_file.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
}

int main() {
  char directory[] = "/tmp/go1-log-test-XXXXXX";
  if (!::mkdtemp(directory)) return 1;
  const std::string path = std::string(directory) + "/existing.csv";
  const std::string link = std::string(directory) + "/link.csv";
  int result = 0;
  try {
    std::istringstream noInput;
    std::ostringstream output;
    require(go1::confirmLogOverwrite(path, noInput, output),
            "new path must not prompt");
    require(output.str().empty(), "new path printed a prompt");
    struct stat info = {};
    require(::lstat(path.c_str(), &info) != 0, "confirmation created a file");
    { std::ofstream file(path); file << "original hardware log\n"; }

    for (const std::string answer : {"", "\n", "n\n", "no\n", "invalid\n",
                                     "y\n", "Y\n", "yes\n", "YES\n"}) {
      std::istringstream input(answer);
      std::ostringstream prompt;
      const bool expected = !answer.empty() &&
                            (answer[0] == 'y' || answer[0] == 'Y');
      require(go1::confirmLogOverwrite(path, input, prompt) == expected,
              "unexpected overwrite decision");
      require(prompt.str().find(path) != std::string::npos,
              "prompt must identify target file");
      std::ifstream file(path);
      std::string content;
      std::getline(file, content);
      require(content == "original hardware log", "confirmation changed old log");
    }

    require(::symlink(path.c_str(), link.c_str()) == 0, "symlink setup failed");
    for (const std::string invalid : {std::string(directory), link}) {
      bool rejected = false;
      std::istringstream input("yes\n");
      try { go1::confirmLogOverwrite(invalid, input, output); }
      catch (const std::runtime_error &) { rejected = true; }
      require(rejected, "non-regular log path must be rejected");
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    result = 1;
  }
  std::remove(link.c_str());
  std::remove(path.c_str());
  ::rmdir(directory);
  return result;
}
