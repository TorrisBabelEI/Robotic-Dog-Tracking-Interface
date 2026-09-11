#define GO1_CORE_TEST
#include "../src/go1_lowlevel_experiment.cpp"

namespace {
void require(bool value, const char *reason) {
  if (!value) throw std::runtime_error(reason);
}
Feedback standing() {
  Feedback f;
  f.levelFlag = 0xEE;
  for (std::size_t i = 0; i < kJointCount; ++i) {
    f.joint[i].q = i % 3 == 0 ? 0 : i % 3 == 1 ? 0.8F : -1.5F;
    f.joint[i].mode = kServoMode;
  }
  return f;
}
void normal() {
  StandingPoseCapture c;
  const auto f = standing();
  std::array<float, kJointCount> output;
  output.fill(999);
  for (int i = 1; i <= 100; ++i)
    require(!c.observe(f, i, 1000000000LL + i * 2000000LL, true, &output),
            "must require full 200 ms even with 100 samples");
  require(output[0] == 999, "partial capture modified output");
  require(c.observe(f, 101, 1202000000LL, true, &output), "quiet capture failed");
  for (std::size_t i = 0; i < kJointCount; ++i)
    require(std::fabs(output[i] - f.joint[i].q) < 1e-6, "wrong captured mean");
  std::cout << "[PASS] quiet capture requires sample count and elapsed time\n";
}
void cached() {
  const auto f = standing();
  std::array<float, kJointCount> output;
  StandingPoseCapture c;
  require(!c.observe(f, 1, 1000000000LL, true, &output), "early capture");
  for (int i = 1; i < 200; ++i)
    require(!c.observe(f, 1, 1000000000LL + i * 2000000LL, true, &output),
            "cached receive counter counted as fresh");
  require(c.count() == 0, "cached data did not expire");
  for (int i = 200; i < 400; ++i)
    require(!c.observe(f, i, 1000000000LL + i * 2000000LL, false, &output),
            "unsuccessful receive counted as fresh");
  std::cout << "[PASS] cached data and unsuccessful receives cannot complete capture\n";
}
void interruptions() {
  std::array<float, kJointCount> output;
  for (int kind = 0; kind < 8; ++kind) {
    StandingPoseCapture c;
    auto f = standing();
    for (int i = 1; i <= 80; ++i)
      c.observe(f, i, 1000000000LL + i * 2000000LL, true, &output);
    auto bad = f;
    if (kind == 0) bad.joint[0].dq = 0.06F;
    if (kind == 1) bad.joint[0].q = 0.03F;
    if (kind == 2) bad.rpy[2] = NAN;
    if (kind == 3) bad.joint[2].q = -2.8F;
    if (kind == 4) bad.joint[0].temperature = 70;
    if (kind == 5) bad.levelFlag = kLowLevel;
    const int64_t now = kind == 6 ? 1200000000LL : 1162000000LL;
    const int seq = kind == 7 ? 1 : 81;
    require(!c.observe(bad, seq, now, true, &output), "invalid capture accepted");
    require(c.count() <= 1, "interrupted window retained samples");
    for (int i = 1; i <= 50; ++i)
      require(!c.observe(f, seq + i, now + i * 2000000LL, true, &output),
              "partial windows combined across interruption");
  }
  std::cout << "[PASS] movement, invalid feedback, gaps and counter rollback reset capture\n";
}
void countGate() {
  StandingPoseCapture c;
  auto f = standing();
  std::array<float, kJointCount> output;
  for (int i = 1; i <= 99; ++i)
    require(!c.observe(f, i, 1000000000LL + i * 10000000LL, true, &output),
            "elapsed time bypassed sample count");
  require(c.observe(f, 100, 2000000000LL, true, &output), "valid slower stream rejected");
  std::cout << "[PASS] elapsed time alone cannot bypass minimum sample count\n";
}
}
int main() {
  try { normal(); cached(); interruptions(); countGate(); }
  catch (const std::exception &e) { std::cerr << "[FAIL] " << e.what() << '\n'; return 1; }
  std::cout << "Standing capture tests passed (synthetic transport and feedback only).\n";
  return 0;
}
