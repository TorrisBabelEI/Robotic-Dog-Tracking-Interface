// Actual controller core, synthetic feedback only; no SDK or robot UDP.
#define GO1_CORE_TEST
#include "../src/go1_lowlevel_experiment.cpp"

namespace {
void require(bool value, const char *reason) {
  if (!value) throw std::runtime_error(reason);
}
Options options() {
  Options o;
  o.mode = ExperimentMode::GroundHandover;
  o.dryRun = true;
  return o;
}
struct Fixture {
  ExperimentCore core{options()};
  Feedback f;
  Command command;
  std::array<float, kJointCount> seed;
  int64_t now = 1000000000LL;
  Fixture() {
    f.levelFlag = kLowLevel;
    f.remote.valid = true;
    for (std::size_t i = 0; i < kJointCount; ++i) {
      seed[i] = i % 3 == 0 ? 0 : i % 3 == 1 ? 0.8F : -1.5F;
      f.joint[i].q = seed[i];
      f.joint[i].mode = kServoMode;
    }
    core.seedGroundPose(seed);
    command = core.seededGroundHoldCommand();
  }
  void tick(bool has = true, bool fresh = true, bool alive = true,
            int send = 0, bool watchdog = false, bool advanceTick = true) {
    now += 2000000;
    if (fresh && advanceTick) f.tickMs += 2;
    command = core.step(f, has, fresh, alive, 0, send, 2000, now, watchdog);
  }
  void held() {
    for (std::size_t i = 0; i < kJointCount; ++i)
      require(command.joint[i].q == seed[i] && command.joint[i].dq == 0 &&
                  command.joint[i].tauFf == 0 && command.joint[i].kp > 0,
              "seeded position changed across entry");
  }
  void panic() {
    require(core.phase() == Phase::PanicDamping && !core.done(), "panic must latch");
    for (const auto &j : command.joint)
      require(j.q == kPosStop && j.kp == 0 && j.kd == 1 && j.tauFf == 0,
              "entry fault must send position-free damping in same cycle");
  }
};
void continuity() {
  Fixture t; t.held();
  // A small, permitted seed/feedback difference must not become a target step.
  for (auto &j : t.f.joint) j.q += 0.02F;
  for (int i = 0; i < 250; ++i) { t.tick(); t.held(); }
  require(t.core.phase() == Phase::CapturePose, "capture not reached");
  t.tick(true, false); t.held();
  require(t.core.phase() == Phase::CapturePose, "capture used repeated feedback");
  t.tick(); t.held();
  require(t.core.phase() == Phase::Hold, "fresh capture failed");
  std::cout << "[PASS] seed target remains continuous through fresh capture and hold\n";
}
void invalidFeedback() {
  for (int kind = 0; kind < 7; ++kind) {
    Fixture t; t.tick();
    if (kind == 0) t.f.joint[1].q += 0.06F;
    if (kind == 1) t.f.joint[1].dq = 0.06F;
    if (kind == 2) t.f.rpy[0] = NAN;
    if (kind == 3) t.f.levelFlag = 0;
    if (kind == 4) t.f.joint[2].q = -2.80F;
    if (kind == 5) t.f.joint[0].temperature = 70;
    t.tick(true, true, true, 0, false, kind != 6);
    t.panic();
  }
  std::cout << "[PASS] mismatched, moving, invalid and duplicate feedback reject entry\n";
}
void failures() {
  Fixture missing;
  for (int i = 0; i < 12; ++i) missing.tick(false, false, false);
  missing.panic();
  Fixture lost; lost.tick();
  for (int i = 0; i < 12; ++i) lost.tick(true, false, false);
  lost.panic();
  Fixture send; send.tick(true, true, true, -1); send.panic();
  Fixture watchdog; watchdog.tick(true, true, true, 0, true); watchdog.panic();
  Fixture remote; remote.f.remote.buttons = kRemoteL2Mask | kRemoteBMask;
  remote.tick(); remote.panic();
  std::cout << "[PASS] feedback loss, send failure, watchdog and remote stop latch damping\n";
}
void invalidSeed() {
  for (float value : {NAN, -2.80F}) {
    Fixture t; auto bad = t.seed; bad[2] = value;
    bool rejected = false;
    try { t.core.seedGroundPose(bad); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "invalid seed accepted");
  }
  std::cout << "[PASS] nonfinite and out-of-bounds seed poses rejected\n";
}
}
int main() {
  try { continuity(); invalidFeedback(); failures(); invalidSeed(); }
  catch (const std::exception &error) {
    std::cerr << "[FAIL] " << error.what() << '\n'; return 1;
  }
  std::cout << "Ground entry core tests passed (synthetic feedback only).\n";
  return 0;
}
