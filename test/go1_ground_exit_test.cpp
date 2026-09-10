// Exercise the actual control core without linking Unitree or opening UDP.
#define GO1_CORE_TEST
#include "../src/go1_lowlevel_experiment.cpp"

namespace {
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
Options endpointOptions() {
  Options options;
  options.mode = ExperimentMode::GroundHandover;
  options.dryRun = true;
  options.dryRunNormalExit = true;
  return options;
}
struct Fixture {
  ExperimentCore core{endpointOptions()};
  Feedback f;
  Command command;
  int64_t now = 1000000000LL;
  Fixture() {
    f.levelFlag = kLowLevel;
    f.remote.valid = true;
    for (std::size_t i = 0; i < kJointCount; ++i) {
      f.joint[i].q = i % 3 == 0 ? 0 : i % 3 == 1 ? 0.8F : -1.5F;
      f.joint[i].mode = kServoMode;
      f.joint[i].temperature = 30;
    }
    f.footForce = {{40, 40, 40, 40}};
  }
  void tick(bool support = false, bool fresh = true, bool alive = true,
            int send = 0, bool watchdog = false) {
    now += 2000000;
    if (fresh) f.tickMs += 2;
    command = core.step(f, true, fresh, alive, 0, send, 2000, now,
                        watchdog, support);
    // Ideal tracking fixture, not contact or motor physics. Floor support is
    // independently supplied by each test and never inferred from this pose.
    for (std::size_t i = 0; i < kJointCount; ++i) {
      if (command.joint[i].q < 1.0e8F) f.joint[i].q = command.joint[i].q;
      f.joint[i].dq = command.joint[i].dq;
    }
  }
  void reach(Phase wanted) {
    for (int i = 0; i < 20000 && core.phase() != wanted; ++i) tick();
    require(core.phase() == wanted, "requested phase not reached");
  }
  void damping() const {
    for (const auto &j : command.joint)
      require(j.q == kPosStop && j.kp == 0 && j.kd == 1 && j.tauFf == 0,
              "fault/exit must publish damping in the same cycle");
  }
};

void normal() {
  Fixture t;
  t.reach(Phase::ExitLower);
  auto previous = t.command;
  const int64_t start = t.now;
  while (t.core.phase() == Phase::ExitLower) {
    t.tick();
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const auto &j = t.command.joint[i];
      require(j.q >= kJointMin[i % 3] && j.q <= kJointMax[i % 3], "joint bounds");
      require(std::fabs(j.q - previous.joint[i].q) < 0.0007, "position discontinuity");
      require(std::fabs(j.dq) < 0.3 && j.tauFf == 0, "lowering speed or torque");
    }
    previous = t.command;
  }
  require(t.now - start >= 8000000000LL, "lowering finished early");
  require(t.core.phase() == Phase::ExitVerify, "lowering bypassed support verification");
  for (int i = 0; i < 490; ++i) t.tick(true);
  require(t.core.phase() == Phase::ExitVerify, "support accepted before stable second");
  for (int i = 0; i < 20 && t.core.phase() == Phase::ExitVerify; ++i) t.tick(true);
  require(t.core.phase() == Phase::ExitDamping, "independent support not accepted");
  t.damping();
  const int64_t dampingStart = t.now;
  while (!t.core.done()) { t.tick(); t.damping(); }
  require(t.now - dampingStart >= 1000000000LL, "damping dwell too short");
  require(!t.core.failed(), "normal exit faulted");
}

void missingSupport() {
  Fixture t;
  // Earlier observations (including during lowering) cannot authorize exit.
  while (t.core.phase() != Phase::ExitVerify) t.tick(true);
  for (int i = 0; i < 2600; ++i) t.tick(false);
  require(t.core.phase() == Phase::ExitHold && !t.core.done(), "no support must latch hold");
  require(t.core.faultReason() == "exit_support_not_confirmed", "missing support reason");
  require(t.command.joint[1].kp > 0, "absent support must not drop impedance");
}

void cancel() {
  for (Phase phase : {Phase::ExitLower, Phase::ExitVerify}) {
    Fixture t; t.reach(phase);
    for (int i = 0; i < 100; ++i) t.tick();
    auto before = t.command;
    t.core.observeSignalCount(1, t.now, t.f);
    t.tick(true);
    require(t.core.phase() == Phase::ExitHold, "cancel must hold, not resume standing");
    for (std::size_t i = 0; i < kJointCount; ++i)
      require(t.command.joint[i].q == before.joint[i].q &&
              t.command.joint[i].dq == 0 && t.command.joint[i].tauFf == 0,
              "cancel must retain last target and remove velocity/torque");
    for (int i = 0; i < 600; ++i) t.tick(true);
    require(!t.core.done() && t.core.phase() == Phase::ExitHold, "cancel must remain latched");
  }
  for (Phase phase : {Phase::GroundHandover, Phase::Return}) {
    Fixture t; t.reach(phase);
    t.core.observeSignalCount(1, t.now, t.f);
    t.reach(Phase::SafeHold);
    require(t.core.phase() == Phase::SafeHold, "early cancel must not start lowering");
  }
}

void faults() {
  for (Phase phase : {Phase::ExitLower, Phase::ExitVerify, Phase::ExitDamping}) {
    Fixture t;
    t.reach(phase == Phase::ExitDamping ? Phase::ExitVerify : phase);
    if (phase == Phase::ExitDamping)
      while (t.core.phase() == Phase::ExitVerify) t.tick(true);
    for (int i = 0; i < 10; ++i) t.tick(false, false, true);
    t.tick(false, false, false); // feedback age just exceeded 20 ms
    require(t.core.phase() == Phase::PanicDamping, "lost feedback must panic");
    require(t.core.faultReason() == "feedback_gap_over_20ms", "wrong feedback fault");
    t.damping();
  }
  Fixture t; t.reach(Phase::ExitLower);
  t.tick(false, true, true, 0, true);
  require(t.core.phase() == Phase::PanicDamping, "watchdog must panic"); t.damping();
  Fixture r; r.reach(Phase::ExitLower);
  r.f.remote.buttons = kRemoteL2Mask | kRemoteBMask;
  r.tick(); require(r.core.phase() == Phase::PanicDamping, "remote stop"); r.damping();
  Fixture c; c.reach(Phase::ExitLower);
  c.core.observeSignalCount(2, c.now, c.f);
  c.tick(); require(c.core.phase() == Phase::PanicDamping, "double Ctrl-C"); c.damping();
  Fixture s; s.reach(Phase::ExitVerify);
  while (s.core.phase() == Phase::ExitVerify) s.tick(true);
  s.tick(false, true, true, -1);
  require(s.core.phase() == Phase::PanicDamping, "failed damping transmission"); s.damping();
  Fixture n; n.reach(Phase::ExitLower);
  n.f.rpy[0] = NAN; n.tick();
  require(n.core.phase() == Phase::PanicDamping, "nonfinite IMU"); n.damping();
  Fixture b; b.reach(Phase::ExitVerify);
  b.f.joint[2].q = -2.80F; b.tick();
  require(b.core.phase() == Phase::PanicDamping, "factory prone feedback must not widen command bounds"); b.damping();
  Fixture e; e.reach(Phase::ExitLower);
  e.f.joint[1].q += 0.15F; e.tick();
  require(e.core.phase() == Phase::PanicDamping, "exit tracking error"); e.damping();
}

void hardwareLock() {
  char program[] = "test", mode[] = "--mode", ground[] = "--ground-confirmed";
  for (const char *name : {"ground-handover", "squat", "leg-lift", "leg-lift-sequence"}) {
    std::string value(name);
    char remote[] = "--remote-confirmed";
    char *argv[] = {program, mode, &value[0], ground, remote};
    bool locked = false;
    try { parseOptions(5, argv); }
    catch (const std::runtime_error &e) {
      locked = std::string(e.what()).find("ground hardware modes are locked") != std::string::npos;
    }
    require(locked, "unreviewed ground hardware must fail before UDP/ARM");
  }
}
}
int main() {
  struct TestCase { const char *name; void (*run)(); };
  const TestCase cases[] = {
      {"normal descent -> support verification -> damping -> completion", normal},
      {"missing/premature support -> latched hold, no automatic exit", missingSupport},
      {"cancel during handover/return/descent/verification", cancel},
      {"feedback loss, watchdog, stops, send failure and invalid feedback", faults},
      {"ground hardware CLI rejected before UDP", hardwareLock}};
  for (const auto &test : cases) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception &error) {
      std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
      return 1;
    }
  }
  std::cout << "Ground exit tests passed (synthetic feedback/support only).\n";
  return 0;
}
