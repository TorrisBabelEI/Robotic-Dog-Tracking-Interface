// Exercise the prone low-rise control core without linking Unitree or opening UDP.
#define GO1_CORE_TEST
#include "../src/go1_lowlevel_experiment.cpp"

namespace {
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

Options proneOptions() {
  Options options;
  options.mode = ExperimentMode::ProneLowRise;
  options.dryRun = true;
  return options;
}

struct Fixture {
  ExperimentCore core{proneOptions()};
  Feedback f;
  Feedback feedbackAtStep;
  Command command;
  int64_t now = 1000000000LL;

  Fixture() {
    const std::array<float, kJointCount> observedProne{{
        -0.311616F, 1.275168F, -2.794212F,
         0.289090F, 1.272262F, -2.797724F,
        -0.289514F, 1.280437F, -2.799541F,
         0.302533F, 1.237503F, -2.768375F}};
    f.levelFlag = kLowLevel;
    f.remote.valid = true;
    f.accel[2] = 9.81F;
    for (std::size_t i = 0; i < kJointCount; ++i) {
      f.joint[i].q = observedProne[i];
      f.joint[i].mode = kServoMode;
      f.joint[i].temperature = 30;
    }
    f.footForce = {{50, 50, 50, 50}};
  }

  void tick(bool support = false, bool fresh = true, bool alive = true,
            int send = 0, bool watchdog = false, bool track = true) {
    now += 2000000;
    if (fresh) f.tickMs += 2;
    feedbackAtStep = f;
    command = core.step(f, true, fresh, alive, 0, send, 2000, now,
                        watchdog, support);
    // Ideal joint tracking is separate from the support observation. Tests
    // supply support explicitly and never infer it from pose or foot force.
    if (track) {
      for (std::size_t i = 0; i < kJointCount; ++i) {
        if (command.joint[i].q < 1.0e8F)
          f.joint[i].q = command.joint[i].q;
        f.joint[i].dq = command.joint[i].dq;
      }
    }
  }

  void reach(Phase wanted) {
    for (int i = 0; i < 20000 && core.phase() != wanted; ++i) tick();
    require(core.phase() == wanted, "requested prone phase not reached");
  }

  void damping() const {
    for (const auto &joint : command.joint)
      require(joint.mode == kDampingMode && joint.q == kPosStop &&
                  joint.kp == 0.0F && joint.kd == 1.0F &&
                  joint.tauFf == 0.0F,
              "fault/final phase must publish damping immediately");
  }
};

void validateBoundedPositionCommand(const Fixture &t) {
  for (std::size_t i = 0; i < kJointCount; ++i) {
    const auto &joint = t.command.joint[i];
    require(joint.mode == kServoMode, "position phase must use servo mode");
    require(joint.q >= kJointMin[i % 3] && joint.q <= kJointMax[i % 3],
            "prone target outside SDK command bounds");
    require(joint.kp >= 0.0F && joint.kp <= kProneKp &&
                joint.kd == kProneKd && joint.tauFf == 0.0F,
            "prone gains or feed-forward torque exceed design envelope");
    const double total = joint.kp * (joint.q - t.feedbackAtStep.joint[i].q) +
                         joint.kd * (joint.dq - t.feedbackAtStep.joint[i].dq) +
                         joint.tauFf;
    require(std::fabs(total) <= kProneEffortLimitNm + 1.0e-5,
            "predicted joint effort exceeds 0.5 Nm");
  }
}

void nominal() {
  Fixture t;
  std::vector<Phase> visited;
  Phase previous = t.core.phase();
  std::array<float, kJointCount> engagement{{0}}, rise{{0}};
  bool capturedEngagement = false, capturedRise = false;
  float priorReleaseKp = kProneKp;
  int releaseSamples = 0;

  for (int cycle = 0; cycle < 15000 && !t.core.done(); ++cycle) {
    const bool support = t.core.phase() == Phase::ProneSettle &&
                         t.core.phaseElapsedS() >= 1.5;
    t.tick(support);
    if (t.core.phase() != previous) {
      visited.push_back(t.core.phase());
      previous = t.core.phase();
    }
    if (isPronePositionPhase(t.core.phase()))
      validateBoundedPositionCommand(t);
    if (t.core.phase() == Phase::ProneEngageHold && !capturedEngagement) {
      for (std::size_t i = 0; i < kJointCount; ++i)
        engagement[i] = t.command.joint[i].q;
      capturedEngagement = true;
    }
    if (t.core.phase() == Phase::ProneRiseHold && !capturedRise) {
      for (std::size_t i = 0; i < kJointCount; ++i)
        rise[i] = t.command.joint[i].q;
      capturedRise = true;
    }
    if (t.core.phase() == Phase::ProneRelease) {
      require(t.command.joint[0].kp <= priorReleaseKp + 1.0e-6F,
              "release gain must be monotonic");
      priorReleaseKp = t.command.joint[0].kp;
      ++releaseSamples;
    }
    if (t.core.phase() == Phase::ProneFinalDamping) t.damping();
  }

  require(t.core.done() && !t.core.failed(), "nominal prone sequence failed");
  require(capturedEngagement && capturedRise, "position endpoints not observed");
  require(releaseSamples >= 1990, "four-second release was shortened");
  const std::vector<Phase> expected{
      Phase::ProneObserve, Phase::ProneEngage, Phase::ProneEngageHold,
      Phase::ProneRise, Phase::ProneRiseHold, Phase::ProneReturn,
      Phase::ProneSettle, Phase::ProneRelease, Phase::ProneFinalDamping,
      Phase::Complete};
  require(visited == expected, "nominal phase order changed");

  for (std::size_t leg = 0; leg < 4; ++leg) {
    const std::size_t b = leg * 3;
    const go1::JointAngles q0{engagement[b], engagement[b + 1],
                              engagement[b + 2]};
    const go1::JointAngles q1{rise[b], rise[b + 1], rise[b + 2]};
    const auto p0 = go1::Kinematics::forward(static_cast<go1::Leg>(leg), q0);
    const auto p1 = go1::Kinematics::forward(static_cast<go1::Leg>(leg), q1);
    require(std::fabs(p1.x - p0.x) < 1.0e-5 &&
                std::fabs(p1.y - p0.y) < 1.0e-5,
            "low-rise trajectory changed foot x/y");
    require(std::fabs((p1.z - p0.z) + kProneRiseM) < 1.0e-5,
            "low-rise trajectory is not exactly 5 mm in z");
  }
}

void supportInterlock() {
  Fixture premature;
  // A confirmation already asserted during the return is not a fresh
  // operator observation of the final prone contact.
  while (premature.core.phase() != Phase::ProneSettle)
    premature.tick(true);
  for (int i = 0; i < 2600; ++i) premature.tick(true);
  require(premature.core.phase() == Phase::ProneSupportHold,
          "pre-settle confirmation must not authorize gradual release");

  Fixture missing;
  missing.reach(Phase::ProneSettle);
  for (int i = 0; i < 2600; ++i) missing.tick(false);
  require(missing.core.phase() == Phase::ProneSupportHold &&
              !missing.core.done() && missing.core.failed(),
          "missing support must latch an impedance hold");
  require(missing.core.faultReason() == "prone_floor_support_not_confirmed" &&
              missing.command.joint[1].kp > 0.0F,
          "support timeout must preserve the prone target");

  Fixture withdrawal;
  withdrawal.reach(Phase::ProneSettle);
  for (int i = 0; i < 300; ++i) withdrawal.tick(true);
  withdrawal.tick(false, false);
  for (int i = 0; i < 300; ++i) withdrawal.tick(true);
  require(withdrawal.core.phase() == Phase::ProneSettle,
          "support withdrawal must restart the one-second dwell");
  for (int i = 0; i < 230 && withdrawal.core.phase() == Phase::ProneSettle; ++i)
    withdrawal.tick(true);
  require(withdrawal.core.phase() == Phase::ProneRelease,
          "continuous support should authorize gradual release");
}

void cancellationAndFaults() {
  Fixture effort;
  effort.reach(Phase::ProneRise);
  effort.f.joint[0].q = effort.command.joint[0].q + 0.09F;
  effort.f.joint[0].dq = -0.29F;
  effort.tick(false, true, true, 0, false, false);
  require(effort.core.phase() == Phase::ProneRise,
          "in-envelope effort-limit fixture unexpectedly aborted");
  validateBoundedPositionCommand(effort);
  require(effort.command.joint[0].kp < kProneKp,
          "effort limiter did not reduce Kp for combined position/damping error");

  Fixture cancel;
  cancel.reach(Phase::ProneRise);
  for (int i = 0; i < 200; ++i) cancel.tick();
  cancel.core.observeSignalCount(1, cancel.now, cancel.f);
  require(cancel.core.phase() == Phase::ProneReturn,
          "single Ctrl-C must return before releasing impedance");
  cancel.tick();
  validateBoundedPositionCommand(cancel);
  cancel.reach(Phase::ProneSupportHold);
  require(!cancel.core.failed() && !cancel.core.done(),
          "single Ctrl-C return must end in a non-fault latched hold");
  for (int i = 0; i < 600; ++i) cancel.tick(true);
  require(cancel.core.phase() == Phase::ProneSupportHold,
          "cancellation must not release merely because support is present");

  Fixture twice;
  twice.reach(Phase::ProneRise);
  twice.core.observeSignalCount(2, twice.now, twice.f);
  twice.tick();
  require(twice.core.phase() == Phase::PanicDamping,
          "double Ctrl-C must enter panic damping");
  twice.damping();

  Fixture feedback;
  feedback.reach(Phase::ProneRise);
  feedback.tick(false, false, false);
  require(feedback.core.phase() == Phase::PanicDamping,
          "feedback loss must enter panic damping");
  feedback.damping();

  Fixture watchdog;
  watchdog.reach(Phase::ProneRise);
  watchdog.tick(false, true, true, 0, true);
  require(watchdog.core.phase() == Phase::PanicDamping,
          "command watchdog must enter panic damping");
  watchdog.damping();

  Fixture remote;
  remote.reach(Phase::ProneRise);
  remote.f.remote.buttons = kRemoteL2Mask | kRemoteBMask;
  remote.tick();
  require(remote.core.phase() == Phase::PanicDamping,
          "L2+B must enter panic damping");
  remote.damping();

  Fixture invalidRemote;
  invalidRemote.reach(Phase::ProneRise);
  invalidRemote.f.remote.valid = false;
  invalidRemote.tick();
  require(invalidRemote.core.phase() == Phase::PanicDamping,
          "loss of the remote stop channel must enter panic damping");
  invalidRemote.damping();

  Fixture bounds;
  bounds.reach(Phase::ProneRise);
  bounds.f.joint[2].q = kJointMin[2] - kProneCalfFeedbackMargin - 0.01F;
  bounds.tick();
  require(bounds.core.phase() == Phase::PanicDamping,
          "feedback outside the prone-only calf margin must panic");
  bounds.damping();

  Fixture tracking;
  tracking.reach(Phase::ProneRise);
  tracking.f.joint[0].q += 0.11F;
  tracking.tick();
  require(tracking.core.phase() == Phase::PanicDamping,
          "prone tracking error must panic");
  tracking.damping();

  Fixture speed;
  speed.reach(Phase::ProneRise);
  speed.f.joint[0].dq = 0.31F;
  speed.tick();
  require(speed.core.phase() == Phase::PanicDamping,
          "prone joint speed limit must panic");
  speed.damping();
}

void hardwareLock() {
  char program[] = "test";
  char mode[] = "--mode";
  char value[] = "prone-low-rise";
  char confirmed[] = "--prone-confirmed";
  char *argv[] = {program, mode, value, confirmed};
  bool locked = false;
  try {
    parseOptions(4, argv);
  } catch (const std::runtime_error &error) {
    locked = std::string(error.what()).find("ground hardware modes are locked") !=
             std::string::npos;
  }
  require(locked, "prone-low-rise hardware must fail before UDP/ARM");
}
} // namespace

int main() {
  struct TestCase { const char *name; void (*run)(); };
  const TestCase cases[] = {
      {"bounded nominal sequence and exact 5 mm kinematics", nominal},
      {"independent continuous support interlock", supportInterlock},
      {"cancel, feedback, watchdog, remote and envelope faults",
       cancellationAndFaults},
      {"hardware CLI rejected before UDP", hardwareLock}};
  for (const auto &test : cases) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception &error) {
      std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
      return 1;
    }
  }
  std::cout << "Prone low-rise tests passed (synthetic feedback/support only).\n";
  return 0;
}
