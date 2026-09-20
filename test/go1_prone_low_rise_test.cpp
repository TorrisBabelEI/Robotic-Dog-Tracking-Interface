// Exercise the prone low-rise control core without linking Unitree or opening UDP.
#define GO1_CORE_TEST
#include "../src/go1_lowlevel_experiment.cpp"
#include "../src/go1_operator_support.hpp"

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
  ExperimentCore core;
  Feedback f;
  Feedback feedbackAtStep;
  Command command;
  int64_t now = 1000000000LL;

  Fixture(Options options = proneOptions()) : core(options) {
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

void recoveryAndRelease() {
  Fixture late;
  late.reach(Phase::ProneSupportHold);
  // A continuously asserted old input cannot authorize recovery.
  for (int i = 0; i < 600; ++i) late.tick(true);
  require(late.core.phase() == Phase::ProneSupportHold,
          "hold must reject an assertion without a new false observation");
  late.tick(false);
  for (int i = 0; i < 300; ++i) late.tick(true);
  late.tick(false);  // disconnect/lease expiry before the dwell completes
  for (int i = 0; i < 300; ++i) late.tick(true);
  require(late.core.phase() == Phase::ProneSupportHold,
          "interrupted recovery dwell must restart");
  for (int i = 0; i < 230; ++i) late.tick(true);
  require(late.core.phase() == Phase::ProneRelease,
          "late fresh confirmation must authorize release");
  float previous = late.command.joint[1].kp;
  for (int i = 0; i < 3000 && !late.core.done(); ++i) {
    late.tick(false); // confirmation expires; no need to hold the button
    require(late.command.joint[1].kp <= previous + 1e-6F,
            "pulse expiry must not re-engage stiffness during release");
    previous = late.command.joint[1].kp;
  }
  require(late.core.done() && late.core.failed() &&
              late.core.faultReason() == "prone_floor_support_not_confirmed",
          "recovered timeout must exit but preserve the failed-run record");
  late.damping();

  Fixture fault;
  fault.reach(Phase::ProneSettle);
  fault.tick(false);
  for (int i = 0; i < 600; ++i) fault.tick(true);
  require(fault.core.phase() == Phase::ProneRelease, "release not reached");
  fault.tick(false, false, false);
  require(fault.core.phase() == Phase::PanicDamping,
          "release authorization must not mask feedback loss");
  fault.damping();
}

void nonidealRelease() {
  Fixture t;
  t.reach(Phase::ProneSettle);
  t.tick(false);
  for (int i = 0; i < 510; ++i) t.tick(true);
  require(t.core.phase() == Phase::ProneRelease, "release not reached");
  t.f.joint[1].q = t.command.joint[1].q - 0.09F;
  t.f.joint[1].dq = -0.29F;
  t.tick(false, true, true, 0, false, false);
  const float limited = t.command.joint[1].kp;
  require(limited < 3.0F, "fixture did not constrain release stiffness");
  t.f.joint[1].q = t.command.joint[1].q;
  t.f.joint[1].dq = 0;
  t.tick(false);
  require(t.command.joint[1].kp <= limited,
          "improved tracking must not raise stiffness during release");
  t.f.rpy[0] = 0.06F;
  t.tick(false);
  require(t.core.phase() == Phase::PanicDamping &&
              t.core.faultReason() == "prone_release_attitude",
          "excess attitude during release must interrupt release");
  t.damping();
}

void transportToCore() {
  go1::OperatorSupportServer server;
  std::string error;
  require(server.start(0, &error), "offline loopback receiver failed to start");
  struct SocketGuard {
    int fd = -1;
    ~SocketGuard() { if (fd >= 0) ::close(fd); }
  } socket;
  socket.fd = ::socket(AF_INET, SOCK_STREAM, 0);
  require(socket.fd >= 0, "sender socket failed");
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(server.port());
  require(::connect(socket.fd, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) == 0, "sender connect failed");
  Fixture t;
  t.reach(Phase::ProneSupportHold);
  t.tick(false);
  // Real loopback transport carries the sender's H/sequence protocol into
  // the real core. Robot feedback and core time remain synthetic.
  for (int i = 0; i < 620; ++i) {
    if (i % 10 == 0) {
      const std::string frame = "H " + std::to_string(i / 10 + 1) + "\n";
      require(::send(socket.fd, frame.data(), frame.size(), 0) ==
                  static_cast<ssize_t>(frame.size()), "heartbeat send failed");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    t.tick(server.active(go1::operatorSupportNowNs()));
  }
  require(t.core.phase() == Phase::ProneRelease,
          "TCP confirmation did not authorize recovery release");
  ::close(socket.fd);
  socket.fd = -1;
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  require(!server.active(go1::operatorSupportNowNs()),
          "disconnect did not revoke transport confirmation");
  for (int i = 0; i < 3000 && !t.core.done(); ++i)
    t.tick(server.active(go1::operatorSupportNowNs()));
  require(t.core.done() && t.core.failed(),
          "authorized release must finish and preserve timeout failure");
  t.damping();
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
  cancel.reach(Phase::ProneSettle);
  cancel.tick(false);
  for (int i = 0; i < 600; ++i) cancel.tick(true);
  require(cancel.core.phase() == Phase::ProneRelease && !cancel.core.failed(),
          "cancelled return must allow a fresh confirmed release");
  for (int i = 0; i < 3000 && !cancel.core.done(); ++i) cancel.tick(false);
  require(cancel.core.done() && !cancel.core.failed(),
          "cancelled run must finish damping after confirmed release");
  cancel.damping();

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

void engagementOnly() {
  Options options = proneOptions(); options.mode = ExperimentMode::ProneEngagement;
  for (int scenario = 0; scenario < 3; ++scenario) {
    Fixture t(options);
    bool cancelled = false;
    bool sawSettle = false, sawRelease = false, sawHold = false;
    std::array<float, kJointCount> target;
    for (std::size_t i=0;i<kJointCount;++i)
      target[i] = clampValue(t.f.joint[i].q, kJointMin[i%3], kJointMax[i%3]);
    for (int cycle=0;cycle<15000 && !t.core.done();++cycle) {
      if (scenario == 1 && cycle == 1000) {
        t.core.observeSignalCount(1,t.now,t.f); cancelled=true;
      }
      const auto phase=t.core.phase();
      const bool support = (phase == Phase::ProneSettle && scenario != 2 &&
                            t.core.phaseElapsedS() > 1.0) ||
                           (phase == Phase::ProneSupportHold && t.core.phaseElapsedS() > 1.0);
      sawHold = sawHold || phase == Phase::ProneSupportHold;
      sawSettle = sawSettle || phase == Phase::ProneSettle;
      sawRelease = sawRelease || phase == Phase::ProneRelease;
      t.tick(support,true,true,0,false,false); // floor holds measured pose fixed
      require(t.core.phase()!=Phase::ProneRise && t.core.phase()!=Phase::ProneRiseHold,
              "engagement-only entered a rise phase");
      if (isPronePositionPhase(t.core.phase())) {
        for(std::size_t i=0;i<kJointCount;++i) {
          const auto &j=t.command.joint[i];
          require(std::fabs(j.q-target[i])<1e-6F && j.dq==0 && j.tauFf==0 && j.kp<=1,
                  "engagement-only changed target or exceeded gains");
          const double tau=j.kp*(j.q-t.f.joint[i].q)-j.kd*t.f.joint[i].dq;
          require(std::fabs(tau)<=0.100001, "engagement-only effort bound exceeded");
        }
      }
    }
    require(t.core.done() && sawSettle && sawRelease, "stationary engagement did not finish");
    require(t.core.failed()==(scenario==2), "incorrect engagement acceptance status");
    require(scenario!=1 || cancelled, "cancel scenario missing");
    require(scenario!=2 || sawHold, "timeout scenario missing");
    t.damping();
  }
  Fixture speed(options);
  speed.reach(Phase::ProneEngage);
  speed.f.joint[0].dq=0.081F;
  speed.tick(false,true,true,0,false,false);
  require(speed.core.phase()==Phase::PanicDamping,"engagement speed guard missing");
  speed.damping();
  char p[]="test", m[]="--mode", v[]="prone-engagement";
  char prone[]="--prone-confirmed", remote[]="--remote-confirmed";
  char *args[]={p,m,v,prone,remote};
  bool rejected=false;
  try { parseOptions(4,args); } catch (const std::runtime_error &) { rejected=true; }
  require(rejected,"engagement hardware must require remote confirmation");
  require(parseOptions(5,args).mode==ExperimentMode::ProneEngagement,
          "explicit engagement hardware options were not accepted");
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
      {"late confirmation recovery and pulse-expiry release", recoveryAndRelease},
      {"TCP receiver to recovery core and disconnect", transportToCore},
      {"nonideal release gain and attitude faults", nonidealRelease},
      {"cancel, feedback, watchdog, remote and envelope faults",
       cancellationAndFaults},
      {"engagement-only stationary, cancel, timeout and speed", engagementOnly},
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
