/**********************************************************************
 * Conservative Go1 low-level ground experiment runner.
 * Hardware access is Linux-only; every state and profile is dry-runnable.
 *********************************************************************/
#include "go1_kinematics.hpp"
#include "go1_log_file.hpp"
#if defined(GO1_WITH_POLICY)
#include "go1_policy_session.hpp"
#include "go1_command_owner.hpp"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(GO1_WITH_SDK)
#include "go1_operator_support.hpp"
#include "go1_sdk_transport.hpp"
#if defined(GO1_WITH_POLICY)
#include "go1_policy_runtime.hpp"
#endif
#include "unitree_legged_sdk/joystick.h"
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <boost/bind/bind.hpp>
#include <cerrno>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {
constexpr std::size_t kJointCount = 12;
constexpr double kControlDt = 0.002;
constexpr double kPi = 3.14159265358979323846;
constexpr float kPosStop = 2.146e9F;
constexpr float kVelStop = 16000.0F;
constexpr uint8_t kLowLevel = 0xFF;
constexpr uint8_t kServoMode = 0x0A;
constexpr uint8_t kDampingMode = 0x00;
constexpr uint8_t kOverheatMode = 0x08;
constexpr uint16_t kRemoteL2Mask = 1U << 5U;
constexpr uint16_t kRemoteBMask = 1U << 9U;
// Development-only prone path. These deliberately small values are exercised
// in dry-run tests before any hardware entry can be enabled.
constexpr double kProneObserveS = 1.0;
constexpr double kProneEngageS = 4.0;
constexpr double kProneMoveS = 4.0;
constexpr double kProneHoldS = 1.0;
constexpr double kProneReleaseS = 4.0;
constexpr double kProneSupportTimeoutS = 5.0;
constexpr double kProneRiseM = 0.005;
constexpr float kProneKp = 5.0F;
constexpr float kProneKd = 1.0F;
constexpr float kProneEffortLimitNm = 0.50F;

const std::array<const char *, kJointCount> kJointNames = {
    "FR_0", "FR_1", "FR_2", "FL_0", "FL_1", "FL_2",
    "RR_0", "RR_1", "RR_2", "RL_0", "RL_1", "RL_2"};
const std::array<float, 3> kJointMin = {-1.047F, -0.663F, -2.721F};
const std::array<float, 3> kJointMax = {+1.047F, +2.966F, -0.837F};
// Factory lie-down feedback on the tested Go1 folds all four calves slightly
// beyond the SDK's position-command limit. This margin is only accepted while
// the robot is confirmed prone and receiving a position-free damping command.
constexpr float kProneCalfFeedbackMargin = 0.10F;
volatile std::sig_atomic_t gSignalCount = 0;

int64_t steadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <typename T> T clampValue(T value, T low, T high) {
  return std::min(std::max(value, low), high);
}
double smoothStep5(double x) {
  x = clampValue(x, 0.0, 1.0);
  return x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
}
double smoothStep5Derivative(double x) {
  x = clampValue(x, 0.0, 1.0);
  return 30.0 * x * x - 60.0 * x * x * x + 30.0 * x * x * x * x;
}

// The SDK returns bytes sent, whereas the core consumes 0=success/-1=failure.
int sdkSendStatus(int bytesSent, int expectedBytes) {
  return expectedBytes > 0 && bytesSent == expectedBytes ? 0 : -1;
}

enum class ExperimentMode {
  RemotePreflight, GroundHandover, Preflight, TorqueSine, TorqueSineGround,
  Squat, LegLift, LegLiftSequence, ProneLowRise, ProneEngagement, WalkingPolicy
};
enum class Phase {
  WalkingPolicy, Disarmed, RemotePreflight, Precheck, CapturePose, Baseline, Hold,
  GroundHandover, TorqueExcite, Squat, WeightShift, Lift, AirHold, Lower,
  ContactVerify, Recenter, InterLeg, Return, SafeHold, PanicDamping, Complete,
  ExitLower, ExitVerify, ExitHold, ExitDamping,
  ProneObserve, ProneEngage, ProneEngageHold, ProneRise, ProneRiseHold,
  ProneReturn, ProneSettle, ProneSupportHold, ProneRelease,
  ProneFinalDamping
};

const char *phaseName(Phase phase) {
  switch (phase) {
  case Phase::WalkingPolicy: return "WALKING_POLICY";
  case Phase::Disarmed: return "DISARMED";
  case Phase::RemotePreflight: return "REMOTE_PREFLIGHT";
  case Phase::Precheck: return "PRECHECK";
  case Phase::CapturePose: return "CAPTURE_POSE";
  case Phase::Baseline: return "BASELINE";
  case Phase::Hold: return "HOLD";
  case Phase::GroundHandover: return "GROUND_HANDOVER";
  case Phase::TorqueExcite: return "TORQUE_EXCITE";
  case Phase::Squat: return "SQUAT";
  case Phase::WeightShift: return "WEIGHT_SHIFT";
  case Phase::Lift: return "LIFT";
  case Phase::AirHold: return "AIR_HOLD";
  case Phase::Lower: return "LOWER";
  case Phase::ContactVerify: return "CONTACT_VERIFY";
  case Phase::Recenter: return "RECENTER";
  case Phase::InterLeg: return "INTER_LEG";
  case Phase::Return: return "RETURN";
  case Phase::SafeHold: return "SAFE_HOLD";
  case Phase::PanicDamping: return "PANIC_DAMPING";
  case Phase::Complete: return "COMPLETE";
  case Phase::ExitLower: return "EXIT_LOWER";
  case Phase::ExitVerify: return "EXIT_VERIFY_SUPPORT";
  case Phase::ExitHold: return "EXIT_HOLD";
  case Phase::ExitDamping: return "EXIT_DAMPING";
  case Phase::ProneObserve: return "PRONE_OBSERVE";
  case Phase::ProneEngage: return "PRONE_ENGAGE";
  case Phase::ProneEngageHold: return "PRONE_ENGAGE_HOLD";
  case Phase::ProneRise: return "PRONE_RISE";
  case Phase::ProneRiseHold: return "PRONE_RISE_HOLD";
  case Phase::ProneReturn: return "PRONE_RETURN";
  case Phase::ProneSettle: return "PRONE_SETTLE";
  case Phase::ProneSupportHold: return "PRONE_SUPPORT_HOLD";
  case Phase::ProneRelease: return "PRONE_RELEASE";
  case Phase::ProneFinalDamping: return "PRONE_FINAL_DAMPING";
  }
  return "UNKNOWN";
}
const char *modeName(ExperimentMode mode) {
  switch (mode) {
  case ExperimentMode::RemotePreflight: return "remote-preflight";
  case ExperimentMode::GroundHandover: return "ground-handover";
  case ExperimentMode::WalkingPolicy: return "walking-policy";
  case ExperimentMode::Preflight: return "preflight";
  case ExperimentMode::TorqueSine: return "torque-sine";
  case ExperimentMode::TorqueSineGround: return "torque-sine-ground";
  case ExperimentMode::Squat: return "squat";
  case ExperimentMode::LegLift: return "leg-lift";
  case ExperimentMode::LegLiftSequence: return "leg-lift-sequence";
  case ExperimentMode::ProneLowRise: return "prone-low-rise";
  case ExperimentMode::ProneEngagement: return "prone-engagement";
  }
  return "unknown";
}
bool isLegMode(ExperimentMode mode) {
  return mode == ExperimentMode::LegLift || mode == ExperimentMode::LegLiftSequence;
}
bool isProneMode(ExperimentMode mode) {
  return mode == ExperimentMode::ProneLowRise || mode == ExperimentMode::ProneEngagement;
}
bool isGroundMode(ExperimentMode mode) {
  return mode == ExperimentMode::WalkingPolicy || mode == ExperimentMode::GroundHandover ||
         mode == ExperimentMode::TorqueSineGround ||
         mode == ExperimentMode::Squat || isLegMode(mode) ||
         isProneMode(mode);
}
bool requiresStandingCapture(ExperimentMode mode) {
  return isGroundMode(mode) && !isProneMode(mode);
}
bool isPronePositionPhase(Phase phase) {
  return phase == Phase::ProneEngage || phase == Phase::ProneEngageHold ||
         phase == Phase::ProneRise || phase == Phase::ProneRiseHold ||
         phase == Phase::ProneReturn || phase == Phase::ProneSettle ||
         phase == Phase::ProneSupportHold || phase == Phase::ProneRelease;
}
bool isLegMotionPhase(Phase phase) {
  return phase == Phase::WeightShift || phase == Phase::Lift ||
         phase == Phase::AirHold || phase == Phase::Lower ||
         phase == Phase::ContactVerify || phase == Phase::Recenter;
}

struct Options {
  ExperimentMode mode = ExperimentMode::Preflight;
  int joint = 1;
  double amplitudeNm = 0.2;
  double frequencyHz = 0.5;
  double durationS = 10.0;
  double liftHeightM = 0.02;
  double tauOverlayNm = 0.10;
  double tauOverlayHz = 0.5;
  bool legAuto = true;
  go1::Leg requestedLeg = go1::Leg::FR;
  std::string logPath = "go1_lowlevel_log.csv";
  std::string targetIp = "192.168.123.10";
  uint16_t localPort = 8090;
  uint16_t targetPort = 8007;
  std::string highTargetIp = "192.168.123.161";
  uint16_t highLocalPort = 8091;
  uint16_t highTargetPort = 8082;
  bool supportConfirmed = false;
  bool groundConfirmed = false;
  bool remoteConfirmed = false;
  bool proneConfirmed = false;
  bool dryRun = false;
  // Development endpoint only. A nominal folded pose is not evidence of
  // belly contact; hardware remains locked pending calibration and review.
  bool dryRunNormalExit = false;
  double injectSoftStopS = -1.0;
  double injectPanicS = -1.0;
  double injectDoubleCtrlCS = -1.0;
  double injectWatchdogS = -1.0;
};
struct RemoteFeedback {
  bool valid = false;
  uint8_t head0 = 0, head1 = 0;
  uint16_t buttons = 0;
  float lx = 0.0F, ly = 0.0F, rx = 0.0F, ry = 0.0F, l2 = 0.0F;
};
struct JointFeedback {
  uint8_t mode = kDampingMode;
  float q = 0.0F, dq = 0.0F, tauEst = 0.0F;
  int8_t temperature = 0;
};
struct Feedback {
  uint8_t levelFlag = 0;
  uint32_t tickMs = 0;
  std::array<JointFeedback, kJointCount> joint;
  std::array<float, 3> rpy{{0, 0, 0}}, gyro{{0, 0, 0}}, accel{{0, 0, 0}};
  std::array<int16_t, 4> footForce{{0, 0, 0, 0}};
  RemoteFeedback remote;
};
struct JointCommand {
  uint8_t mode = kDampingMode;
  float q = kPosStop, dq = 0.0F, kp = 0.0F, kd = 1.0F, tauFf = 0.0F;
};
struct Command { std::array<JointCommand, kJointCount> joint; };
struct SupportData {
  bool valid = false;
  go1::Vec2 cop;
  double totalForce = 0.0;
  std::array<double, 4> margin{{NAN, NAN, NAN, NAN}};
  std::array<go1::Vec3, 4> feet;
};
struct LogSample {
  int64_t hostNs = 0;
  uint32_t tickMs = 0;
  Phase phase = Phase::Disarmed;
  bool recvFresh = false, watchdogActive = false;
  int recvResult = 0, sendResult = 0, activeLeg = -1;
  double loopDtUs = 0.0;
  std::string abortReason, stopSource;
  int64_t stopRequestNs = 0, dampingCommandNs = 0;
  bool exitSupportConfirmed = false;
  double exitStableS = 0;
  Feedback feedback;
  Command command;
  std::array<float, kJointCount> tauTotal;
  SupportData support;
  std::array<double, 4> forceBaseline{{NAN, NAN, NAN, NAN}};
  std::array<double, 4> forceMad{{NAN, NAN, NAN, NAN}};
  std::array<go1::Vec3, 4> footTarget;
};
double median(std::vector<double> values) {
  if (values.empty()) return NAN;
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle]
                           : 0.5 * (values[middle - 1] + values[middle]);
}

// SDK-independent accumulator used by the real high-level capture path.
// receiveSequence is the transport receive counter, NOT a robot source tick.
class StandingPoseCapture {
public:
  bool observe(const Feedback &f, uint64_t receiveSequence, int64_t nowNs,
               bool received, std::array<float, kJointCount> *pose) {
    if (count_ && (nowNs <= lastNs_ || nowNs - lastNs_ > 20000000LL)) resetWindow();
    if (!received || receiveSequence == sequence_) return false;
    if (receiveSequence < sequence_) {
      sequence_ = receiveSequence;
      resetWindow();
      return false;
    }
    sequence_ = receiveSequence;
    bool valid = nowNs > 0 && f.levelFlag == 0xEE;
    for (float angle : f.rpy) valid = valid && std::isfinite(angle);
    valid = valid && std::fabs(f.rpy[0]) <= 0.5F && std::fabs(f.rpy[1]) <= 0.5F;
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const auto &j = f.joint[i];
      valid = valid && j.mode != kOverheatMode && j.temperature < 70 &&
              std::isfinite(j.q) && std::isfinite(j.dq) &&
              j.q >= kJointMin[i % 3] && j.q <= kJointMax[i % 3] &&
              std::fabs(j.dq) <= 0.05F;
      if (count_) valid = valid && std::fabs(j.q - reference_[i]) <= 0.02F;
    }
    if (!valid) { resetWindow(); return false; }
    if (!count_) {
      firstNs_ = nowNs;
      for (std::size_t i = 0; i < kJointCount; ++i) reference_[i] = f.joint[i].q;
    }
    lastNs_ = nowNs;
    ++count_;
    for (std::size_t i = 0; i < kJointCount; ++i) sum_[i] += f.joint[i].q;
    if (count_ < 100 || nowNs - firstNs_ < 200000000LL || !pose) return false;
    for (std::size_t i = 0; i < kJointCount; ++i)
      (*pose)[i] = static_cast<float>(sum_[i] / count_);
    return true;
  }
  int count() const { return count_; }
private:
  void resetWindow() { count_ = 0; firstNs_ = lastNs_ = 0; sum_.fill(0); }
  uint64_t sequence_ = 0;
  int count_ = 0;
  int64_t firstNs_ = 0, lastNs_ = 0;
  std::array<float, kJointCount> reference_{{0}};
  std::array<double, kJointCount> sum_{{0}};
};

class ExperimentCore {
public:
  explicit ExperimentCore(const Options &options) : options_(options) {
    logs_.reserve(static_cast<std::size_t>(100.0 / kControlDt));
    for (auto &samples : forceSamples_) samples.reserve(1100);
  }
#if defined(GO1_WITH_POLICY)
  go1::PolicySession& policySession() { return policySession_; }
  void setPolicyEvidence(const go1::PolicyEvidence& evidence) {
    policyEvidence_=evidence;
    // Operator stop must apply during startup/hold, before policy engagement.
    if(options_.mode==ExperimentMode::WalkingPolicy && evidence.stop && !failed_)
      enterPanic("policy_operator_stop",evidence.nowNs);
  }
#endif
  Phase phase() const { return phase_; }
  double phaseElapsedS() const { return phaseElapsedS_; }
  bool done() const { return doneFlag_.load(); }
  bool safeHoldReached() const { return safeHoldReached_.load(); }
  bool panicReached() const { return panicReached_.load(); }
  bool failed() const { return failed_; }
  const std::string &faultReason() const { return faultReason_; }
  bool proneContactConfirmationAllowed() const {
    return (phase_ == Phase::ProneSettle || phase_ == Phase::ProneSupportHold) &&
        (stopSource_.empty() || stopSource_ == "ctrl_c" ||
         stopSource_ == "prone_floor_support_not_confirmed");
  }
  std::size_t logCount() const { return logs_.size(); }
  int activeLegIndex() const {
    return activeLegValid_ ? go1::legIndex(activeLeg_) : -1;
  }
  Command emergencyDampingCommand() const { return dampingCommand(); }
  void seedGroundPose(const std::array<float, kJointCount> &pose) {
    for (std::size_t i = 0; i < kJointCount; ++i)
      if (!std::isfinite(pose[i]) || pose[i] < kJointMin[i % 3] ||
          pose[i] > kJointMax[i % 3])
        throw std::runtime_error("invalid_ground_seed_pose");
    precheckQ_ = pose;
    precheckPoseCaptured_ = true;
  }
#if defined(GO1_WITH_POLICY)
  // Offline-only handover experiment: preserve the previous actuator torque
  // using an equivalent position offset at the unchanged parent gains.
  // This is deliberately unavailable to hardware and after startup begins.
  bool seedOfflinePolicySupport(const std::array<float, kJointCount>& torque,
                                const std::array<float, kJointCount>& velocity) {
    if (!options_.dryRun || options_.mode != ExperimentMode::WalkingPolicy ||
        phase_ != Phase::Disarmed || !precheckPoseCaptured_) return false;
    std::array<float, kJointCount> bias{},feedforward{};
    for (std::size_t i=0;i<kJointCount;++i) {
      feedforward[i]=clampValue(torque[i],-1.0F,1.0F);
      bias[i]=(torque[i]+1.5F*velocity[i]-feedforward[i])/20.0F;
      const float target=precheckQ_[i]+bias[i];
      if (!std::isfinite(bias[i]) || std::fabs(velocity[i])>0.05F ||
          std::fabs(bias[i])>0.3F || target<kJointMin[i%3] || target>kJointMax[i%3])
        return false;
    }
    offlineSupportBias_=bias;offlineSupportFeedforward_=feedforward;
    return true;
  }
#endif
  Command seededGroundHoldCommand() const {
    return precheckPoseCaptured_ ? holdCommand(precheckQ_) : dampingCommand();
  }
  std::string feedbackReadinessIssue(const Feedback &f, bool hasState) const {
    if (!hasState) return "no_state";
    if (f.levelFlag != kLowLevel) return "level_flag_not_lowlevel";
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const auto &joint = f.joint[i];
      if (!std::isfinite(joint.q) || !std::isfinite(joint.dq) ||
          !std::isfinite(joint.tauEst))
        return std::string("nonfinite_") + kJointNames[i];
      float feedbackMin = kJointMin[i % 3];
      if ((options_.mode == ExperimentMode::RemotePreflight ||
           isProneMode(options_.mode)) && i % 3 == 2)
        feedbackMin -= kProneCalfFeedbackMargin;
      if (joint.q < feedbackMin || joint.q > kJointMax[i % 3])
        return std::string("joint_limit_") + kJointNames[i];
      if (joint.mode == kOverheatMode)
        return std::string("overheat_mode_") + kJointNames[i];
      if (joint.temperature >= 70)
        return std::string("temperature_") + kJointNames[i];
      if (isGroundMode(options_.mode) && std::fabs(joint.dq) > 1.0)
        return std::string("ground_speed_") + kJointNames[i];
    }
    if (isLegMode(options_.mode) &&
        ((!options_.dryRun && !options_.remoteConfirmed) || !f.remote.valid))
      return "remote_not_valid_for_leg_mode";
    if (isProneMode(options_.mode) && !f.remote.valid)
      return "remote_not_valid_for_prone_mode";
    return "";
  }

  void observeSignalCount(int count, int64_t nowNs, const Feedback &feedback) {
    while (handledSignalCount_ < count) {
      ++handledSignalCount_;
      if (phase_ == Phase::PanicDamping) {
        requestPanicExit(nowNs);
      } else if (lastSignalNs_ > 0 && nowNs - lastSignalNs_ <= 1000000000LL) {
        enterPanic("double_ctrl_c", nowNs);
      } else {
        lastSignalNs_ = nowNs;
        requestSoftStop("ctrl_c", feedback, nowNs);
      }
    }
  }
  void forceHardFault(const std::string &reason, int64_t nowNs) {
    enterPanic(reason, nowNs);
  }
  void amendLastLoggedCommand(const Command &command, bool watchdogActive, int rawSendResult) {
    if (logs_.empty()) return;
    auto &sample = logs_.back();
    sample.phase = phase_;
    sample.abortReason = faultReason_;
    sample.stopSource = stopSource_;
    sample.stopRequestNs = stopRequestNs_;
    sample.dampingCommandNs = dampingCommandNs_;
    sample.command = command;
    sample.sendResult = rawSendResult;
    sample.watchdogActive = watchdogActive;
    fillTauTotal(sample);
  }

  Command step(const Feedback &feedback, bool hasState, bool recvFresh,
               bool recvAlive, int recvResult, int sendResult, double loopDtUs,
               int64_t hostNs, bool watchdogActive,
               bool floorSupportObserved = false) {
    // This is a per-cycle independent observation, never a startup permission
    // or a conclusion inferred from the commanded joint pose/foot unloading.
    exitSupportConfirmed_ = false;
    if (phase_ == Phase::Disarmed) {
      if (options_.mode == ExperimentMode::RemotePreflight)
        transition(Phase::RemotePreflight);
      else if (isProneMode(options_.mode))
        transition(Phase::ProneObserve);
      else
        transition(Phase::Precheck);
    }
    const bool tickValid = validateFreshTick(feedback, recvFresh);
    if (recvFresh && !tickValid && phase_ != Phase::RemotePreflight &&
        phase_ != Phase::Precheck && phase_ != Phase::ProneObserve &&
        phase_ != Phase::PanicDamping &&
        phase_ != Phase::Complete)
      enterPanic("state_tick_not_monotonic_or_gap", hostNs);
    if (watchdogActive && phase_ != Phase::RemotePreflight &&
        phase_ != Phase::Precheck && phase_ != Phase::ProneObserve &&
        phase_ != Phase::PanicDamping &&
        phase_ != Phase::Complete)
      enterPanic("command_watchdog_over_20ms", hostNs);
    observeRemoteStop(feedback, recvFresh, hostNs);
    if (isGroundMode(options_.mode) && precheckPoseCaptured_ &&
        (phase_ == Phase::Precheck || phase_ == Phase::CapturePose)) {
      if (groundEntryStartNs_ == 0) groundEntryStartNs_ = hostNs;
      if (watchdogActive || sendResult != 0)
        enterPanic("ground_entry_command_failure", hostNs);
      else if ((!hasState || !recvAlive) &&
               hostNs - groundEntryStartNs_ > 20000000LL)
        enterPanic("ground_entry_feedback_timeout", hostNs);
      else if (hasState && recvFresh) {
        bool valid = recvAlive && tickValid && feedbackReady(feedback, true);
        for (std::size_t i = 0; i < kJointCount; ++i)
          valid = valid && std::fabs(feedback.joint[i].q - precheckQ_[i]) <= 0.05F &&
                  std::fabs(feedback.joint[i].dq) <= 0.05F;
        for (float angle : feedback.rpy) valid = valid && std::isfinite(angle);
        valid = valid && std::fabs(feedback.rpy[0]) <= 0.5F &&
                        std::fabs(feedback.rpy[1]) <= 0.5F;
        if (!valid) enterPanic("ground_entry_pose_mismatch_or_invalid", hostNs);
      }
    }
    if (phase_ != Phase::RemotePreflight && phase_ != Phase::ProneObserve)
      applyFeedbackSafety(feedback, hasState, recvFresh, recvAlive, hostNs);
    updateSupport(feedback);

    Command command = dampingCommand();
    switch (phase_) {
    case Phase::RemotePreflight:
      if (feedbackReady(feedback, hasState) && recvAlive && tickValid) {
        if (recvFresh) {
          ++validPacketStreak_;
          if (feedback.remote.valid) {
            remoteEverValid_ = true;
            const uint16_t required = kRemoteL2Mask | kRemoteBMask;
            if ((feedback.remote.buttons & required) == required)
              remoteChordSeen_ = true;
          }
        }
      } else {
        validPacketStreak_ = 0;
      }
      if (validPacketStreak_ >= 25) remoteCommunicationReady_ = true;
      if (remoteCommunicationReady_ &&
          (!recvAlive || (recvFresh && !tickValid) ||
           !feedbackReady(feedback, hasState))) {
        enterPanic(!recvAlive
                       ? "remote_preflight_feedback_gap_over_20ms"
                       : (recvFresh && !tickValid
                              ? "remote_preflight_tick_invalid"
                              : "remote_preflight_invalid_lowstate"),
                   hostNs);
        requestPanicExit(hostNs);
      } else if (!remoteCommunicationReady_ && phaseElapsedS_ >= 5.0) {
        const std::string issue = feedbackReadinessIssue(feedback, hasState);
        enterPanic(std::string("remote_preflight_") +
                       (issue.empty() ? "insufficient_packet_streak" : issue),
                   hostNs);
        requestPanicExit(hostNs);
      } else if (phaseElapsedS_ >= options_.durationS) {
        if (!remoteEverValid_ || !remoteChordSeen_) {
          enterPanic(!remoteEverValid_ ? "remote_data_never_valid"
                                      : "remote_l2_b_not_seen",
                     hostNs);
          requestPanicExit(hostNs);
        } else {
          transition(Phase::Complete);
        }
      }
      break;
    case Phase::ProneObserve:
      command = dampingCommand();
      if (observePronePose(feedback, hasState, recvFresh, recvAlive,
                           tickValid, hostNs)) {
        if (!prepareProneTargets()) {
          enterPanic("prone_rise_target_kinematics", hostNs);
        } else {
          transition(Phase::ProneEngage);
          command = pronePositionCommand(
              proneEngagementQ_, zeroJointVelocity(), 0.0F, feedback);
        }
      } else if (phaseElapsedS_ >= 5.0) {
        const std::string issue = feedbackReadinessIssue(feedback, hasState);
        enterPanic(std::string("prone_observe_") +
                       (issue.empty() ? "unstable_or_insufficient_feedback"
                                      : issue),
                   hostNs);
      }
      break;
    case Phase::ProneEngage: {
      const double x = phaseElapsedS_ / kProneEngageS;
      command = pronePositionCommand(
          proneEngagementQ_, zeroJointVelocity(),
          static_cast<float>(proneMaxKp() * smoothStep5(x)), feedback);
      if (phaseElapsedS_ >= kProneEngageS)
        transition(Phase::ProneEngageHold);
      break;
    }
    case Phase::ProneEngageHold:
      command = pronePositionCommand(proneEngagementQ_, zeroJointVelocity(),
                                     proneMaxKp(), feedback);
      if (updateProneStable(feedback, proneEngagementQ_, recvFresh, recvAlive)) {
        transition(options_.mode == ExperimentMode::ProneEngagement
                       ? Phase::ProneSettle : Phase::ProneRise);
      } else if (phaseElapsedS_ >= 3.0) {
        requestProneStop("prone_engagement_not_settled", feedback, hostNs);
      }
      break;
    case Phase::ProneRise: {
      const double x = phaseElapsedS_ / kProneMoveS;
      const double p = smoothStep5(x);
      const double pd = smoothStep5Derivative(x) / kProneMoveS;
      std::array<float, kJointCount> target, velocity;
      for (std::size_t i = 0; i < kJointCount; ++i) {
        const double delta = proneRiseQ_[i] - proneEngagementQ_[i];
        target[i] = proneEngagementQ_[i] + static_cast<float>(delta * p);
        velocity[i] = static_cast<float>(delta * pd);
      }
      command = pronePositionCommand(target, velocity, proneMaxKp(), feedback);
      if (phaseElapsedS_ >= kProneMoveS)
        transition(Phase::ProneRiseHold);
      break;
    }
    case Phase::ProneRiseHold:
      command = pronePositionCommand(proneRiseQ_, zeroJointVelocity(),
                                     proneMaxKp(), feedback);
      if (updateProneStable(feedback, proneRiseQ_, recvFresh, recvAlive)) {
        beginProneReturn();
      } else if (phaseElapsedS_ >= 3.0) {
        requestProneStop("prone_rise_not_settled", feedback, hostNs);
      }
      break;
    case Phase::ProneReturn: {
      const double x = phaseElapsedS_ / kProneMoveS;
      const double p = smoothStep5(x);
      const double pd = smoothStep5Derivative(x) / kProneMoveS;
      std::array<float, kJointCount> target, velocity;
      for (std::size_t i = 0; i < kJointCount; ++i) {
        const double delta = proneEngagementQ_[i] - proneReturnStartQ_[i];
        target[i] = proneReturnStartQ_[i] + static_cast<float>(delta * p);
        velocity[i] = static_cast<float>(delta * pd);
      }
      command = pronePositionCommand(target, velocity, proneMaxKp(), feedback);
      if (phaseElapsedS_ >= kProneMoveS)
        transition(Phase::ProneSettle);
      break;
    }
    case Phase::ProneSettle:
    case Phase::ProneSupportHold: {
      command = pronePositionCommand(proneEngagementQ_, zeroJointVelocity(),
                                     proneMaxKp(), feedback);
      // Only normal cancellation and a missed operator confirmation may
      // recover through a new contact assertion. Other faults stay held.
      if (!stopSource_.empty() && stopSource_ != "ctrl_c" &&
          stopSource_ != "prone_floor_support_not_confirmed") {
        if (phase_ != Phase::ProneSupportHold) holdProne(stopSource_, hostNs);
        break;
      }
      const bool settled = proneTargetSettled(
          feedback, proneEngagementQ_, recvFresh, recvAlive);
      // A confirmation that started before PRONE_SETTLE may be stale by the
      // time the trunk reaches the floor. Require a false observation in this
      // phase before accepting a new assertion, not merely a still-true flag.
      if (!floorSupportObserved) {
        proneSupportStableStartNs_ = 0;
        proneSupportReadyForNewAssert_ = true;
      }
      const bool supportEligible =
          proneSupportReadyForNewAssert_ && floorSupportObserved;
      if (recvFresh) {
        if (!settled || !supportEligible)
          proneSupportStableStartNs_ = 0;
        else if (proneSupportStableStartNs_ == 0)
          proneSupportStableStartNs_ = hostNs;
      }
      const double supportedS = proneSupportStableStartNs_ > 0
          ? (hostNs - proneSupportStableStartNs_) / 1.0e9 : 0.0;
      exitStableS_ = supportedS;
      if (settled && supportEligible && supportedS >= kProneHoldS) {
        exitSupportConfirmed_ = true;
        transition(Phase::ProneRelease);
      } else if (phase_ == Phase::ProneSettle &&
                 phaseElapsedS_ >= kProneSupportTimeoutS) {
        holdProne("prone_floor_support_not_confirmed", hostNs);
        command = pronePositionCommand(proneEngagementQ_, zeroJointVelocity(),
                                       proneMaxKp(), feedback);
      }
      break;
    }
    case Phase::ProneRelease: {
      // The fresh one-second dwell authorizes this whole release. The GUI's
      // short pulse is an acknowledgement, not a continuous contact sensor.
      // Expiry must not re-engage stiffness; normal feedback/remote fault
      // checks still run before this state on every cycle.
      const double remaining = 1.0 - smoothStep5(
          phaseElapsedS_ / kProneReleaseS);
      command = pronePositionCommand(
          proneEngagementQ_, zeroJointVelocity(),
          static_cast<float>(proneMaxKp() * remaining), feedback);
      // A reduced tracking error can relax the effort limiter. Never let
      // that raise an already-released gain on a later cycle.
      for (std::size_t i = 0; i < kJointCount; ++i)
        command.joint[i].kp = std::min(command.joint[i].kp,
                                      lastCommand_.joint[i].kp);
      if (phaseElapsedS_ >= kProneReleaseS) {
        dampingCommandNs_ = hostNs;
        transition(Phase::ProneFinalDamping);
        command = dampingCommand();
      }
      break;
    }
    case Phase::ProneFinalDamping:
      command = dampingCommand();
      if (sendResult != 0) {
        enterPanic("prone_final_damping_send_failed", hostNs);
      } else if (phaseElapsedS_ >= kProneHoldS) {
        transition(Phase::Complete);
      }
      break;
    case Phase::Precheck:
      command = precheckCommand(feedback, hasState, recvAlive);
      if (feedbackReady(feedback, hasState) && recvAlive && tickValid) {
        if (recvFresh) ++validPacketStreak_;
      } else validPacketStreak_ = 0;
      if (validPacketStreak_ >= 250) transition(Phase::CapturePose);
      else if (phaseElapsedS_ >= 5.0)
        enterPanic("precheck_timeout_or_not_lowlevel", hostNs);
      break;
    case Phase::CapturePose:
      if (!recvFresh) {
        command = precheckCommand(feedback, hasState, recvAlive);
      } else if (!capturePose(feedback)) {
        enterPanic("capture_pose_kinematics", hostNs);
      } else {
        command = holdCommand(initialQ_);
        transition(isLegMode(options_.mode) ? Phase::Baseline : Phase::Hold);
      }
      break;
    case Phase::Baseline:
      command = holdCommand(initialQ_);
      collectForceBaseline(feedback, recvFresh);
      if (phaseElapsedS_ >= 2.0) {
        if (!finishForceBaseline())
          requestSoftStop("foot_force_baseline_invalid", feedback, hostNs);
        else if (!selectAndPrepareLeg())
          requestSoftStop("no_safe_or_reachable_leg", feedback, hostNs);
        else transition(Phase::Hold);
      }
      break;
    case Phase::Hold:
      command = holdCommand(initialQ_);
      if (phaseElapsedS_ >= (isLegMode(options_.mode) ? 1.0 : 2.0)) {
        if (options_.mode == ExperimentMode::WalkingPolicy)
          transition(Phase::WalkingPolicy);
        else if (options_.mode == ExperimentMode::GroundHandover)
          transition(Phase::GroundHandover);
        else if (options_.mode == ExperimentMode::Squat)
          transition(Phase::Squat);
        else if (isLegMode(options_.mode)) transition(Phase::WeightShift);
        else transition(Phase::TorqueExcite);
      }
      break;
#if defined(GO1_WITH_POLICY)
    case Phase::WalkingPolicy: {
      policyEvidence_.nowNs=hostNs;
      policySession_.tick(policyEvidence_);
      // Bound policy authority from its first enabled session, including blend.
      // Do not reset this deadline on a late reply or an enable re-press.
      if (policySession_.running()) {
        if (!policyTrialStartNs_) policyTrialStartNs_ = hostNs;
        if (hostNs - policyTrialStartNs_ >=
            static_cast<int64_t>(options_.durationS * 1e9)) {
          enterPanic("policy_trial_duration_elapsed", hostNs);
          command = dampingCommand();
          break;
        }
      }
      if (options_.dryRun && policySession_.running()) {
        if (!offlinePolicyStartNs_) offlinePolicyStartNs_=hostNs;
        offlineSupportBlend_=std::max(0.0,1.0-(hostNs-offlinePolicyStartNs_)/2e9);
      }
      if(!policySession_.fault().empty()) {
        enterPanic(policySession_.fault(),hostNs);
        command=dampingCommand();
      } else {
        const auto target=policySession_.target(hostNs);
        std::array<float,kJointCount> q;
        for(std::size_t i=0;i<kJointCount;++i)q[i]=static_cast<float>(target[i]);
        command=holdCommand(q);
      }
      break;
    }
#else
    case Phase::WalkingPolicy:
      enterPanic("policy_development_disabled", hostNs);
      command = dampingCommand();
      break;
#endif
    case Phase::GroundHandover:
      command = holdCommand(initialQ_);
      if (phaseElapsedS_ >= 10.0) beginReturn(feedback);
      break;
    case Phase::TorqueExcite:
      command = torqueCommand(feedback);
      if (phaseElapsedS_ >= options_.durationS) beginReturn(feedback);
      break;
    case Phase::Squat:
      command = squatCommand();
      if (phaseElapsedS_ >= 8.0) beginReturn(feedback);
      break;
    case Phase::WeightShift: command = weightShiftCommand(feedback, hostNs); break;
    case Phase::Lift: command = liftCommand(feedback, hostNs); break;
    case Phase::AirHold: command = airHoldCommand(feedback, hostNs); break;
    case Phase::Lower: command = lowerCommand(feedback, hostNs); break;
    case Phase::ContactVerify: command = contactCommand(feedback, hostNs); break;
    case Phase::Recenter: command = recenterCommand(feedback, hostNs); break;
    case Phase::InterLeg:
      command = holdCommand(initialQ_);
      if (phaseElapsedS_ >= 3.0) transition(Phase::Baseline);
      break;
    case Phase::Return:
      command = returnCommand();
      if (phaseElapsedS_ >= 2.0) {
        if (options_.dryRunNormalExit && stopSource_.empty()) {
          for (std::size_t i = 0; i < kJointCount; ++i)
            exitStartQ_[i] = feedback.joint[i].q;
          exitStartNs_ = hostNs;
          transition(Phase::ExitLower);
        } else transition(Phase::SafeHold);
      }
      break;
    case Phase::ExitLower:
      command = exitLowerCommand(hostNs);
      if (hostNs - exitStartNs_ >= 8000000000LL) {
        exitVerifyNs_ = hostNs;
        exitStableStartNs_ = 0;
        transition(Phase::ExitVerify);
      }
      break;
    case Phase::ExitVerify: {
      command = holdCommand(exitTarget());
      bool settled = recvFresh && recvAlive && feedback.levelFlag == kLowLevel;
      for (std::size_t i = 0; i < kJointCount; ++i)
        settled = settled && std::fabs(feedback.joint[i].q - exitTarget()[i]) < 0.05
                  && std::fabs(feedback.joint[i].dq) < 0.05;
      settled = settled && std::fabs(feedback.rpy[0] - initialRpy_[0]) < 0.10
                        && std::fabs(feedback.rpy[1] - initialRpy_[1]) < 0.10;
      // Support and stable feedback must overlap for the whole dwell. A brief
      // confirmation after a long quiet hold must not authorize damping.
      // Withdrawal clears the dwell even between fresh feedback packets.
      if (!floorSupportObserved) exitStableStartNs_ = 0;
      // Only new robot feedback can start the verification window.
      if (recvFresh) {
        if (!settled || !floorSupportObserved) exitStableStartNs_ = 0;
        else if (exitStableStartNs_ == 0) exitStableStartNs_ = hostNs;
      }
      exitStableS_ = exitStableStartNs_ > 0
          ? (hostNs - exitStableStartNs_) / 1.0e9 : 0;
      if (settled && exitStableS_ >= 1.0 && floorSupportObserved) {
        exitSupportConfirmed_ = true;
        exitDampingStartNs_ = hostNs;
        dampingCommandNs_ = hostNs;
        transition(Phase::ExitDamping);
        command = dampingCommand();
      } else if (hostNs - exitVerifyNs_ >= 5000000000LL) {
        holdExit("exit_support_not_confirmed", hostNs);
        command = holdCommand(exitHoldQ_);
      }
      break;
    }
    case Phase::ExitHold:
      command = holdCommand(exitHoldQ_);
      // Deliberately latched: cancellation or absent support never authorizes
      // damping, automatic standing, or process exit.
      break;
    case Phase::ExitDamping:
      command = dampingCommand();
      if (sendResult != 0) enterPanic("exit_damping_send_failed", hostNs);
      else if (hostNs - exitDampingStartNs_ >= 1000000000LL)
        transition(Phase::Complete);
      break;
    case Phase::SafeHold:
      command = holdCommand(initialQ_);
      if (options_.dryRun && phaseElapsedS_ >= 2.0) transition(Phase::Complete);
      break;
    case Phase::PanicDamping:
      command = dampingCommand();
      if (dampingCommandNs_ == 0) dampingCommandNs_ = hostNs;
      if ((options_.dryRun && phaseElapsedS_ >= 0.5) ||
          (panicExitRequestNs_ > 0 &&
           hostNs - panicExitRequestNs_ >= 500000000LL))
        transition(Phase::Complete);
      break;
    case Phase::Complete:
    case Phase::Disarmed: break;
    }
    // A phase handler may discover a fault after it constructed a position
    // command. Do not publish that command for one extra control cycle.
    if (phase_ == Phase::PanicDamping) command = dampingCommand();
    sanitizeCommand(command);
    lastCommand_ = command;
    appendLog(feedback, command, recvFresh, recvResult, sendResult, loopDtUs, hostNs,
              watchdogActive);
    phaseElapsedS_ += kControlDt;
    return command;
  }

  bool writeLog() const {
    std::ofstream out(options_.logPath.c_str(), std::ios::out | std::ios::trunc);
    if (!out) return false;
    out << "host_monotonic_ns,state_tick_ms,phase,recv_ok,recv_result,send_result,"
           "loop_dt_us,watchdog_active,abort_reason,stop_source,"
           "stop_request_ns,damping_command_ns,active_leg,remote_valid,"
           "remote_head0,remote_head1,remote_buttons,remote_lx,remote_ly,"
           "remote_rx,remote_ry,remote_l2,level_flag,imu_roll,imu_pitch,imu_yaw,"
           "gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,cop_valid,"
           "cop_x_m,cop_y_m,total_foot_force,exit_support_confirmed,exit_stable_s";
    for (std::size_t leg = 0; leg < 4; ++leg)
      out << ",foot_force_" << leg << ",foot_force_baseline_" << leg
          << ",foot_force_mad_" << leg << ",support_margin_"
          << go1::legName(static_cast<go1::Leg>(leg)) << "_m,foot_target_"
          << leg << "_x_m,foot_target_" << leg << "_y_m,foot_target_"
          << leg << "_z_m";
    for (const char *name : kJointNames)
      out << ',' << name << "_cmd_q," << name << "_cmd_dq," << name
          << "_cmd_kp," << name << "_cmd_kd," << name << "_tau_ff,"
          << name << "_tau_cmd_total," << name << "_state_q," << name
          << "_state_dq," << name << "_tau_est," << name << "_mode,"
          << name << "_temperature";
    out << '\n' << std::setprecision(9);
    for (const auto &sample : logs_) writeLogRow(out, sample);
    out.flush();
    return out.good();
  }

private:
  float proneMaxKp() const {
    return options_.mode == ExperimentMode::ProneEngagement ? 1.0F : kProneKp;
  }
  float proneEffortLimit() const {
    return options_.mode == ExperimentMode::ProneEngagement ? 0.10F : kProneEffortLimitNm;
  }
  static std::array<float, kJointCount> zeroJointVelocity() {
    std::array<float, kJointCount> result;
    result.fill(0.0F);
    return result;
  }
  void resetProneObservation() {
    proneObserveCount_ = 0;
    proneObserveFirstNs_ = 0;
    proneObserveReference_.fill(0.0F);
    proneObserveSum_.fill(0.0);
  }
  bool observePronePose(const Feedback &f, bool hasState, bool fresh,
                        bool alive, bool tickValid, int64_t nowNs) {
    bool valid = hasState && alive && tickValid &&
                 feedbackReadinessIssue(f, hasState).empty();
    for (const auto &joint : f.joint)
      valid = valid && std::fabs(joint.dq) <= 0.05F;
    for (float angle : f.rpy) valid = valid && std::isfinite(angle);
    valid = valid && std::fabs(f.rpy[0]) <= 0.5F &&
                     std::fabs(f.rpy[1]) <= 0.5F;
    if (!valid) {
      resetProneObservation();
      return false;
    }
    if (!fresh) return false;
    if (proneObserveCount_ > 0) {
      bool nearReference = true;
      for (std::size_t i = 0; i < kJointCount; ++i)
        nearReference = nearReference &&
            std::fabs(f.joint[i].q - proneObserveReference_[i]) <= 0.02F;
      if (!nearReference) resetProneObservation();
    }
    if (proneObserveCount_ == 0) {
      proneObserveFirstNs_ = nowNs;
      for (std::size_t i = 0; i < kJointCount; ++i)
        proneObserveReference_[i] = f.joint[i].q;
    }
    ++proneObserveCount_;
    for (std::size_t i = 0; i < kJointCount; ++i)
      proneObserveSum_[i] += f.joint[i].q;
    if (proneObserveCount_ < 500 ||
        nowNs - proneObserveFirstNs_ <
            static_cast<int64_t>(kProneObserveS * 1.0e9))
      return false;
    for (std::size_t i = 0; i < kJointCount; ++i)
      initialQ_[i] = static_cast<float>(
          proneObserveSum_[i] / proneObserveCount_);
    initialRpy_ = f.rpy;
    poseCaptured_ = true;
    return true;
  }
  bool prepareProneTargets() {
    for (std::size_t i = 0; i < kJointCount; ++i)
      proneEngagementQ_[i] = clampValue(
          initialQ_[i], kJointMin[i % 3], kJointMax[i % 3]);
    for (std::size_t leg = 0; leg < 4; ++leg) {
      const std::size_t b = leg * 3;
      const go1::JointAngles seed{proneEngagementQ_[b],
                                  proneEngagementQ_[b + 1],
                                  proneEngagementQ_[b + 2]};
      const auto which = static_cast<go1::Leg>(leg);
      initialFeet_[leg] = go1::Kinematics::forward(which, seed);
      auto raisedFoot = initialFeet_[leg];
      // Holding the foot in place while moving it downward in trunk
      // coordinates raises the trunk by the same nominal amount.
      if (options_.mode != ExperimentMode::ProneEngagement)
        raisedFoot.z -= kProneRiseM;
      go1::JointAngles result;
      if (!go1::Kinematics::inverse(which, raisedFoot, seed, &result))
        return false;
      const std::array<double, 3> values{{result.hip, result.thigh,
                                           result.calf}};
      for (std::size_t joint = 0; joint < 3; ++joint) {
        if (std::fabs(values[joint] - proneEngagementQ_[b + joint]) > 0.05)
          return false;
        proneRiseQ_[b + joint] = static_cast<float>(values[joint]);
      }
      footTargets_[leg] = raisedFoot;
    }
    return true;
  }
  Command pronePositionCommand(
      const std::array<float, kJointCount> &target,
      const std::array<float, kJointCount> &velocity, float requestedKp,
      const Feedback &f) const {
    Command command;
    requestedKp = clampValue(requestedKp, 0.0F, proneMaxKp());
    for (std::size_t i = 0; i < kJointCount; ++i) {
      auto &joint = command.joint[i];
      joint.mode = kServoMode;
      joint.q = target[i];
      joint.dq = velocity[i];
      joint.kd = kProneKd;
      joint.tauFf = 0.0F;
      const float velocityError = velocity[i] - f.joint[i].dq;
      const float dampingBound = joint.kd * std::fabs(velocityError);
      const float remaining = std::max(0.0F,
          proneEffortLimit() - dampingBound);
      const float positionError = target[i] - f.joint[i].q;
      joint.kp = std::fabs(positionError) > 1.0e-6F
          ? std::min(requestedKp, remaining / std::fabs(positionError))
          : requestedKp;
      if (options_.mode == ExperimentMode::ProneEngagement && !stopSource_.empty())
        joint.kp = std::min(joint.kp, lastCommand_.joint[i].kp);
    }
    return command;
  }
  bool proneTargetSettled(
      const Feedback &f, const std::array<float, kJointCount> &target,
      bool fresh, bool alive) const {
    bool settled = fresh && alive && f.levelFlag == kLowLevel;
    for (std::size_t i = 0; i < kJointCount; ++i)
      settled = settled && std::isfinite(f.joint[i].q) &&
          std::isfinite(f.joint[i].dq) &&
          std::fabs(f.joint[i].q - target[i]) <=
              (options_.mode == ExperimentMode::ProneEngagement ? 0.10F : 0.05F) &&
          std::fabs(f.joint[i].dq) <= 0.05F;
    settled = settled && std::fabs(f.rpy[0] - initialRpy_[0]) <= 0.05F &&
                          std::fabs(f.rpy[1] - initialRpy_[1]) <= 0.05F;
    return settled;
  }
  bool updateProneStable(
      const Feedback &f, const std::array<float, kJointCount> &target,
      bool fresh, bool alive) {
    // Reusing a live sample supplies no new stability evidence, but does
    // not prove disturbed posture. Stale feedback still resets credit.
    if (!fresh) {
      if (!alive) proneStableS_ = 0.0;
      return false;
    }
    if (proneTargetSettled(f, target, fresh, alive))
      proneStableS_ += kControlDt;
    else
      proneStableS_ = 0.0;
    return proneStableS_ >= kProneHoldS;
  }
  void beginProneReturn() {
    if (phase_ == Phase::ProneReturn) return;
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const float candidate = lastCommand_.joint[i].q;
      proneReturnStartQ_[i] = candidate < 1.0e8F && std::isfinite(candidate)
          ? clampValue(candidate, kJointMin[i % 3], kJointMax[i % 3])
          : proneEngagementQ_[i];
    }
    transition(Phase::ProneReturn);
  }
  void holdProne(const std::string &reason, int64_t nowNs) {
    stopSource_ = reason;
    stopRequestNs_ = nowNs;
    if (reason != "ctrl_c") {
      faultReason_ = reason;
      failed_ = true;
    }
    transition(Phase::ProneSupportHold);
  }
  void requestProneStop(const std::string &source, const Feedback &,
                        int64_t nowNs) {
    if (phase_ == Phase::Complete || phase_ == Phase::PanicDamping ||
        phase_ == Phase::ProneSupportHold) return;
    if (stopSource_.empty()) {
      stopSource_ = source;
      stopRequestNs_ = nowNs;
      if (source != "ctrl_c") {
        failed_ = true;
        faultReason_ = source;
      }
    }
    if (phase_ == Phase::ProneObserve) {
      dampingCommandNs_ = nowNs;
      transition(Phase::ProneFinalDamping);
    } else if (phase_ != Phase::ProneReturn &&
               phase_ != Phase::ProneSettle &&
               phase_ != Phase::ProneRelease &&
               phase_ != Phase::ProneFinalDamping) {
      beginProneReturn();
    }
  }
  bool validateFreshTick(const Feedback &f, bool fresh) {
    if (!fresh) return true;
    if (!haveLastStateTick_) {
      lastStateTick_ = f.tickMs;
      haveLastStateTick_ = true;
      return true;
    }
    const uint32_t delta = f.tickMs - lastStateTick_;
    if (delta == 0 || delta > 20) return false;
    lastStateTick_ = f.tickMs;
    return true;
  }
  bool feedbackFiniteAndInBounds(const Feedback &f) const {
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const auto &j = f.joint[i];
      if (!std::isfinite(j.q) || !std::isfinite(j.dq) || !std::isfinite(j.tauEst))
        return false;
      if (j.q < kJointMin[i % 3] || j.q > kJointMax[i % 3]) return false;
    }
    for (float value : f.rpy) if (!std::isfinite(value)) return false;
    return true;
  }
  bool feedbackReady(const Feedback &f, bool hasState) const {
    return feedbackReadinessIssue(f, hasState).empty();
  }
  bool provisionalGroundPoseReady(const Feedback &f, bool hasState) const {
    if (!hasState || !feedbackFiniteAndInBounds(f)) return false;
    for (const auto &j : f.joint) {
      if (j.mode == kOverheatMode || j.temperature >= 70 ||
          std::fabs(j.dq) > 1.0) return false;
    }
    return true;
  }
  void applyFeedbackSafety(const Feedback &f, bool hasState, bool recvFresh,
                           bool recvAlive, int64_t nowNs) {
    if (phase_ == Phase::Precheck || phase_ == Phase::Disarmed ||
        phase_ == Phase::PanicDamping || phase_ == Phase::Complete) return;
    if (!hasState || !recvAlive) { enterPanic("feedback_gap_over_20ms", nowNs); return; }
    if (f.levelFlag != kLowLevel) {
      if (++invalidFeedbackStreak_ >= 5) enterPanic("left_lowlevel_mode", nowNs);
      return;
    }
    if (!recvFresh) return;
    invalidFeedbackStreak_ = 0;
    if (isProneMode(options_.mode) && !f.remote.valid) {
      enterPanic("remote_invalid_for_prone_mode", nowNs);
      return;
    }
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const auto &j = f.joint[i];
      if (j.mode == kOverheatMode) {
        enterPanic(std::string("motor_overheat_mode_") + kJointNames[i], nowNs); return;
      }
      if (j.temperature >= 70) {
        enterPanic(std::string("motor_temperature_") + kJointNames[i], nowNs); return;
      }
      if (!std::isfinite(j.q) || !std::isfinite(j.dq) || !std::isfinite(j.tauEst)) {
        enterPanic(std::string("nonfinite_feedback_") + kJointNames[i], nowNs); return;
      }
      float feedbackMin = kJointMin[i % 3];
      if (isProneMode(options_.mode) && i % 3 == 2)
        feedbackMin -= kProneCalfFeedbackMargin;
      if (j.q < feedbackMin || j.q > kJointMax[i % 3]) {
        enterPanic(std::string("joint_limit_") + kJointNames[i], nowNs); return;
      }
      if (poseCaptured_ && !inExitEnvelope() && std::fabs(j.q - initialQ_[i]) > 0.3) {
        enterPanic(std::string("joint_displacement_") + kJointNames[i], nowNs); return;
      }
      if (inExitEnvelope() && phase_ != Phase::ExitDamping &&
          lastCommand_.joint[i].q < 1.0e8F &&
          std::fabs(j.q - lastCommand_.joint[i].q) > 0.10) {
        enterPanic(std::string("exit_tracking_error_") + kJointNames[i], nowNs); return;
      }
      if (isPronePositionPhase(phase_) &&
          phase_ != Phase::ProneRelease &&
          lastCommand_.joint[i].q < 1.0e8F &&
          std::fabs(j.q - lastCommand_.joint[i].q) > 0.10) {
        enterPanic(std::string("prone_tracking_error_") + kJointNames[i], nowNs);
        return;
      }
      const double limit = options_.mode == ExperimentMode::ProneEngagement ? 0.08
                         : isProneMode(options_.mode) ? 0.3
                         : inExitEnvelope() ? 0.3 : isLegMotionPhase(phase_) ? 0.8
                         : isGroundMode(options_.mode) ? 1.0 : 2.0;
      if (std::fabs(j.dq) > limit) {
        enterPanic(std::string("joint_speed_") + kJointNames[i], nowNs); return;
      }
    }
    if (isGroundMode(options_.mode) && poseCaptured_) {
      for (float angle : f.rpy) {
        if (!std::isfinite(angle)) { enterPanic("nonfinite_imu", nowNs); return; }
      }
      const double limit = isProneMode(options_.mode) ? 0.05
                         : inExitEnvelope() || isLegMotionPhase(phase_) ? 0.10 : 0.20;
      if (std::fabs(f.rpy[0] - initialRpy_[0]) > limit ||
          std::fabs(f.rpy[1] - initialRpy_[1]) > limit) {
        if (phase_ == Phase::ProneRelease)
          enterPanic("prone_release_attitude", nowNs);
#if defined(GO1_WITH_POLICY)
        else if (phase_ == Phase::Return || phase_ == Phase::SafeHold)
          enterPanic("ground_attitude_during_recovery", nowNs);
#endif
        else
          requestSoftStop("ground_attitude_soft_abort", f, nowNs);
      }
    }
  }
  void observeRemoteStop(const Feedback &f, bool fresh, int64_t nowNs) {
    if (!fresh || !f.remote.valid || phase_ == Phase::RemotePreflight ||
        phase_ == Phase::PanicDamping || phase_ == Phase::Complete) {
      return;
    }
    const uint16_t required = kRemoteL2Mask | kRemoteBMask;
    if ((f.remote.buttons & required) == required) {
      // L2+B is already a deliberate two-button chord. Latch it on the first
      // fresh valid packet so damping is published in this control cycle.
      enterPanic("remote_l2_b", nowNs);
    }
  }
  Command precheckCommand(const Feedback &f, bool hasState, bool alive) {
    if (isGroundMode(options_.mode) && precheckPoseCaptured_)
      return holdCommand(precheckQ_);
    if (isGroundMode(options_.mode) && alive &&
        provisionalGroundPoseReady(f, hasState)) {
      if (!precheckPoseCaptured_) {
        for (std::size_t i = 0; i < kJointCount; ++i) precheckQ_[i] = f.joint[i].q;
        precheckPoseCaptured_ = true;
      }
      return holdCommand(precheckQ_);
    }
    return dampingCommand();
  }
  bool capturePose(const Feedback &f) {
    for (std::size_t i = 0; i < kJointCount; ++i)
      // Keep the already published target through PRECHECK -> HOLD. Replacing
      // it with a later measurement would introduce a command position step.
      initialQ_[i] = returnQ_[i] = isGroundMode(options_.mode) && precheckPoseCaptured_
          ? precheckQ_[i] : f.joint[i].q;
    initialRpy_ = f.rpy;
    for (std::size_t leg = 0; leg < 4; ++leg) {
      const std::size_t b = leg * 3;
      initialFeet_[leg] = go1::Kinematics::forward(
          static_cast<go1::Leg>(leg),
          {initialQ_[b], initialQ_[b + 1], initialQ_[b + 2]});
      footTargets_[leg] = initialFeet_[leg];
      if (!std::isfinite(initialFeet_[leg].x) ||
          !std::isfinite(initialFeet_[leg].y) ||
          !std::isfinite(initialFeet_[leg].z)) return false;
    }
    poseCaptured_ = true;
    return true;
  }
  void updateSupport(const Feedback &f) {
    SupportData next;
    std::array<double, 4> forces;
    for (std::size_t leg = 0; leg < 4; ++leg) {
      const std::size_t b = leg * 3;
      next.feet[leg] = go1::Kinematics::forward(
          static_cast<go1::Leg>(leg),
          {f.joint[b].q, f.joint[b + 1].q, f.joint[b + 2].q});
      forces[leg] = std::max(0.0, static_cast<double>(f.footForce[leg]));
      next.totalForce += forces[leg];
    }
    next.valid = go1::centerOfPressure(next.feet, forces, &next.cop);
    if (next.valid)
      for (std::size_t leg = 0; leg < 4; ++leg)
        next.margin[leg] = go1::signedSupportMargin(
            next.feet, static_cast<go1::Leg>(leg), next.cop);
    support_ = next;
  }
  void clearForceBaseline() {
    for (auto &samples : forceSamples_) samples.clear();
    forceBaseline_.fill(NAN); forceMad_.fill(NAN);
    shiftGateSeconds_ = contactGateSeconds_ = 0.0;
  }
  void collectForceBaseline(const Feedback &f, bool fresh) {
    if (!fresh) return;
    for (std::size_t leg = 0; leg < 4; ++leg)
      forceSamples_[leg].push_back(std::max(0.0, static_cast<double>(f.footForce[leg])));
  }
  bool finishForceBaseline() {
    baselineTotalForce_ = 0.0;
    for (std::size_t leg = 0; leg < 4; ++leg) {
      if (forceSamples_[leg].size() < 750) return false;
      forceBaseline_[leg] = median(forceSamples_[leg]);
      std::vector<double> deviations;
      for (double value : forceSamples_[leg])
        deviations.push_back(std::fabs(value - forceBaseline_[leg]));
      forceMad_[leg] = median(deviations);
      if (forceBaseline_[leg] < 5.0 ||
          forceMad_[leg] > 0.2 * forceBaseline_[leg]) return false;
      baselineTotalForce_ += forceBaseline_[leg];
    }
    return baselineTotalForce_ > 20.0;
  }
  bool selectAndPrepareLeg() {
    go1::Vec2 cop;
    if (!go1::centerOfPressure(initialFeet_, forceBaseline_, &cop)) return false;
    int selected = -1;
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t leg = 0; leg < 4; ++leg) {
      if (completedLeg_[leg] ||
          (!options_.legAuto && leg != static_cast<std::size_t>(options_.requestedLeg)))
        continue;
      const double margin = go1::signedSupportMargin(
          initialFeet_, static_cast<go1::Leg>(leg), cop);
      if (std::isfinite(margin) && margin > best) {
        selected = static_cast<int>(leg); best = margin;
      }
    }
    if (selected < 0) return false;
    activeLeg_ = static_cast<go1::Leg>(selected);
    activeLegValid_ = true;
    baselineMargin_ = best;
    const auto centroid = go1::supportCentroid(initialFeet_, activeLeg_);
    bodyShift_ = go1::clampNorm({centroid.x - cop.x, centroid.y - cop.y}, 0.030);
    for (std::size_t leg = 0; leg < 4; ++leg) {
      go1::Vec3 target = initialFeet_[leg];
      target.x -= bodyShift_.x; target.y -= bodyShift_.y;
      if (leg == static_cast<std::size_t>(activeLeg_)) target.z += options_.liftHeightM;
      const std::size_t b = leg * 3;
      const go1::JointAngles seed{initialQ_[b], initialQ_[b + 1], initialQ_[b + 2]};
      go1::JointAngles result;
      if (!go1::Kinematics::inverse(static_cast<go1::Leg>(leg), target, seed, &result) ||
          std::fabs(result.hip - seed.hip) > 0.3 ||
          std::fabs(result.thigh - seed.thigh) > 0.3 ||
          std::fabs(result.calf - seed.calf) > 0.3) return false;
    }
    return true;
  }

  Command dampingCommand() const {
    Command c;
    for (auto &j : c.joint) {
      j.mode = kDampingMode; j.q = kPosStop; j.dq = 0; j.kp = 0; j.kd = 1; j.tauFf = 0;
    }
    return c;
  }
  Command holdCommand(const std::array<float, kJointCount> &target) const {
    Command c;
    const float kp = isGroundMode(options_.mode) ? 20.0F : 5.0F;
    const float kd = isGroundMode(options_.mode) ? 1.5F : 1.0F;
    for (std::size_t i = 0; i < kJointCount; ++i) {
      c.joint[i].mode = kServoMode; c.joint[i].q = target[i];
      c.joint[i].dq = 0; c.joint[i].kp = kp; c.joint[i].kd = kd; c.joint[i].tauFf = 0;
#if defined(GO1_WITH_POLICY)
      if (options_.dryRun && options_.mode == ExperimentMode::WalkingPolicy &&
          (phase_ == Phase::Disarmed || phase_ == Phase::Precheck ||
           phase_ == Phase::CapturePose || phase_ == Phase::Hold))
        c.joint[i].q += offlineSupportBias_[i];
      if (options_.dryRun && options_.mode == ExperimentMode::WalkingPolicy)
        c.joint[i].tauFf=offlineSupportFeedforward_[i]*offlineSupportBlend_;
#endif
    }
    return c;
  }
  bool makeFootCommand(double shift, double lift, Command *command) {
    if (!command) return false;
    *command = holdCommand(initialQ_);
    for (std::size_t leg = 0; leg < 4; ++leg) {
      auto target = initialFeet_[leg];
      target.x -= bodyShift_.x * shift; target.y -= bodyShift_.y * shift;
      if (activeLegValid_ && leg == static_cast<std::size_t>(activeLeg_))
        target.z += options_.liftHeightM * lift;
      footTargets_[leg] = target;
      const std::size_t b = leg * 3;
      go1::JointAngles result;
      if (!go1::Kinematics::inverse(static_cast<go1::Leg>(leg), target,
            {initialQ_[b], initialQ_[b + 1], initialQ_[b + 2]}, &result)) return false;
      command->joint[b].q = static_cast<float>(result.hip);
      command->joint[b + 1].q = static_cast<float>(result.thigh);
      command->joint[b + 2].q = static_cast<float>(result.calf);
    }
    return true;
  }
  bool legSafe(const Feedback &f, double minMargin) const {
    if (!support_.valid || baselineTotalForce_ <= 0 || !activeLegValid_) return false;
    const double ratio = support_.totalForce / baselineTotalForce_;
    if (ratio < 0.70 || ratio > 1.30 ||
        support_.margin[static_cast<std::size_t>(activeLeg_)] < minMargin) return false;
    for (std::size_t leg = 0; leg < 4; ++leg)
      if (leg != static_cast<std::size_t>(activeLeg_) &&
          f.footForce[leg] < 0.05 * forceBaseline_[leg]) return false;
    return true;
  }
  Command weightShiftCommand(const Feedback &f, int64_t nowNs) {
    Command c;
    if (!makeFootCommand(smoothStep5(phaseElapsedS_ / 2.0), 0, &c)) {
      requestSoftStop("weight_shift_ik", f, nowNs); return holdCommand(initialQ_);
    }
    if (phaseElapsedS_ > 0.20 && !legSafe(f, 0.0)) {
      requestSoftStop("weight_shift_support", f, nowNs);
      return c;
    }
    const std::size_t leg = static_cast<std::size_t>(activeLeg_);
    if (support_.valid && support_.margin[leg] >= 0.015 &&
        f.footForce[leg] <= 0.30 * forceBaseline_[leg]) shiftGateSeconds_ += kControlDt;
    else shiftGateSeconds_ = 0;
    if (phaseElapsedS_ >= 1.0 && support_.valid &&
        support_.margin[leg] < baselineMargin_ + 0.005) {
      requestSoftStop("weight_shift_not_improving", f, nowNs);
      return c;
    } else if (phaseElapsedS_ >= 2.0) {
      if (shiftGateSeconds_ >= 0.5) transition(Phase::Lift);
      else requestSoftStop("weight_shift_gate_timeout", f, nowNs);
    }
    return c;
  }
  Command liftCommand(const Feedback &f, int64_t nowNs) {
    Command c;
    if (!makeFootCommand(1, smoothStep5(phaseElapsedS_ / 1.5), &c)) {
      requestSoftStop("lift_ik", f, nowNs); return holdCommand(initialQ_);
    }
    const std::size_t leg = static_cast<std::size_t>(activeLeg_);
    if (!legSafe(f, 0.010) ||
        (phaseElapsedS_ > 0.20 && f.footForce[leg] > 0.20 * forceBaseline_[leg])) {
      requestSoftStop("lift_support_or_contact", f, nowNs);
      return c;
    } else if (phaseElapsedS_ >= 1.5) transition(Phase::AirHold);
    return c;
  }
  Command airHoldCommand(const Feedback &f, int64_t nowNs) {
    Command c;
    if (!makeFootCommand(1, 1, &c)) {
      requestSoftStop("air_hold_ik", f, nowNs); return holdCommand(initialQ_);
    }
    const std::size_t leg = static_cast<std::size_t>(activeLeg_);
    if (!legSafe(f, 0.010) || f.footForce[leg] > 0.20 * forceBaseline_[leg]) {
      requestSoftStop("air_hold_support_or_contact", f, nowNs); return c;
    }
    double envelope = 1.0;
    if (phaseElapsedS_ < 0.05) envelope = smoothStep5(phaseElapsedS_ / 0.05);
    else if (phaseElapsedS_ > 0.95) envelope = smoothStep5((1.0 - phaseElapsedS_) / 0.05);
    c.joint[leg * 3 + 1].tauFf = static_cast<float>(
        options_.tauOverlayNm * envelope *
        std::sin(2.0 * kPi * options_.tauOverlayHz * phaseElapsedS_));
    if (phaseElapsedS_ >= 1.0) transition(Phase::Lower);
    return c;
  }
  Command lowerCommand(const Feedback &f, int64_t nowNs) {
    Command c;
    if (!makeFootCommand(1, 1.0 - smoothStep5(phaseElapsedS_ / 1.5), &c)) {
      requestSoftStop("lower_ik", f, nowNs); return holdCommand(initialQ_);
    }
    if (!legSafe(f, 0.010)) {
      requestSoftStop("lower_support", f, nowNs);
      return c;
    } else if (phaseElapsedS_ >= 1.5) transition(Phase::ContactVerify);
    return c;
  }
  Command contactCommand(const Feedback &f, int64_t nowNs) {
    Command c;
    if (!makeFootCommand(1, 0, &c)) {
      requestSoftStop("contact_ik", f, nowNs); return holdCommand(initialQ_);
    }
    if (!legSafe(f, -0.002)) {
      requestSoftStop("contact_support", f, nowNs);
      return c;
    }
    const std::size_t leg = static_cast<std::size_t>(activeLeg_);
    if (f.footForce[leg] >= 0.60 * forceBaseline_[leg]) contactGateSeconds_ += kControlDt;
    else contactGateSeconds_ = 0;
    if (contactGateSeconds_ >= 0.5) transition(Phase::Recenter);
    else if (phaseElapsedS_ >= 1.5) requestSoftStop("contact_verify_timeout", f, nowNs);
    return c;
  }
  Command recenterCommand(const Feedback &f, int64_t nowNs) {
    Command c;
    if (!makeFootCommand(1.0 - smoothStep5(phaseElapsedS_ / 2.0), 0, &c)) {
      requestSoftStop("recenter_ik", f, nowNs); return holdCommand(initialQ_);
    }
    const double ratio = baselineTotalForce_ > 0 ? support_.totalForce / baselineTotalForce_ : 0;
    if (!support_.valid || ratio < 0.70 || ratio > 1.30) {
      requestSoftStop("recenter_force", f, nowNs);
      return c;
    } else if (phaseElapsedS_ >= 2.0) {
      completedLeg_[static_cast<std::size_t>(activeLeg_)] = true;
      if (options_.mode == ExperimentMode::LegLiftSequence &&
          std::find(completedLeg_.begin(), completedLeg_.end(), false) != completedLeg_.end()) {
        activeLegValid_ = false; transition(Phase::InterLeg);
      } else beginReturn(f);
    }
    return c;
  }
  Command torqueCommand(const Feedback &f) const {
    Command c = holdCommand(initialQ_);
    if (options_.mode == ExperimentMode::Preflight) return c;
    const double ramp = std::min(0.05, options_.durationS * 0.5);
    double envelope = 1.0;
    if (phaseElapsedS_ < ramp) envelope = smoothStep5(phaseElapsedS_ / ramp);
    else if (phaseElapsedS_ > options_.durationS - ramp)
      envelope = smoothStep5((options_.durationS - phaseElapsedS_) / ramp);
    const float torque = static_cast<float>(options_.amplitudeNm * envelope *
        std::sin(2.0 * kPi * options_.frequencyHz * phaseElapsedS_));
    if (options_.mode == ExperimentMode::TorqueSine) {
      auto &j = c.joint[static_cast<std::size_t>(options_.joint)];
      const auto &state = f.joint[static_cast<std::size_t>(options_.joint)];
      j.q = kPosStop; j.dq = kVelStop; j.kp = 0; j.kd = 0;
      j.tauFf = static_cast<float>(torque +
          2.0 * (initialQ_[static_cast<std::size_t>(options_.joint)] - state.q) -
          0.2 * state.dq);
    } else if (options_.mode == ExperimentMode::TorqueSineGround) {
      for (std::size_t leg = 0; leg < 4; ++leg) c.joint[leg * 3 + 1].tauFf = torque;
    }
    return c;
  }
  Command squatCommand() const {
    Command c = holdCommand(initialQ_);
    double p = 0, pd = 0;
    if (phaseElapsedS_ < 3.0) {
      const double x = phaseElapsedS_ / 3.0;
      p = smoothStep5(x); pd = smoothStep5Derivative(x) / 3.0;
    } else if (phaseElapsedS_ < 5.0) p = 1;
    else {
      const double x = (phaseElapsedS_ - 5.0) / 3.0;
      p = 1.0 - smoothStep5(x); pd = -smoothStep5Derivative(x) / 3.0;
    }
    for (std::size_t leg = 0; leg < 4; ++leg) {
      const std::size_t thigh = leg * 3 + 1, calf = leg * 3 + 2;
      c.joint[thigh].q = initialQ_[thigh] + static_cast<float>(0.12 * p);
      c.joint[thigh].dq = static_cast<float>(0.12 * pd);
      c.joint[calf].q = initialQ_[calf] + static_cast<float>(-0.24 * p);
      c.joint[calf].dq = static_cast<float>(-0.24 * pd);
    }
    return c;
  }
  Command returnCommand() const {
    Command c = holdCommand(initialQ_);
    const double p = smoothStep5(phaseElapsedS_ / 2.0);
    const double pd = smoothStep5Derivative(phaseElapsedS_ / 2.0) / 2.0;
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const double delta = initialQ_[i] - returnQ_[i];
      c.joint[i].q = returnQ_[i] + static_cast<float>(delta * p);
      c.joint[i].dq = static_cast<float>(delta * pd);
    }
    return c;
  }
  bool inExitEnvelope() const {
    return options_.dryRunNormalExit &&
        (phase_ == Phase::ExitLower || phase_ == Phase::ExitVerify ||
         phase_ == Phase::ExitHold || phase_ == Phase::ExitDamping);
  }
  static std::array<float, kJointCount> exitTarget() {
    // Simulation fixture, NOT a calibrated prone pose. Factory calf feedback
    // (~-2.80) is outside the command limit and must never be replayed here.
    return {{-0.28F, 1.25F, -2.70F, 0.28F, 1.25F, -2.70F,
             -0.28F, 1.25F, -2.70F, 0.28F, 1.25F, -2.70F}};
  }
  Command exitLowerCommand(int64_t nowNs) const {
    const double t = (nowNs - exitStartNs_) / 8.0e9;
    Command c = holdCommand(exitStartQ_);
    const auto target = exitTarget();
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const double delta = target[i] - exitStartQ_[i];
      c.joint[i].q += static_cast<float>(delta * smoothStep5(t));
      c.joint[i].dq = static_cast<float>(delta * smoothStep5Derivative(t) / 8.0);
    }
    return c;
  }
  void holdExit(const std::string &reason, int64_t nowNs) {
    for (std::size_t i = 0; i < kJointCount; ++i)
      exitHoldQ_[i] = lastCommand_.joint[i].q;
    stopSource_ = reason; stopRequestNs_ = nowNs;
    if (reason != "ctrl_c") { failed_ = true; faultReason_ = reason; }
    transition(Phase::ExitHold);
  }
  void beginReturn(const Feedback &f) {
    if (!poseCaptured_ || phase_ == Phase::Return ||
        phase_ == Phase::PanicDamping || phase_ == Phase::Complete) return;
    for (std::size_t i = 0; i < kJointCount; ++i) returnQ_[i] = f.joint[i].q;
    transition(Phase::Return);
  }
  void requestSoftStop(const std::string &source, const Feedback &f, int64_t nowNs) {
    if (isProneMode(options_.mode)) {
      requestProneStop(source, f, nowNs);
      return;
    }
    if (inExitEnvelope()) {
      if (phase_ != Phase::ExitDamping && phase_ != Phase::ExitHold)
        holdExit(source, nowNs);
      return;
    }
    if (options_.dryRunNormalExit && phase_ == Phase::Return) {
      // A cancellation arriving during the return must also prevent the
      // following automatic descent; continue the existing return smoothly.
      stopSource_ = source; stopRequestNs_ = nowNs;
      if (source != "ctrl_c") { failed_ = true; faultReason_ = source; }
      return;
    }
    if (phase_ == Phase::PanicDamping || phase_ == Phase::Complete ||
        phase_ == Phase::Return || phase_ == Phase::SafeHold) return;
    stopSource_ = source; stopRequestNs_ = nowNs;
    if (!poseCaptured_) { enterPanic(source + "_before_pose_capture", nowNs); return; }
    if (source != "ctrl_c") { failed_ = true; faultReason_ = source; }
    beginReturn(f);
  }
  void enterPanic(const std::string &source, int64_t nowNs) {
#if defined(GO1_WITH_POLICY)
    // External faults (worker loss, watchdog, SDK guards) must cancel pending
    // inference too, not only switch the motor command to damping.
    if (options_.mode == ExperimentMode::WalkingPolicy) policySession_.stop(source);
#endif
    if (phase_ == Phase::Complete || phase_ == Phase::PanicDamping) return;
    failed_ = true; faultReason_ = stopSource_ = source; stopRequestNs_ = nowNs;
    transition(Phase::PanicDamping); panicReached_.store(true);
    // Remote preflight is only permitted while the robot is fully prone, so
    // every preflight panic may close automatically after a final damping
    // window. Ground-mode panic remains latched until the operator requests
    // exit explicitly.
    if (options_.mode == ExperimentMode::RemotePreflight ||
        isProneMode(options_.mode))
      requestPanicExit(nowNs);
  }
  void requestPanicExit(int64_t nowNs) {
    if (panicExitRequestNs_ == 0) panicExitRequestNs_ = nowNs;
  }
  void sanitizeCommand(Command &c) const {
    for (std::size_t i = 0; i < kJointCount; ++i) {
      auto &j = c.joint[i];
      j.tauFf = clampValue(j.tauFf, -1.0F, 1.0F);
      if (poseCaptured_ && j.q < 1.0e8F) {
        if (!inExitEnvelope())
          j.q = clampValue(j.q, initialQ_[i] - 0.3F, initialQ_[i] + 0.3F);
        j.q = clampValue(j.q, kJointMin[i % 3], kJointMax[i % 3]);
      }
    }
  }
  void appendLog(const Feedback &f, const Command &c, bool fresh, int recv,
                 int send, double loopUs, int64_t hostNs, bool watchdog) {
    LogSample s;
    s.hostNs = hostNs; s.tickMs = f.tickMs; s.phase = phase_; s.recvFresh = fresh;
    s.recvResult = recv; s.sendResult = send;
    s.loopDtUs = loopUs; s.watchdogActive = watchdog;
    s.abortReason = faultReason_; s.stopSource = stopSource_;
    s.stopRequestNs = stopRequestNs_; s.dampingCommandNs = dampingCommandNs_;
    s.exitSupportConfirmed = exitSupportConfirmed_; s.exitStableS = exitStableS_;
    s.activeLeg = activeLegIndex(); s.feedback = f; s.command = c;
    s.support = support_; s.forceBaseline = forceBaseline_; s.forceMad = forceMad_;
    s.footTarget = footTargets_; fillTauTotal(s); logs_.push_back(s);
  }
  static void fillTauTotal(LogSample &s) {
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const auto &c = s.command.joint[i]; const auto &f = s.feedback.joint[i];
      s.tauTotal[i] = c.tauFf + c.kd * (c.dq - f.dq);
      if (c.q < 1.0e8F) s.tauTotal[i] += c.kp * (c.q - f.q);
    }
  }
  static void writeLogRow(std::ofstream &out, const LogSample &s) {
    const auto &r = s.feedback.remote;
    const char *legName = s.activeLeg >= 0
        ? go1::legName(static_cast<go1::Leg>(s.activeLeg)) : "";
    out << s.hostNs << ',' << s.tickMs << ',' << phaseName(s.phase) << ','
        << (s.recvFresh ? 1 : 0) << ',' << s.recvResult << ','
        << s.sendResult << ',' << s.loopDtUs
        << ',' << (s.watchdogActive ? 1 : 0) << ',' << s.abortReason << ','
        << s.stopSource << ',' << s.stopRequestNs << ',' << s.dampingCommandNs
        << ',' << legName << ',' << (r.valid ? 1 : 0) << ','
        << static_cast<int>(r.head0) << ',' << static_cast<int>(r.head1) << ','
        << r.buttons << ',' << r.lx << ',' << r.ly << ',' << r.rx << ','
        << r.ry << ',' << r.l2 << ',' << static_cast<int>(s.feedback.levelFlag)
        << ',' << s.feedback.rpy[0] << ','
        << s.feedback.rpy[1] << ',' << s.feedback.rpy[2] << ','
        << s.feedback.gyro[0] << ',' << s.feedback.gyro[1] << ','
        << s.feedback.gyro[2] << ',' << s.feedback.accel[0] << ','
        << s.feedback.accel[1] << ',' << s.feedback.accel[2] << ','
        << (s.support.valid ? 1 : 0) << ',' << s.support.cop.x << ','
        << s.support.cop.y << ',' << s.support.totalForce << ','
        << s.exitSupportConfirmed << ',' << s.exitStableS;
    for (std::size_t leg = 0; leg < 4; ++leg)
      out << ',' << s.feedback.footForce[leg] << ',' << s.forceBaseline[leg]
          << ',' << s.forceMad[leg] << ',' << s.support.margin[leg] << ','
          << s.footTarget[leg].x << ',' << s.footTarget[leg].y << ','
          << s.footTarget[leg].z;
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const auto &c = s.command.joint[i]; const auto &f = s.feedback.joint[i];
      out << ',' << c.q << ',' << c.dq << ',' << c.kp << ',' << c.kd << ','
          << c.tauFf << ',' << s.tauTotal[i] << ',' << f.q << ',' << f.dq
          << ',' << f.tauEst << ',' << static_cast<int>(f.mode) << ','
          << static_cast<int>(f.temperature);
    }
    out << '\n';
  }
  void transition(Phase next) {
    phase_ = next; phaseElapsedS_ = 0;
    if (next == Phase::Baseline) clearForceBaseline();
    if (next == Phase::WeightShift) shiftGateSeconds_ = 0;
    if (next == Phase::ContactVerify) contactGateSeconds_ = 0;
    if (next == Phase::ProneEngageHold || next == Phase::ProneRiseHold)
      proneStableS_ = 0.0;
    if (next == Phase::ProneSettle || next == Phase::ProneSupportHold) {
      proneSupportStableStartNs_ = 0;
      proneSupportReadyForNewAssert_ = false;
      exitStableS_ = 0.0;
    }
    if (next == Phase::SafeHold) safeHoldReached_.store(true);
    if (next == Phase::Complete) doneFlag_.store(true);
  }

  Options options_;
#if defined(GO1_WITH_POLICY)
  go1::PolicySession policySession_;
  go1::PolicyEvidence policyEvidence_;
#endif
  Phase phase_ = Phase::Disarmed;
  double phaseElapsedS_ = 0;
  int validPacketStreak_ = 0, invalidFeedbackStreak_ = 0;
  uint32_t lastStateTick_ = 0;
  bool haveLastStateTick_ = false;
  bool poseCaptured_ = false, precheckPoseCaptured_ = false, failed_ = false;
  bool remoteCommunicationReady_ = false, remoteEverValid_ = false;
  bool remoteChordSeen_ = false;
  int handledSignalCount_ = 0;
  int64_t lastSignalNs_ = 0, stopRequestNs_ = 0, dampingCommandNs_ = 0;
  int64_t panicExitRequestNs_ = 0;
  std::string faultReason_, stopSource_;
  std::atomic<bool> doneFlag_{false}, safeHoldReached_{false}, panicReached_{false};
#if defined(GO1_WITH_POLICY)
  std::array<float, kJointCount> offlineSupportBias_{},offlineSupportFeedforward_{};
  int64_t offlinePolicyStartNs_=0;
  int64_t policyTrialStartNs_=0;
  double offlineSupportBlend_=1.;
#endif
  std::array<float, kJointCount> precheckQ_{{0}}, initialQ_{{0}}, returnQ_{{0}};
  std::array<float, kJointCount> proneObserveReference_{{0}};
  std::array<double, kJointCount> proneObserveSum_{{0}};
  std::array<float, kJointCount> proneEngagementQ_{{0}}, proneRiseQ_{{0}};
  std::array<float, kJointCount> proneReturnStartQ_{{0}};
  int proneObserveCount_ = 0;
  int64_t proneObserveFirstNs_ = 0, proneSupportStableStartNs_ = 0;
  bool proneSupportReadyForNewAssert_ = false;
  double proneStableS_ = 0.0;
  int64_t groundEntryStartNs_ = 0;
  std::array<float, kJointCount> exitStartQ_{{0}}, exitHoldQ_{{0}};
  Command lastCommand_;
  int64_t exitStartNs_ = 0, exitVerifyNs_ = 0, exitStableStartNs_ = 0;
  int64_t exitDampingStartNs_ = 0;
  double exitStableS_ = 0;
  bool exitSupportConfirmed_ = false;
  std::array<float, 3> initialRpy_{{0, 0, 0}};
  std::array<go1::Vec3, 4> initialFeet_, footTargets_;
  std::array<std::vector<double>, 4> forceSamples_;
  std::array<double, 4> forceBaseline_{{NAN, NAN, NAN, NAN}};
  std::array<double, 4> forceMad_{{NAN, NAN, NAN, NAN}};
  double baselineTotalForce_ = 0;
  SupportData support_;
  go1::Leg activeLeg_ = go1::Leg::FR;
  bool activeLegValid_ = false;
  std::array<bool, 4> completedLeg_{{false, false, false, false}};
  go1::Vec2 bodyShift_;
  double baselineMargin_ = NAN, shiftGateSeconds_ = 0, contactGateSeconds_ = 0;
  std::vector<LogSample> logs_;
};

void updateDryFootForces(const ExperimentCore &core, Feedback *f) {
  f->footForce = {{50, 50, 50, 50}};
  const int active = core.activeLegIndex();
  if (active < 0) return;
  double target = 50;
  if (core.phase() == Phase::WeightShift)
    target = 50.0 * (1.0 - 0.8 * smoothStep5(core.phaseElapsedS() / 2.0));
  else if (core.phase() == Phase::Lift || core.phase() == Phase::AirHold ||
           core.phase() == Phase::Lower) target = 5;
  else if (core.phase() == Phase::ContactVerify)
    target = 5.0 + 45.0 * smoothStep5(core.phaseElapsedS() / 0.5);
  const double add = (50.0 - target) / 3.0;
  for (std::size_t leg = 0; leg < 4; ++leg)
    f->footForce[leg] = static_cast<int16_t>(
        leg == static_cast<std::size_t>(active) ? target : 50.0 + add);
}
void simulatePlant(const Command &command, Feedback *f,
                   bool allowProneCalfMargin = false) {
  for (std::size_t i = 0; i < kJointCount; ++i) {
    const auto &c = command.joint[i]; auto &state = f->joint[i];
    double torque = c.tauFf;
    if (c.q < 1.0e8F) torque += c.kp * (c.q - state.q) + c.kd * (c.dq - state.dq);
    else torque -= 0.15 * state.dq;
    if (c.mode == kDampingMode) torque = -c.kd * state.dq;
    const double acceleration = 4.0 * torque - 0.4 * state.dq;
    state.dq += static_cast<float>(acceleration * kControlDt);
    state.q += state.dq * static_cast<float>(kControlDt);
    const float feedbackMin = allowProneCalfMargin && i % 3 == 2
        ? kJointMin[2] - kProneCalfFeedbackMargin : kJointMin[i % 3];
    state.q = clampValue(state.q, feedbackMin, kJointMax[i % 3]);
    state.tauEst = static_cast<float>(0.95 * torque);
    state.mode = c.mode; state.temperature = 30;
  }
  f->tickMs += 2;
}
int runDry(const Options &options) {
  ExperimentCore core(options);
  Feedback f;
  f.levelFlag = kLowLevel; f.accel[2] = 9.81F;
  f.remote.valid = true; f.remote.head0 = 0xFE; f.remote.head1 = 0xEF;
  const std::array<float, kJointCount> observedProne{{
      -0.311616F, 1.275168F, -2.794212F,
       0.289090F, 1.272262F, -2.797724F,
      -0.289514F, 1.280437F, -2.799541F,
       0.302533F, 1.237503F, -2.768375F}};
  for (std::size_t leg = 0; leg < 4; ++leg) {
    const std::size_t b = leg * 3;
    if (isProneMode(options.mode)) {
      for (std::size_t joint = 0; joint < 3; ++joint)
        f.joint[b + joint].q = observedProne[b + joint];
    } else {
      f.joint[b].q = 0;
      f.joint[b + 1].q = 0.8F;
      f.joint[b + 2].q = options.mode == ExperimentMode::RemotePreflight
                             ? -2.80F : -1.5F;
    }
    for (std::size_t j = 0; j < 3; ++j) {
      f.joint[b + j].mode = kServoMode; f.joint[b + j].temperature = 30;
    }
  }
  Command command;
  const int64_t start = steadyNowNs();
  const std::size_t maxCycles = static_cast<std::size_t>(100.0 / kControlDt);
  bool soft = false, panicStarted = false, doubleCtrl = false;
  bool watchdogInjected = false;
  int panicFrames = 0;
  for (std::size_t cycle = 0; cycle < maxCycles && !core.done(); ++cycle) {
    const double elapsed = cycle * kControlDt;
    updateDryFootForces(core, &f);
    simulatePlant(command, &f, isProneMode(options.mode));
    if (options.mode == ExperimentMode::RemotePreflight)
      for (std::size_t leg = 0; leg < 4; ++leg)
        f.joint[leg * 3 + 2].q = -2.80F;
    const int64_t now = start + static_cast<int64_t>(elapsed * 1.0e9);
    if (!soft && options.injectSoftStopS >= 0 && elapsed >= options.injectSoftStopS) {
      soft = true; gSignalCount = 1;
    }
    if (!doubleCtrl && options.injectDoubleCtrlCS >= 0 &&
        elapsed >= options.injectDoubleCtrlCS) {
      doubleCtrl = true;
      gSignalCount = 2;
    }
    if (!panicStarted && options.injectPanicS >= 0 &&
        elapsed >= options.injectPanicS)
      panicStarted = true;
    if (panicStarted && panicFrames++ < 5)
      f.remote.buttons = kRemoteL2Mask | kRemoteBMask;
    else if (panicStarted)
      f.remote.buttons = 0;
    if (options.mode == ExperimentMode::RemotePreflight && cycle >= 500 && cycle < 510)
      f.remote.buttons = kRemoteL2Mask | kRemoteBMask;
    core.observeSignalCount(static_cast<int>(gSignalCount), now, f);
    const bool watchdog = !watchdogInjected && options.injectWatchdogS >= 0 &&
                          elapsed >= options.injectWatchdogS;
    watchdogInjected = watchdogInjected || watchdog;
    // Explicit synthetic belly-support evidence for this development fixture.
    // simulatePlant has no ground-contact dynamics and cannot validate this.
    const bool simulatedSupport =
        (options.dryRunNormalExit && core.phase() == Phase::ExitVerify &&
         core.phaseElapsedS() >= 1.5) ||
        (isProneMode(options.mode) &&
         core.phase() == Phase::ProneSettle &&
         core.phaseElapsedS() >= 1.5);
    command = core.step(f, true, true, true, 0, 0, 2000, now, watchdog,
                        simulatedSupport);
  }
  gSignalCount = 0;
  if (!core.writeLog()) return 2;
  if (!core.done()) {
    std::cerr << "Dry run reached its simulation time limit in "
              << phaseName(core.phase()) << "; diagnostic log=" << options.logPath << '\n';
    return 4;
  }
  std::cout << "Dry run complete: mode=" << modeName(options.mode)
            << ", samples=" << core.logCount() << ", log=" << options.logPath << '\n';
  if (core.failed() && options.injectPanicS < 0 &&
      options.injectDoubleCtrlCS < 0 &&
      options.injectWatchdogS < 0) {
    std::cerr << "Dry run entered a fault state: " << core.faultReason() << '\n';
    return 3;
  }
  return 0;
}

#if defined(GO1_WITH_SDK)
using namespace UNITREE_LEGGED_SDK;

void requireUdpPortAvailable(uint16_t port, const char *purpose) {
  const int socketFd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socketFd < 0)
    throw std::runtime_error(std::string("cannot create UDP port probe for ") +
                             purpose + ": " + std::strerror(errno));
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);
  if (::bind(socketFd, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) < 0) {
    const std::string detail = std::strerror(errno);
    ::close(socketFd);
    throw std::runtime_error(
        std::string(purpose) + " local UDP port " + std::to_string(port) +
        " is unavailable: " + detail +
        ". Inspect every UDP state with: sudo ss -uanp; then identify the "
        "owner with: sudo fuser -v " + std::to_string(port) + "/udp");
  }
  ::close(socketFd);
}

void validateHardwarePorts(const Options &options) {
  requireUdpPortAvailable(options.localPort, "low-level controller");
  if (requiresStandingCapture(options.mode))
    requireUdpPortAvailable(options.highLocalPort,
                            "high-level pose capture");
}

Feedback convertFeedback(const LowState &state) {
  Feedback f;
  f.levelFlag = state.levelFlag; f.tickMs = state.tick;
  for (std::size_t i = 0; i < kJointCount; ++i) {
    f.joint[i].mode = state.motorState[i].mode;
    f.joint[i].q = state.motorState[i].q;
    f.joint[i].dq = state.motorState[i].dq;
    f.joint[i].tauEst = state.motorState[i].tauEst;
    f.joint[i].temperature = state.motorState[i].temperature;
  }
  for (std::size_t i = 0; i < 3; ++i) {
    f.rpy[i] = state.imu.rpy[i]; f.gyro[i] = state.imu.gyroscope[i];
    f.accel[i] = state.imu.accelerometer[i];
  }
  for (std::size_t i = 0; i < 4; ++i) f.footForce[i] = state.footForce[i];
  xRockerBtnDataStruct remote = {};
  std::memcpy(&remote, &state.wirelessRemote[0], sizeof(remote));
  f.remote.head0 = remote.head[0]; f.remote.head1 = remote.head[1];
  f.remote.buttons = remote.btn.value; f.remote.lx = remote.lx; f.remote.ly = remote.ly;
  f.remote.rx = remote.rx; f.remote.ry = remote.ry; f.remote.l2 = remote.L2;
  const bool payloadNonzero = std::any_of(
      state.wirelessRemote.begin(), state.wirelessRemote.end(),
      [](uint8_t value) { return value != 0; });
  // The SDK does not document a required value for the first two joystick
  // bytes, and its own example does not validate them. Some Go1 firmware
  // leaves them at values other than FE EF, so validate the payload and
  // decoded ranges instead.
  f.remote.valid = payloadNonzero &&
      std::isfinite(remote.lx) && std::isfinite(remote.ly) &&
      std::isfinite(remote.rx) && std::isfinite(remote.ry) &&
      std::fabs(remote.lx) <= 1.5F && std::fabs(remote.ly) <= 1.5F &&
      std::fabs(remote.rx) <= 1.5F && std::fabs(remote.ry) <= 1.5F;
  return f;
}
void copyCommand(const Command &command, LowCmd *sdk) {
  for (std::size_t i = 0; i < kJointCount; ++i) {
    sdk->motorCmd[i].mode = command.joint[i].mode;
    sdk->motorCmd[i].q = command.joint[i].q;
    sdk->motorCmd[i].dq = command.joint[i].dq;
    sdk->motorCmd[i].Kp = command.joint[i].kp;
    sdk->motorCmd[i].Kd = command.joint[i].kd;
    sdk->motorCmd[i].tau = command.joint[i].tauFf;
  }
}
Command convertCommand(const LowCmd &sdk) {
  Command command;
  for (std::size_t i = 0; i < kJointCount; ++i) {
    command.joint[i].mode = sdk.motorCmd[i].mode;
    command.joint[i].q = sdk.motorCmd[i].q;
    command.joint[i].dq = sdk.motorCmd[i].dq;
    command.joint[i].kp = sdk.motorCmd[i].Kp;
    command.joint[i].kd = sdk.motorCmd[i].Kd;
    command.joint[i].tauFf = sdk.motorCmd[i].tau;
  }
  return command;
}
// Defense in depth: direct construction cannot bypass parseOptions' walking
// lock and open an SDK socket. Future promotion needs a separate review.
const Options& lockedHardwareOptions(const Options& options) {
  if(options.mode==ExperimentMode::WalkingPolicy)
    throw std::runtime_error("hardware walking policy remains locked before SDK construction");
  return options;
}
class HardwareRunner {
public:
  explicit HardwareRunner(const Options &options)
      : options_(lockedHardwareOptions(options)), safety_(LeggedType::Go1),
        udp_(LOWLEVEL, options.localPort, options.targetIp.c_str(), options.targetPort),
        core_(options),
        controlLoop_("go1_control", static_cast<float>(kControlDt),
                     boost::bind(&HardwareRunner::controlStep, this)),
        sendLoop_("go1_send", static_cast<float>(kControlDt), 3,
                  boost::bind(&HardwareRunner::sendStep, this)),
        recvLoop_("go1_recv", static_cast<float>(kControlDt), 3,
                  boost::bind(&HardwareRunner::recvStep, this)) {
    udp_.InitCmdData(sendPacket_); udp_.InitCmdData(safetyPacket_);
    publishedCommand_ = core_.emergencyDampingCommand();
    lastCommandPublishNs_.store(steadyNowNs());
  }
  int run() {
#if defined(GO1_WITH_POLICY)
    if(options_.mode==ExperimentMode::WalkingPolicy) {
      // No launch paths/evidence are provisioned in the locked release. A
      // readiness failure must happen before any capture or motor threads.
      try{policyRuntime_.start(policyLaunch_,policyQualification_);}
      catch(const std::exception& e){std::cerr<<e.what()<<"; no motor sends started\n";return 3;}
    }
#endif
    if (requiresStandingCapture(options_.mode)) {
      std::array<float, kJointCount> standingPose{{0}};
      if (!captureStandingPoseHighLevel(&standingPose)) {
        std::cerr << "Ground takeover aborted before low-level transmission: "
                     "a stable high-level standing pose was not received.\n";
        core_.forceHardFault("highlevel_standing_pose_capture_failed",
                             steadyNowNs());
        if (!core_.writeLog()) return 2;
        return gSignalCount > 0 ? 130 : 3;
      }
      core_.seedGroundPose(standingPose);
      {
        std::lock_guard<std::mutex> lock(commandMutex_);
        publishedCommand_ = core_.seededGroundHoldCommand();
      }
      lastCommandPublishNs_.store(steadyNowNs());
    }

    if (isProneMode(options_.mode)) {
      std::string supportError;
      if (!supportServer_.start(go1::kOperatorSupportPort, &supportError)) {
        std::cerr << "Prone operator-support loopback listener unavailable: "
                  << supportError << "; no motor packets were sent.\n";
        return 3;
      }
    }

    if (options_.mode == ExperimentMode::ProneEngagement) {
      std::cout << "Support receiver ready; no motor packets sent yet. Open the Ubuntu GUI.\n"
                   "With belly/all feet on floor and remote ready, press Enter here to begin: "
                << std::flush;
      std::string ready;
      if (!std::getline(std::cin, ready) || gSignalCount > 0) return 130;
    }

    // The Unitree SDK examples actively send before expecting state. Starting
    // the send loop first also creates the return path when Ubuntu reaches the
    // robot through the onboard computer's NAT.
    sendLoop_.start();
    recvLoop_.start();
    controlLoop_.start();
    bool safeShown = false, panicShown = false;
    Phase shownPhase = Phase::Disarmed;
    while (!finished_.load()) {
      const Phase phase = reportedPhase_.load();
      if (isProneMode(options_.mode) && phase != shownPhase) {
        shownPhase = phase;
        std::cout << "phase=" << phaseName(phase) << std::endl;
        if (reportedConfirmationAllowed_.load())
          std::cout << "CONFIRM CONTACT NOW: click once only if belly and all feet remain on floor. "
                       "If movement/contact is abnormal, use L2+B; do not confirm." << std::endl;
        else if (phase == Phase::ProneSettle || phase == Phase::ProneSupportHold)
          std::cout << "FAULT HOLD: contact confirmation cannot release this fault. "
                       "Use L2+B for final damping and exit; keep clear." << std::endl;
      }
      if (core_.safeHoldReached() && !safeShown) {
        safeShown = true;
        std::cout << "Sequence complete; holding four-foot pose. Press Ctrl-C "
                     "twice within one second for damping.\n";
      }
      if (core_.panicReached() && !panicShown) {
        panicShown = true;
        std::cerr << "PANIC DAMPING ACTIVE: " << core_.faultReason()
                  << ". Keep clear.";
        if (options_.mode == ExperimentMode::RemotePreflight ||
            isProneMode(options_.mode))
          std::cerr << " The prone preflight will close automatically after "
                       "its damping window.\n";
        else
          std::cerr << " When safe, press Ctrl-C once to close UDP and write "
                       "the log.\n";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    controlLoop_.shutdown();
    sendLoop_.shutdown();
    recvLoop_.shutdown();
    supportServer_.stop();
#if defined(GO1_WITH_POLICY)
    if(options_.mode==ExperimentMode::WalkingPolicy && !policyRuntime_.stop()) {
      core_.forceHardFault("policy_worker_cleanup_failed",steadyNowNs());
      core_.writeLog();return 3;
    }
#endif
    if (!core_.writeLog()) return 2;
    std::cout << "Hardware run complete: samples=" << core_.logCount()
              << ", log=" << options_.logPath << '\n';
    return core_.failed() ? 3 : 0;
  }
private:
  bool captureStandingPoseHighLevel(
      std::array<float, kJointCount> *standingPose) {
    if (!standingPose) return false;
    std::cout << "Requesting a stable pose through the high-level idle "
                 "endpoint; standing posture must be established separately...\n";
    UDP highUdp(HIGHLEVEL, options_.highLocalPort,
                options_.highTargetIp.c_str(), options_.highTargetPort);
    HighCmd command = {};
    highUdp.InitCmdData(command);
    command.mode = 0;       // Active high-level idle request; NOT a stand command.
    command.gaitType = 0;
    command.speedLevel = 0;
    command.footRaiseHeight = 0.0F;
    command.bodyHeight = 0.0F;
    command.euler = {{0.0F, 0.0F, 0.0F}};
    command.velocity = {{0.0F, 0.0F}};
    command.yawSpeed = 0.0F;

    StandingPoseCapture capture;
    const int64_t deadline = steadyNowNs() + 3000000000LL;
    while (steadyNowNs() < deadline && gSignalCount == 0) {
      if (highUdp.SetSend(command) < 0 ||
          sdkSendStatus(highUdp.Send(), HIGH_CMD_LENGTH) != 0) return false;
      const auto before = highUdp.udpState;
      const int result = highUdp.Recv();
      const auto after = highUdp.udpState;
      const bool received = result >= 0 && after.RecvCount > before.RecvCount &&
          after.FlagError == before.FlagError && after.RecvCRCError == before.RecvCRCError;
      Feedback sample;
      if (received) {
        HighState candidate = {};
        highUdp.GetRecv(candidate);
        xRockerBtnDataStruct remote = {};
        std::memcpy(&remote, candidate.wirelessRemote.data(), sizeof(remote));
        const uint16_t stop = kRemoteL2Mask | kRemoteBMask;
        if ((remote.btn.value & stop) == stop) return false;
        sample.levelFlag = candidate.levelFlag;
        sample.rpy = candidate.imu.rpy;
        for (std::size_t i = 0; i < kJointCount; ++i) {
          sample.joint[i].q = candidate.motorState[i].q;
          sample.joint[i].dq = candidate.motorState[i].dq;
          sample.joint[i].mode = candidate.motorState[i].mode;
          sample.joint[i].temperature = candidate.motorState[i].temperature;
        }
      }
      if (capture.observe(sample, after.RecvCount, steadyNowNs(), received, standingPose)) {
        std::cout << "Captured " << capture.count()
                  << " quiet high-level receive events over at least 200 ms; "
                     "preparing the seeded low-level hold.\n";
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
  }
  void recvStep() {
    const auto before = go1::SdkReceiveCounters::snapshot(udp_.udpState);
    const int result = udp_.Recv();
    const auto after = go1::SdkReceiveCounters::snapshot(udp_.udpState);
    const bool received = go1::validatedSdkReceive(result,before,after);
    recvResult_.store(received ? result : -1);
    if (!received) return; // Never restamp GetRecv's retained/CRC-invalid packet.
    LowState candidate = {}; udp_.GetRecv(candidate);
    if (!hasState_.load() || candidate.tick != recvThreadLastTick_) {
      recvThreadLastTick_ = candidate.tick;
      // Publish the state and its sequence under the same lock. Previously the
      // sequence increment happened after releasing stateMutex_. A control
      // cycle could therefore copy the old state, observe the new sequence,
      // and incorrectly label a duplicate tick as a fresh packet.
      {
        std::lock_guard<std::mutex> lock(stateMutex_);
        lowState_ = candidate;
        feedback_ = convertFeedback(candidate);
        stateSequence_.fetch_add(1, std::memory_order_relaxed);
        lastRecvNs_.store(steadyNowNs(), std::memory_order_relaxed);
        hasState_.store(true, std::memory_order_relaxed);
      }
    }
  }
  void sendStep() {
    const int64_t now = steadyNowNs();
    Command command;
    { std::lock_guard<std::mutex> lock(commandMutex_); command = publishedCommand_; }
    const int64_t published = lastCommandPublishNs_.load();
    const bool stale = published <= 0 || now - published > 20000000LL;
    if (stale) command = core_.emergencyDampingCommand();
    watchdogActive_.store(stale);
    copyCommand(command, &sendPacket_);
    const int result=go1::stageAndSend(udp_, sendPacket_);
    sendResult_.store(result);
#if defined(GO1_WITH_POLICY)
    // Acknowledge what this send thread actually selected (including watchdog
    // damping), not the actor proposal or the control thread's pre-limit target.
    { std::lock_guard<std::mutex> lock(sentAckMutex_);
      sentAckCommand_=command; sentAckSuccess_=sdkSendStatus(result,LOW_CMD_LENGTH)==0;
      ++sentAckSequence_; }
#endif

  }
  void controlStep() {
#if defined(GO1_WITH_POLICY)
    if(options_.mode==ExperimentMode::WalkingPolicy) {
      std::lock_guard<std::mutex> lock(sentAckMutex_);
      if(sentAckSequence_!=consumedSentAckSequence_) {
        std::array<double,12> target; bool position=true;
        for(std::size_t i=0;i<12;++i) {
          const auto& j=sentAckCommand_.joint[i]; target[i]=j.q;
          position=position && j.mode==kServoMode && j.kp>0 && j.q<1e8F;
        }
        core_.policySession().acknowledge(target,position,sentAckSuccess_);
        consumedSentAckSequence_=sentAckSequence_;
      }
    }
#endif
    Feedback f; LowState state = {};
    bool hasState = false;
    uint64_t sequence = 0;
    int64_t lastRecv = 0;
    {
      // Copy the packet, sequence and arrival time atomically with respect to recvStep()
      // so recv_ok/fresh always describes the copied tick.
      std::lock_guard<std::mutex> lock(stateMutex_);
      hasState = hasState_.load(std::memory_order_relaxed);
      if (hasState) {
        f = feedback_;
        state = lowState_;
      }
      sequence = stateSequence_.load(std::memory_order_relaxed);
      lastRecv = lastRecvNs_.load(std::memory_order_relaxed);
    }
    // Sample control time after acquisition snapshot, so a just-arrived packet
    // cannot legitimately have an arrival timestamp newer than this tick.
    const int64_t now = steadyNowNs();
    const double loopUs = lastControlNs_ == 0 ? 2000.0 : (now - lastControlNs_) / 1000.0;
    lastControlNs_ = now;
    const bool fresh = sequence != controlLastSequence_; controlLastSequence_ = sequence;
    if (options_.mode == ExperimentMode::RemotePreflight && fresh &&
        shouldPrintRemote(f, now)) {
      const uint16_t required = kRemoteL2Mask | kRemoteBMask;
      std::cout << "remote valid=" << (f.remote.valid ? 1 : 0)
                << " buttons=0x" << std::hex << f.remote.buttons << std::dec
                << " L2+B="
                << (((f.remote.buttons & required) == required) ? 1 : 0)
                << " lx=" << f.remote.lx << " ly=" << f.remote.ly
                << " rx=" << f.remote.rx << " ry=" << f.remote.ry
                << " L2=" << f.remote.l2
                << " head=0x" << std::hex << static_cast<int>(f.remote.head0)
                << static_cast<int>(f.remote.head1)
                << " level=0x" << static_cast<int>(f.levelFlag) << std::dec
                << " tick=" << f.tickMs
                << " ready_issue="
                << core_.feedbackReadinessIssue(f, hasState) << '\n';
    } else if (options_.mode == ExperimentMode::RemotePreflight && !hasState &&
               now - lastNoStatePrintNs_ >= 1000000000LL) {
      lastNoStatePrintNs_ = now;
      std::cout << "waiting for valid LowState; recv_result="
                << recvResult_.load() << " damping_stream=active\n";
    }
    core_.observeSignalCount(static_cast<int>(gSignalCount), now, f);
    const bool alive = hasState && lastRecv > 0 && now >= lastRecv &&
                       now - lastRecv <= 20000000LL;
    const int rawSendResult = sendResult_.load();
#if defined(GO1_WITH_POLICY)
    if(options_.mode==ExperimentMode::WalkingPolicy)
      policyRuntime_.before(core_,state,hasState,fresh,lastRecv,sequence,now,policyQualification_);
#endif
    Command command = core_.step(
        f, hasState, fresh, alive, recvResult_.load(),
        sdkSendStatus(rawSendResult, LOW_CMD_LENGTH),
        loopUs, now, watchdogActive_.load(),
        isProneMode(options_.mode) && supportServer_.active(now));
#if defined(GO1_WITH_POLICY)
    if(options_.mode==ExperimentMode::WalkingPolicy && !policyRuntime_.after(core_,now))
      command=core_.emergencyDampingCommand();
#endif
    copyCommand(command, &safetyPacket_);
    safety_.PositionLimit(safetyPacket_);
    if (hasState) {
      if (safety_.PowerProtect(safetyPacket_, state, 1) < 0) {
        core_.forceHardFault("power_protect", now);
        copyCommand(core_.emergencyDampingCommand(), &safetyPacket_);
      }
    }
    command = convertCommand(safetyPacket_);
    core_.amendLastLoggedCommand(command, watchdogActive_.load(), rawSendResult);
    { std::lock_guard<std::mutex> lock(commandMutex_); publishedCommand_ = command; }
    lastCommandPublishNs_.store(now);
    reportedConfirmationAllowed_.store(core_.proneContactConfirmationAllowed());
    reportedPhase_.store(core_.phase());
    if (core_.done()) finished_.store(true);
  }
  bool shouldPrintRemote(const Feedback &f, int64_t now) {
    const bool changed = f.remote.valid != lastRemoteValid_ ||
                         f.remote.buttons != lastRemoteButtons_;
    if (!changed && now - lastRemotePrintNs_ < 200000000LL) return false;
    lastRemoteValid_ = f.remote.valid;
    lastRemoteButtons_ = f.remote.buttons;
    lastRemotePrintNs_ = now;
    return true;
  }
  Options options_; Safety safety_; UDP udp_; ExperimentCore core_;
  go1::OperatorSupportServer supportServer_;
#if defined(GO1_WITH_POLICY)
  go1::PolicyRuntime policyRuntime_;
  go1::PolicyLaunchSpec policyLaunch_; // No runtime/CLI bypass of release lock.
  go1::PolicyQualification policyQualification_; // All false until independently qualified.
#endif
  LowCmd sendPacket_ = {}, safetyPacket_ = {}; LowState lowState_ = {};
  Feedback feedback_; Command publishedCommand_;
  std::mutex stateMutex_, commandMutex_;
#if defined(GO1_WITH_POLICY)
  std::mutex sentAckMutex_;
  Command sentAckCommand_;
  bool sentAckSuccess_=false;
  uint64_t sentAckSequence_=0,consumedSentAckSequence_=0;
#endif
  std::atomic<Phase> reportedPhase_{Phase::Disarmed};
  std::atomic<bool> reportedConfirmationAllowed_{false};
  std::atomic<bool> hasState_{false}, watchdogActive_{false}, finished_{false};
  std::atomic<uint64_t> stateSequence_{0};
  std::atomic<int64_t> lastRecvNs_{0}, lastCommandPublishNs_{0};
  std::atomic<int> recvResult_{0}, sendResult_{0};
  uint64_t controlLastSequence_ = 0; uint32_t recvThreadLastTick_ = 0;
  int64_t lastControlNs_ = 0, lastRemotePrintNs_ = 0;
  int64_t lastNoStatePrintNs_ = 0;
  uint16_t lastRemoteButtons_ = 0;
  bool lastRemoteValid_ = false;
  LoopFunc controlLoop_, sendLoop_, recvLoop_;
};
#endif

int jointIndex(const std::string &name) {
  for (std::size_t i = 0; i < kJointCount; ++i)
    if (name == kJointNames[i]) return static_cast<int>(i);
  throw std::runtime_error("Unknown joint name: " + name);
}
double parseDouble(const std::string &flag, const std::string &value) {
  std::size_t parsed = 0; const double result = std::stod(value, &parsed);
  if (parsed != value.size() || !std::isfinite(result))
    throw std::runtime_error("Invalid value for " + flag + ": " + value);
  return result;
}
int parseInt(const std::string &flag, const std::string &value) {
  std::size_t parsed = 0; const int result = std::stoi(value, &parsed);
  if (parsed != value.size()) throw std::runtime_error("Invalid value for " + flag);
  return result;
}
void printUsage(const char *program) {
  std::cout << "Engagement-only mode: --mode prone-engagement (no rise, Kp<=1).\n";
  std::cout << "Usage: " << program << " [options]\n\n"
      << "  --mode remote-preflight|ground-handover|preflight|torque-sine|\n"
         "         torque-sine-ground|squat|leg-lift|leg-lift-sequence|\n"
         "         prone-low-rise\n"
      << "  --leg auto|FR|FL|RR|RL\n  --lift-height-m 0.02\n"
      << "  --tau-overlay-nm 0.10\n  --tau-overlay-hz 0.5\n"
      << "  --joint FR_1\n  --amplitude-nm 0.2\n  --frequency-hz 0.5\n"
      << "  --duration-s 10\n  --log PATH\n  --support-confirmed\n"
      << "  --ground-confirmed\n  --remote-confirmed\n  --prone-confirmed\n"
      << "  --dry-run\n"
      << "  --dry-run-normal-exit     ground-handover endpoint development fixture\n"
      << "  --target-ip ADDRESS       (default 192.168.123.10)\n"
      << "  --local-port PORT         (default 8090)\n"
      << "  --target-port PORT        (default 8007)\n"
      << "  --high-target-ip ADDRESS  (default 192.168.123.161)\n"
      << "  --high-local-port PORT    (default 8091)\n"
      << "  --high-target-port PORT   (default 8082)\n";
}
Options parseOptions(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value = [&](const std::string &flag) {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
      return std::string(argv[++i]);
    };
    if (arg == "--help") { printUsage(argv[0]); std::exit(0); }
    else if (arg == "--mode") {
      const std::string v = value(arg);
      if (v == "remote-preflight") o.mode = ExperimentMode::RemotePreflight;
      else if (v == "ground-handover") o.mode = ExperimentMode::GroundHandover;
      else if (v == "preflight") o.mode = ExperimentMode::Preflight;
      else if (v == "torque-sine") o.mode = ExperimentMode::TorqueSine;
      else if (v == "torque-sine-ground") o.mode = ExperimentMode::TorqueSineGround;
      else if (v == "squat") o.mode = ExperimentMode::Squat;
      else if (v == "leg-lift") o.mode = ExperimentMode::LegLift;
      else if (v == "leg-lift-sequence") o.mode = ExperimentMode::LegLiftSequence;
      else if (v == "walking-policy") o.mode = ExperimentMode::WalkingPolicy;
      else if (v == "prone-engagement") o.mode = ExperimentMode::ProneEngagement;
      else if (v == "prone-low-rise") o.mode = ExperimentMode::ProneLowRise;
      else throw std::runtime_error("Unknown mode: " + v);
    } else if (arg == "--leg") {
      const std::string v = value(arg);
      if (v == "auto") o.legAuto = true;
      else { go1::Leg leg; if (!go1::parseLeg(v.c_str(), &leg)) throw std::runtime_error("Unknown leg: " + v);
             o.legAuto = false; o.requestedLeg = leg; }
    } else if (arg == "--lift-height-m") o.liftHeightM = parseDouble(arg, value(arg));
    else if (arg == "--tau-overlay-nm") o.tauOverlayNm = parseDouble(arg, value(arg));
    else if (arg == "--tau-overlay-hz") o.tauOverlayHz = parseDouble(arg, value(arg));
    else if (arg == "--joint") o.joint = jointIndex(value(arg));
    else if (arg == "--amplitude-nm") o.amplitudeNm = parseDouble(arg, value(arg));
    else if (arg == "--frequency-hz") o.frequencyHz = parseDouble(arg, value(arg));
    else if (arg == "--duration-s") o.durationS = parseDouble(arg, value(arg));
    else if (arg == "--log") o.logPath = value(arg);
    else if (arg == "--target-ip") o.targetIp = value(arg);
    else if (arg == "--high-target-ip") o.highTargetIp = value(arg);
    else if (arg == "--local-port" || arg == "--target-port" ||
             arg == "--high-local-port" || arg == "--high-target-port") {
      const int port = parseInt(arg, value(arg));
      if (port < 1 || port > 65535) throw std::runtime_error(arg + " must be 1..65535");
      if (arg == "--local-port") o.localPort = static_cast<uint16_t>(port);
      else if (arg == "--target-port") o.targetPort = static_cast<uint16_t>(port);
      else if (arg == "--high-local-port")
        o.highLocalPort = static_cast<uint16_t>(port);
      else o.highTargetPort = static_cast<uint16_t>(port);
    } else if (arg == "--support-confirmed") o.supportConfirmed = true;
    else if (arg == "--ground-confirmed") o.groundConfirmed = true;
    else if (arg == "--remote-confirmed") o.remoteConfirmed = true;
    else if (arg == "--prone-confirmed") o.proneConfirmed = true;
    else if (arg == "--dry-run") o.dryRun = true;
    else if (arg == "--dry-run-normal-exit") o.dryRunNormalExit = true;
    else if (arg == "--inject-soft-stop-s") o.injectSoftStopS = parseDouble(arg, value(arg));
    else if (arg == "--inject-panic-s") o.injectPanicS = parseDouble(arg, value(arg));
    else if (arg == "--inject-double-ctrl-c-s") o.injectDoubleCtrlCS = parseDouble(arg, value(arg));
    else if (arg == "--inject-watchdog-s") o.injectWatchdogS = parseDouble(arg, value(arg));
    else throw std::runtime_error("Unknown option: " + arg);
  }
  if (o.amplitudeNm < 0 || o.amplitudeNm > 1) throw std::runtime_error("--amplitude-nm must be in [0, 1]");
  if (o.frequencyHz < 0.1 || o.frequencyHz > 3) throw std::runtime_error("--frequency-hz must be in [0.1, 3]");
  if (o.durationS < 0.1 || o.durationS > 300) throw std::runtime_error("--duration-s must be in [0.1, 300]");
  if (o.liftHeightM < 0.005 || o.liftHeightM > 0.03) throw std::runtime_error("--lift-height-m must be in [0.005, 0.03]");
  if (o.tauOverlayNm < 0 || o.tauOverlayNm > 0.2) throw std::runtime_error("--tau-overlay-nm must be in [0, 0.2]");
  if (o.tauOverlayHz < 0.1 || o.tauOverlayHz > 3) throw std::runtime_error("--tau-overlay-hz must be in [0.1, 3]");
  if (o.supportConfirmed && o.groundConfirmed) throw std::runtime_error("support and ground confirmations are mutually exclusive");
  if (o.proneConfirmed && o.mode != ExperimentMode::RemotePreflight &&
      !isProneMode(o.mode))
    throw std::runtime_error(
        "--prone-confirmed is only valid with remote-preflight or a prone mode");
#if !defined(GO1_WITH_POLICY)
  if (o.mode == ExperimentMode::WalkingPolicy)
    throw std::runtime_error("walking-policy development is not included in this build; no UDP opened");
#endif
  if (!o.dryRun && o.mode == ExperimentMode::WalkingPolicy)
    throw std::runtime_error("walking-policy is SDK-free/offline only: estimator, actuator profile and operator transport are not hardware-qualified; no UDP opened");
  if (!o.dryRun && (o.mode == ExperimentMode::RemotePreflight ||
                    isProneMode(o.mode)) &&
      !o.proneConfirmed)
    throw std::runtime_error("prone hardware mode requires --prone-confirmed");
  if (o.localPort == o.highLocalPort)
    throw std::runtime_error("low-level and high-level local UDP ports must differ");
  if (!o.dryRun && (o.mode == ExperimentMode::Preflight || o.mode == ExperimentMode::TorqueSine) && !o.supportConfirmed)
    throw std::runtime_error("supported hardware mode requires --support-confirmed");
  if (!o.dryRun && requiresStandingCapture(o.mode) && !o.groundConfirmed)
    throw std::runtime_error("ground hardware mode requires --ground-confirmed");
  if (!o.dryRun && o.mode == ExperimentMode::ProneEngagement && !o.remoteConfirmed)
    throw std::runtime_error("prone-engagement requires --remote-confirmed");
  if (!o.dryRun && isLegMode(o.mode) && !o.remoteConfirmed)
    throw std::runtime_error("hardware leg-lift requires --remote-confirmed");
  if (o.mode == ExperimentMode::LegLiftSequence && !o.legAuto)
    throw std::runtime_error("leg-lift-sequence requires --leg auto");
  if (o.dryRunNormalExit && (!o.dryRun || o.mode != ExperimentMode::GroundHandover))
    throw std::runtime_error("--dry-run-normal-exit requires --dry-run --mode ground-handover");
  if (!o.dryRun && isGroundMode(o.mode) && o.mode != ExperimentMode::ProneEngagement)
    throw std::runtime_error(
        "ground hardware modes are locked pending prone-low-rise software "
        "verification and takeover review; no UDP opened");
  if (!o.dryRun && (o.injectSoftStopS >= 0 || o.injectPanicS >= 0 ||
                    o.injectDoubleCtrlCS >= 0 ||
                    o.injectWatchdogS >= 0))
    throw std::runtime_error("stop injection is dry-run-only");
  return o;
}
void signalHandler(int) { if (gSignalCount < 100) ++gSignalCount; }
} // namespace

#ifndef GO1_CORE_TEST
int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    std::signal(SIGINT, signalHandler); std::signal(SIGTERM, signalHandler);
  if (options.dryRun) return runDry(options);
#if defined(GO1_WITH_SDK)
    if (!go1::confirmLogOverwrite(options.logPath, std::cin, std::cout) ||
        gSignalCount > 0) return 130;
    validateHardwarePorts(options);
    { std::ofstream probe(options.logPath.c_str(), std::ios::out | std::ios::app);
      if (!probe) throw std::runtime_error("cannot write log path: " + options.logPath); }
    if (options.mode != ExperimentMode::RemotePreflight) {
      if (options.mode == ExperimentMode::ProneEngagement)
        std::cout << "PRONE ENGAGEMENT TRIAL: all 12 joints, Kp<=1, zero feed-forward torque, "
                     "no rise trajectory. Clamped calf targets can still move/load the robot.\n"
                     "Confirm belly and all feet on floor, factory L2+B, remote ready.\n";
      std::cout << "WARNING: low-level control can injure people or damage the robot.\n"
                   "Software damping is not a physical emergency stop.\n"
                   "Type ARM and press Enter to continue: " << std::flush;
      std::string confirmation;
      if (!std::getline(std::cin, confirmation) || confirmation != "ARM" || gSignalCount > 0) {
        std::cerr << "Not armed; no hardware packets were sent.\n"; return 130;
      }
    } else {
      std::cout <<
          "WARNING: remote-preflight actively sends low-level damping to all "
          "joints.\nThe robot must already be fully prone with nobody near the "
          "legs.\nType ARM DAMPING and press Enter to continue: " << std::flush;
      std::string confirmation;
      if (!std::getline(std::cin, confirmation) ||
          confirmation != "ARM DAMPING" || gSignalCount > 0) {
        std::cerr << "Not armed; no hardware packets were sent.\n";
        return 130;
      }
    }
#if defined(GO1_WITH_POLICY)
    go1::CommandOwner commandOwner;
#endif
    HardwareRunner runner(options); return runner.run();
#else
    std::cerr << "This build is dry-run-only. Rebuild on Ubuntu/Go1 for hardware access.\n";
    return 2;
#endif
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n'; return 2;
  }
}
#endif
