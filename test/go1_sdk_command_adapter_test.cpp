// Exercises SDK Safety and conversion only. Never constructs UDP/HardwareRunner.
#define GO1_CORE_TEST
#define GO1_WITH_SDK 1
#include "../src/go1_lowlevel_experiment.cpp"

int main() {
  try {
    UNITREE_LEGGED_SDK::Safety safety(UNITREE_LEGGED_SDK::LeggedType::Go1);
    for (int scenario = 0; scenario < 3; ++scenario) {
      Command original;
      LowState state = {};
      state.levelFlag = kLowLevel;
      for (std::size_t i = 0; i < kJointCount; ++i) {
        auto &j = original.joint[i];
        if (scenario == 0) { // position-free emergency/final damping
          j.mode = kDampingMode; j.q = kPosStop;
          j.dq = 0; j.kp = 0; j.kd = 1; j.tauFf = 0;
        } else if (scenario == 1) { // prone engagement at the calf bound
          j.mode = kServoMode;
          j.q = i % 3 == 2 ? kJointMin[2] : i % 3 == 1 ? 1.27F : 0;
          j.dq = 0; j.kp = 5; j.kd = 1; j.tauFf = 0;
        } else { // selected torque channel and its sentinel fields
          j.mode = kServoMode; j.q = kPosStop; j.dq = kVelStop;
          j.kp = 0; j.kd = 0; j.tauFf = i == 1 ? 0.10F : 0;
        }
        state.motorState[i].mode = kServoMode;
        state.motorState[i].q = i % 3 == 2 ? -2.79F : i % 3 == 1 ? 1.27F : 0;
        state.motorState[i].dq = 0;
        state.motorState[i].tauEst = 0;
        state.motorState[i].temperature = 30;
      }
      LowCmd packet = {};
      packet.levelFlag = kLowLevel;
      copyCommand(original, &packet);
      safety.PositionLimit(packet);
      if (safety.PowerProtect(packet, state, 1) < 0)
        throw std::runtime_error("unexpected power protection in zero-power fixture");
      const Command result = convertCommand(packet);
      for (std::size_t i = 0; i < kJointCount; ++i) {
        const auto &a = original.joint[i]; const auto &b = result.joint[i];
        // PositionLimit clamps even PosStop sentinels in the bundled SDK.
        // With Kp=0, q contributes no commanded stiffness. Require all effort
        // fields unchanged, and only permit this known zero-Kp q clamping.
        const bool sentinelClamp = a.q == kPosStop && a.kp == 0 && b.kp == 0 &&
            std::isfinite(b.q) && b.q >= kJointMin[i % 3] - 0.002F &&
            b.q <= kJointMax[i % 3] + 0.002F;
        if (a.mode != b.mode || (std::fabs(a.q-b.q) > 1e-6F && !sentinelClamp) ||
            std::fabs(a.dq-b.dq) > 1e-6F || std::fabs(a.kp-b.kp) > 1e-6F ||
            std::fabs(a.kd-b.kd) > 1e-6F || std::fabs(a.tauFf-b.tauFf) > 1e-6F) {
          std::cerr << "scenario=" << scenario << " joint=" << i
                    << " q=" << a.q << " -> " << b.q << '\n';
          throw std::runtime_error("SDK changed a reviewed command field");
        }
      }
      std::cout << "PASS " << (scenario == 0 ? "damping" : scenario == 1 ?
                               "prone engagement" : "torque channel") << '\n';
    }
    std::cout << "NOTE: SDK PositionLimit clamps PosStop q with Kp=0; effort fields preserved\n";
    std::cout << "sdk_command_adapter=PASS; no UDP constructed or motor commands sent\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
