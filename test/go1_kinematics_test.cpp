#include "go1_kinematics.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

bool near(double left, double right, double tolerance) {
  return std::fabs(left - right) <= tolerance;
}

bool inGo1JointBounds(const go1::JointAngles &q) {
  return q.hip >= -1.047 && q.hip <= 1.047 && q.thigh >= -0.663 &&
         q.thigh <= 2.966 && q.calf >= -2.721 && q.calf <= -0.837;
}

int fail(const char *message) {
  std::cerr << "FAIL: " << message << '\n';
  return EXIT_FAILURE;
}

} // namespace

int main() {
  const go1::JointAngles standing{0.0, 0.8, -1.5};
  std::array<go1::Vec3, go1::kLegCount> feet;
  for (std::size_t index = 0; index < go1::kLegCount; ++index) {
    const go1::Leg leg = static_cast<go1::Leg>(index);
    feet[index] = go1::Kinematics::forward(leg, standing);
    go1::JointAngles recovered;
    if (!go1::Kinematics::inverse(leg, feet[index], standing, &recovered)) {
      return fail("IK rejected a nominal standing FK target");
    }
    if (!near(recovered.hip, standing.hip, 1.0e-8) ||
        !near(recovered.thigh, standing.thigh, 1.0e-8) ||
        !near(recovered.calf, standing.calf, 1.0e-8)) {
      return fail("FK/IK round trip changed the standing posture");
    }

    go1::Vec3 shifted = feet[index];
    shifted.x += 0.015;
    shifted.y -= 0.010;
    go1::JointAngles shiftedQ;
    if (!go1::Kinematics::inverse(leg, shifted, standing, &shiftedQ)) {
      return fail("IK rejected a conservative body-shift target");
    }
    if (!inGo1JointBounds(shiftedQ)) {
      return fail("IK returned a target outside Go1 joint bounds");
    }
    const go1::Vec3 roundTrip = go1::Kinematics::forward(leg, shiftedQ);
    if (!near(roundTrip.x, shifted.x, 1.0e-6) ||
        !near(roundTrip.y, shifted.y, 1.0e-6) ||
        !near(roundTrip.z, shifted.z, 1.0e-6)) {
      return fail("shifted FK/IK round trip exceeded tolerance");
    }

    go1::Vec3 unreachable = go1::Kinematics::hipOrigin(leg);
    unreachable.z = -1.0;
    if (go1::Kinematics::inverse(leg, unreachable, standing, &shiftedQ)) {
      return fail("IK accepted a target outside the leg workspace");
    }
  }

  const std::array<double, go1::kLegCount> equalForce{{50, 50, 50, 50}};
  go1::Vec2 cop;
  go1::Vec2 expectedCop;
  for (const go1::Vec3 &foot : feet) {
    expectedCop.x += foot.x / 4.0;
    expectedCop.y += foot.y / 4.0;
  }
  if (!go1::centerOfPressure(feet, equalForce, &cop) ||
      !near(cop.x, expectedCop.x, 1.0e-10) ||
      !near(cop.y, expectedCop.y, 1.0e-10)) {
    return fail("equal-force CoP is not centered");
  }

  const double boundary =
      go1::signedSupportMargin(feet, go1::Leg::FR, cop);
  if (!near(boundary, 0.0, 1.0e-9)) {
    return fail("four-foot center should lie on the FR-excluded boundary");
  }
  const go1::Vec2 centroid = go1::supportCentroid(feet, go1::Leg::FR);
  if (!(go1::signedSupportMargin(feet, go1::Leg::FR, centroid) > 0.01)) {
    return fail("support centroid is not safely inside the support triangle");
  }
  const go1::Vec2 outside{feet[0].x + 0.05, feet[0].y - 0.05};
  if (!(go1::signedSupportMargin(feet, go1::Leg::FR, outside) < 0.0)) {
    return fail("outside point was classified as supported");
  }

  std::cout << "go1 kinematics tests passed\n";
  return EXIT_SUCCESS;
}
