#pragma once

#include <array>
#include <cstddef>

namespace go1 {

constexpr std::size_t kLegCount = 4;

enum class Leg : std::size_t { FR = 0, FL = 1, RR = 2, RL = 3 };

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct JointAngles {
  double hip = 0.0;
  double thigh = 0.0;
  double calf = 0.0;
};

const char *legName(Leg leg);
int legIndex(Leg leg);
bool parseLeg(const char *text, Leg *leg);

class Kinematics {
public:
  // Geometry is taken from Unitree's go1_description xacro/const.xacro.
  static constexpr double kHipOffsetX = 0.1881;
  static constexpr double kHipOffsetY = 0.04675;
  static constexpr double kAbductionLink = 0.08;
  static constexpr double kThighLength = 0.213;
  static constexpr double kCalfLength = 0.213;

  static Vec3 hipOrigin(Leg leg);
  static Vec3 forward(Leg leg, const JointAngles &q);

  // Returns the knee-bent solution nearest seed. The result is rejected when
  // it leaves the conservative SDK joint limits or fails the FK round trip.
  static bool inverse(Leg leg, const Vec3 &footInTrunk,
                      const JointAngles &seed, JointAngles *result);
};

bool centerOfPressure(const std::array<Vec3, kLegCount> &feet,
                      const std::array<double, kLegCount> &verticalForce,
                      Vec2 *cop);

Vec2 supportCentroid(const std::array<Vec3, kLegCount> &feet,
                     Leg excluded);

// Positive inside the remaining three-foot support triangle, zero on its
// boundary, and negative outside. Units are metres.
double signedSupportMargin(const std::array<Vec3, kLegCount> &feet,
                           Leg excluded, const Vec2 &point);

double norm(const Vec2 &value);
Vec2 clampNorm(const Vec2 &value, double maximum);

} // namespace go1
