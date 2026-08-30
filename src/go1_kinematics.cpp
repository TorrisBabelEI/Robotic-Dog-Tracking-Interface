#include "go1_kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace go1 {
namespace {

constexpr double kHipMin = -1.047;
constexpr double kHipMax = 1.047;
constexpr double kThighMin = -0.663;
constexpr double kThighMax = 2.966;
constexpr double kCalfMin = -2.721;
constexpr double kCalfMax = -0.837;

double clamp(double value, double low, double high) {
  return std::min(std::max(value, low), high);
}

double squaredDistance(const Vec3 &left, const Vec3 &right) {
  const double dx = left.x - right.x;
  const double dy = left.y - right.y;
  const double dz = left.z - right.z;
  return dx * dx + dy * dy + dz * dz;
}

double wrappedDistance(double left, double right) {
  return std::fabs(std::atan2(std::sin(left - right), std::cos(left - right)));
}

double cross(const Vec2 &a, const Vec2 &b, const Vec2 &p) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

bool inJointLimits(const JointAngles &q) {
  return q.hip >= kHipMin && q.hip <= kHipMax && q.thigh >= kThighMin &&
         q.thigh <= kThighMax && q.calf >= kCalfMin && q.calf <= kCalfMax;
}

double sideSign(Leg leg) {
  return leg == Leg::FL || leg == Leg::RL ? 1.0 : -1.0;
}

double frontSign(Leg leg) {
  return leg == Leg::FR || leg == Leg::FL ? 1.0 : -1.0;
}

} // namespace

const char *legName(Leg leg) {
  switch (leg) {
  case Leg::FR:
    return "FR";
  case Leg::FL:
    return "FL";
  case Leg::RR:
    return "RR";
  case Leg::RL:
    return "RL";
  }
  return "UNKNOWN";
}

int legIndex(Leg leg) { return static_cast<int>(leg); }

bool parseLeg(const char *text, Leg *leg) {
  if (text == nullptr || leg == nullptr) {
    return false;
  }
  for (std::size_t index = 0; index < kLegCount; ++index) {
    const Leg candidate = static_cast<Leg>(index);
    if (std::strcmp(text, legName(candidate)) == 0) {
      *leg = candidate;
      return true;
    }
  }
  return false;
}

Vec3 Kinematics::hipOrigin(Leg leg) {
  return {frontSign(leg) * kHipOffsetX, sideSign(leg) * kHipOffsetY,
          0.0};
}

Vec3 Kinematics::forward(Leg leg, const JointAngles &q) {
  const double side = sideSign(leg);
  const double planarX = -kThighLength * std::sin(q.thigh) -
                         kCalfLength * std::sin(q.thigh + q.calf);
  const double planarZ = -kThighLength * std::cos(q.thigh) -
                         kCalfLength * std::cos(q.thigh + q.calf);
  const double localY = side * kAbductionLink;
  const double cosHip = std::cos(q.hip);
  const double sinHip = std::sin(q.hip);
  const Vec3 origin = hipOrigin(leg);
  return {origin.x + planarX, origin.y + cosHip * localY - sinHip * planarZ,
          sinHip * localY + cosHip * planarZ};
}

bool Kinematics::inverse(Leg leg, const Vec3 &footInTrunk,
                         const JointAngles &seed, JointAngles *result) {
  if (result == nullptr || !std::isfinite(footInTrunk.x) ||
      !std::isfinite(footInTrunk.y) || !std::isfinite(footInTrunk.z)) {
    return false;
  }

  const Vec3 origin = hipOrigin(leg);
  const double x = footInTrunk.x - origin.x;
  const double y = footInTrunk.y - origin.y;
  const double z = footInTrunk.z;
  const double side = sideSign(leg);
  const double yzRadius = std::hypot(y, z);
  if (yzRadius <= kAbductionLink + 1.0e-7) {
    return false;
  }

  const double theta = std::atan2(z, y);
  const double alpha =
      std::acos(clamp(side * kAbductionLink / yzRadius, -1.0, 1.0));
  const std::array<double, 2> hipCandidates{{theta + alpha, theta - alpha}};

  bool found = false;
  JointAngles best;
  double bestCost = std::numeric_limits<double>::infinity();
  for (double hip : hipCandidates) {
    hip = std::atan2(std::sin(hip), std::cos(hip));
    const double sinHip = std::sin(hip);
    const double cosHip = std::cos(hip);
    const double planarZ = -sinHip * y + cosHip * z;
    const double radiusSquared = x * x + planarZ * planarZ;
    const double cosCalf = clamp(
        (radiusSquared - kThighLength * kThighLength -
         kCalfLength * kCalfLength) /
            (2.0 * kThighLength * kCalfLength),
        -1.0, 1.0);
    const double calf = -std::acos(cosCalf);
    const double thetaPlanar = std::atan2(-x, -planarZ);
    const double thigh =
        thetaPlanar -
        std::atan2(kCalfLength * std::sin(calf),
                   kThighLength + kCalfLength * std::cos(calf));
    const JointAngles candidate{hip, thigh, calf};
    if (!inJointLimits(candidate)) {
      continue;
    }
    const Vec3 reconstructed = forward(leg, candidate);
    if (squaredDistance(reconstructed, footInTrunk) > 1.0e-10) {
      continue;
    }
    const double cost = wrappedDistance(candidate.hip, seed.hip) +
                        wrappedDistance(candidate.thigh, seed.thigh) +
                        wrappedDistance(candidate.calf, seed.calf);
    if (cost < bestCost) {
      bestCost = cost;
      best = candidate;
      found = true;
    }
  }
  if (found) {
    *result = best;
  }
  return found;
}

bool centerOfPressure(const std::array<Vec3, kLegCount> &feet,
                      const std::array<double, kLegCount> &verticalForce,
                      Vec2 *cop) {
  if (cop == nullptr) {
    return false;
  }
  double total = 0.0;
  Vec2 weighted;
  for (std::size_t index = 0; index < kLegCount; ++index) {
    const double force = verticalForce[index];
    if (!std::isfinite(force) || force < 0.0) {
      return false;
    }
    total += force;
    weighted.x += force * feet[index].x;
    weighted.y += force * feet[index].y;
  }
  if (total <= 1.0e-9) {
    return false;
  }
  cop->x = weighted.x / total;
  cop->y = weighted.y / total;
  return std::isfinite(cop->x) && std::isfinite(cop->y);
}

Vec2 supportCentroid(const std::array<Vec3, kLegCount> &feet, Leg excluded) {
  Vec2 centroid;
  for (std::size_t index = 0; index < kLegCount; ++index) {
    if (index == static_cast<std::size_t>(excluded)) {
      continue;
    }
    centroid.x += feet[index].x / 3.0;
    centroid.y += feet[index].y / 3.0;
  }
  return centroid;
}

double signedSupportMargin(const std::array<Vec3, kLegCount> &feet,
                           Leg excluded, const Vec2 &point) {
  std::array<Vec2, 3> triangle;
  std::size_t output = 0;
  for (std::size_t index = 0; index < kLegCount; ++index) {
    if (index != static_cast<std::size_t>(excluded)) {
      triangle[output++] = {feet[index].x, feet[index].y};
    }
  }
  const double twiceArea = cross(triangle[0], triangle[1], triangle[2]);
  if (std::fabs(twiceArea) < 1.0e-9) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (twiceArea < 0.0) {
    std::swap(triangle[1], triangle[2]);
  }

  double margin = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < 3; ++index) {
    const Vec2 &a = triangle[index];
    const Vec2 &b = triangle[(index + 1) % 3];
    const double edgeLength = std::hypot(b.x - a.x, b.y - a.y);
    if (edgeLength < 1.0e-9) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    margin = std::min(margin, cross(a, b, point) / edgeLength);
  }
  return margin;
}

double norm(const Vec2 &value) { return std::hypot(value.x, value.y); }

Vec2 clampNorm(const Vec2 &value, double maximum) {
  const double length = norm(value);
  if (length <= maximum || length <= 1.0e-12) {
    return value;
  }
  const double scale = maximum / length;
  return {value.x * scale, value.y * scale};
}

} // namespace go1
