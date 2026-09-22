#pragma once
// Data-only SDK-to-worker conversion. Including comm.h opens no SDK transport.
// Caller must validate acquisition/CRC and supply the matching monotonic epoch.
#include "unitree_legged_sdk/comm.h"
#include <array>
#include <cmath>
#include <stdexcept>
namespace go1 {
inline std::array<double,58> sdkPolicySensors(const UNITREE_LEGGED_SDK::LowState& s,
                                            bool acquisitionValidated=false) {
  if(!acquisitionValidated || s.levelFlag!=0xff)
    throw std::runtime_error("unvalidated low-level acquisition");
  std::array<double,58> a{};
  for(unsigned i=0;i<12;++i){a[i]=s.motorState[i].q;a[12+i]=s.motorState[i].dq;a[37+i]=s.motorState[i].tauEst;}
  for(unsigned i=0;i<3;++i){a[24+i]=s.imu.gyroscope[i];a[27+i]=s.imu.accelerometer[i];a[30+i]=s.imu.rpy[i];}
  for(unsigned i=0;i<4;++i)a[33+i]=s.footForce[i]; // footForceEst is reserved, not measured force.
  double norm2=0;for(float q:s.imu.quaternion)norm2+=double(q)*q;
  const double norm=std::sqrt(norm2);
  if(!std::isfinite(norm)||norm<.95||norm>1.05)throw std::runtime_error("invalid SDK quaternion norm");
  const double w=s.imu.quaternion[0]/norm,x=s.imu.quaternion[1]/norm,
               y=s.imu.quaternion[2]/norm,z=s.imu.quaternion[3]/norm;
  const double r[]={1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                    2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                    2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
  for(unsigned i=0;i<9;++i)a[49+i]=r[i];
  for(double v:a)if(!std::isfinite(v))throw std::runtime_error("nonfinite SDK policy input");
  return a;
}
} // namespace go1
