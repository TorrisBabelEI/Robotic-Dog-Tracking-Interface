#pragma once
// Data-only receive-event predicate. Does not construct SDK transport.
// Only receive-owned fields are copied: SendCount may change in another thread.
#include <cstdint>
namespace go1 {
struct SdkReceiveCounters {
  uint64_t received=0,flagErrors=0,crcErrors=0;
  template<class Counters> static SdkReceiveCounters snapshot(const Counters& s) {
    return {s.RecvCount,s.FlagError,s.RecvCRCError};
  }
};
inline bool validatedSdkReceive(int result,const SdkReceiveCounters& before,
                               const SdkReceiveCounters& after) {
  // Reset/wrap is not accepted as a fresh receive event. Duplicate packet ticks
  // are filtered by the caller before publishing a new acquisition timestamp.
  return result>=0 && after.received>before.received &&
    after.flagErrors==before.flagErrors && after.crcErrors==before.crcErrors;
}
}
