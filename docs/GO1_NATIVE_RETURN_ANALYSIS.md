# Go1 factory return-to-prone capture: offline analysis

## Result and provenance

The supplied archive is readable and internally consistent. The PCAP checksum
matches the value reported after the Ubuntu copy. Re-decoding the PCAP with the
repository decoder reproduced the supplied `summary.json` byte for byte. No
commands were sent to the robot during this analysis.

- ZIP SHA-256:
  `c91a06c556022f059d7d438e734bc2d71e47ce60bddd0a19c71c226d25e5d041`
- PCAP SHA-256:
  `91271ff4c5bb79d85082a5c83d4d93e095157c21766c4fdd89b79d4e46991b8e`
- Ubuntu archive:
  `/home/aims/Yuxuan/Robotic-Dog-Tracking-Interface/logs/factory-observation/return-CxCtzXzf/`
- Capture report: 59,788 packets captured, 60,015 received by the filter,
  and zero kernel drops.

The decoder accepted exactly the 59,788 captured packets. All 39,859 state
packets and 19,929 command packets passed their profile-specific SDK CRC check.
The state and command spans were 39.858 and 39.856 seconds, with median capture
gaps of 1 and 2 ms. These are capture observations, not guarantees about the
controller's scheduling or transport loss.

## Reconstructed sequence

Times below are relative to the first decoded state packet. They are inferred
from packet fields rather than from the remote buttons, which this decoder does
not decode.

| Time | Packet evidence | Interpretation |
| ---: | --- | --- |
| 0.000–9.789 s | Quiet standing feedback; commands use `Kp=50`, `Kd=3` | Stable initial factory-controlled pose |
| 9.789–12.829 s | Symmetric thigh/calf command targets change | Factory lie-down trajectory |
| 12.829–15.167 s | Targets held at `(0, 1.4, -2.58)` rad per leg | Controlled folded-pose hold |
| 15.169 s | All targets and `Kp` become zero; `Kd` becomes 2 | Factory switch to position-free damping |
| 15.177–16.087 s | Joint speed remains above 0.1 rad/s | Large post-switch settling motion |
| through 21.208 s | Isolated joint speed remains above 0.05 rad/s | Smaller residual settling |
| final 18.65 s | `Kp=0`, `Kd=2`; near-zero torque and quiet feedback | Stable damping endpoint |

The largest speed in the whole capture was 2.239 rad/s during the controlled
lie-down. After the damping switch, the largest speed was 1.845 rad/s on RL_0,
at 15.341 s. This confirms that the factory operation did not simply change
gains after an already motionless endpoint: releasing position stiffness caused
substantial additional joint motion. It does not, by itself, identify the
trunk's contact state or whether that motion was visually acceptable.

## Measured endpoint

The final values below are medians over 35–40 seconds. Every joint's range in
that window was at most 0.000182 rad; median roll and pitch were -0.01648 and
-0.01024 rad. Their peak-to-peak ranges were 0.001326 and 0.000672 rad. Maximum
reported motor temperature was 36 degrees C.

| Joint | Final q (rad) | Existing development target (rad) | Difference (rad) |
| --- | ---: | ---: | ---: |
| FR_0 | -0.311616 | -0.280000 | -0.031616 |
| FR_1 | 1.275168 | 1.250000 | 0.025168 |
| FR_2 | -2.794212 | -2.700000 | -0.094212 |
| FL_0 | 0.289090 | 0.280000 | 0.009090 |
| FL_1 | 1.272262 | 1.250000 | 0.022262 |
| FL_2 | -2.797724 | -2.700000 | -0.097724 |
| RR_0 | -0.289514 | -0.280000 | -0.009514 |
| RR_1 | 1.280437 | 1.250000 | 0.030437 |
| RR_2 | -2.799541 | -2.700000 | -0.099541 |
| RL_0 | 0.302533 | 0.280000 | 0.022533 |
| RL_1 | 1.237503 | 1.250000 | -0.012497 |
| RL_2 | -2.768375 | -2.700000 | -0.068375 |

The calves again settle beyond the SDK position-command minimum of -2.721 rad.
These are valid passive feedback measurements, not valid position commands.
Do not widen the command bounds or replay the factory packet stream. The
existing -2.70 rad development target remains uncalibrated: its 0.068–0.100 rad
calf difference from this passive endpoint means that reaching that target
cannot be treated as evidence of belly contact.

## Recovered operator observation and acceptance

`observation.txt` contains the three shell commands that followed its
interactive `read`, indicating that a pasted command block was consumed as the
answer. The operator subsequently supplied the missing observation from memory:

```text
Initial posture: Standing; L2+A used: yes; trunk fully on floor before L2+B: yes; motion after damping: nothing but lie flat on the ground; Sound: None
```

This confirms that the trunk was fully on the floor before the damping command,
that no motion beyond settling flat was observed afterward, and that there was
no sound. Slip and impact were not listed as separate events, and no abnormal
event was reported. Combined with the independently verified telemetry, this
completes the physical-observation gate in section 2.1.8. Do not repeat the
robot motion merely to repair the archived text file.

## Consequence for the next controller

The first custom no-lift path must not copy the factory's abrupt transition from
full position stiffness to damping. The new dry-run-only `prone-low-rise` path
therefore begins prone, engages bounded impedance gradually, requests a 5 mm
symmetric rise, returns to its clamped engagement pose, requires continuous
independent support observation, and releases stiffness gradually. The existing
standing `ground-handover` path and the new path's hardware entry remain locked.
Software and synthetic-support tests in section 2.1.9 come before any new
hardware command.
