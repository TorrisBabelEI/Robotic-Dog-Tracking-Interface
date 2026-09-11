# Native Go1 standing capture: offline analysis

## Result and provenance

The uploaded capture is readable and its checksum matches the Pi and Ubuntu
archives. It contains approximately 10 seconds of stable **standing**, as
confirmed by the operator. The filename `native-prone.pcap` does not describe
the actual posture. No commands were sent during decoding.

- SHA-256: `a1d5d005c33a2aae89e36d5beb906735112cda483e4c1d109cd033f966212ff3`
- Pi archive: `/home/pi/Robotic-Dog-Tracking-Interface/logs/native-observe-P56RnBAT/`
- Ubuntu archive: `/home/aims/Yuxuan/Robotic-Dog-Tracking-Interface/logs/factory-observation/standing-CCwLT0uH/`
- Operator capture output: 15,000 packets captured; 77 kernel drops; exit 0.
  The packet cap ended acquisition before the requested 15 seconds.

## Packet validation

| Direction | Payload | Packets | CRC matches | Median capture gap | Largest capture gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| MCU `.10:8007` to Pi `.161:8008` | 820 bytes | 10,000 | 10,000 | 1.000 ms | 2.830 ms |
| Pi `.161:8008` to MCU `.10:8007` | 614 bytes | 5,000 | 5,000 | 2.000 ms | 2.055 ms |

These are captured traffic intervals, not guarantees of control-loop timing.
The state tick increments were 1 ms on 9,929 adjacent pairs, 2 ms on 55,
and 0 ms on 15. Capturing on `any`, kernel drops, and repeated ticks prevent
using this file to certify zero loss or unique feedback frequency.

## Decoder evidence and scope

The decoder is [decode_native_go1_pcap.py](../experiment/decode_native_go1_pcap.py).
Offsets and quantization were checked against `refineCmd`, `refineState`,
`crc32`, and `encryptCRC` in the bundled official SDK static library
`externals/unitree_legged_sdk/lib/cpp/amd64/libunitree_legged_sdk.a`.
The SDK submodule revision is `4539a6c10dfbc9781cea6fcb7d51bc6ddc6f71e1`.
The expanded public declarations are in
[comm.h](../externals/unitree_legged_sdk/include/unitree_legged_sdk/comm.h).
This is a profile-specific decoder, not a claim about every Go1 firmware.

All byte offsets below are zero-based within the UDP payload:

| Field | State | Command |
| --- | --- | --- |
| Motor array start / stride | 75 / 32 bytes | 22 / 27 bytes |
| Motor mode / q / dq | +0 uint8 / +1 float32 / +5 float32 | same |
| Torque | +11 int16, divide by 256 | +9 int16, divide by 256 |
| Kp / Kd | not decoded | +11 uint16 / 32; +13 uint16 / 16 |
| Motor temperature | +23 int8 | absent |
| IMU roll/pitch/yaw | float32 at 62/66/70 | absent |
| State tick | uint32 at 755 | absent |
| CRC stored / bytes covered | 803 / 0–799 | 610 / 0–607 |

The CRC polynomial is `0x04c11db7`, initial value `0xffffffff`, processing
little-endian 32-bit words MSB first. Command CRC additionally XORs
`0xedcab9de`, then swaps byte pairs (1,2), (0,3), (0,2). This reproduces all
5,000 recorded command CRCs; applying the plain state CRC rule to commands
would incorrectly reject them. CRC is not sender authentication. Final partial
words are outside CRC coverage; the 13 state bytes at 807–819 remain unparsed.
The decoder fails on an unexpected flow, layout, truncation, or CRC mismatch.

## Standing measurements

Maximum observed absolute joint speed: **0.0381 rad/s**. Maximum reported motor
temperature: **50 °C**. Median roll: **−0.02492 rad** (about −1.43°);
median pitch: **−0.01427 rad** (about −0.82°). Roll and pitch peak-to-peak ranges
were 0.001416 and 0.000884 rad. These support the operator's report of stable,
approximately level standing during this interval.

The following values are medians over the capture. Torque units follow the
SDK's N·m convention; `tau_est` is motor-reported estimation, not an external
calibrated torque measurement. Command feedforward torque is not total applied
torque: the position/velocity terms also contribute.

| Joint | Measured q (rad) | tau_est (N·m) | Command tau_ff (N·m) | Kp | Kd |
| --- | ---: | ---: | ---: | ---: | ---: |
| FR_0 | -0.024283 | 2.324 | 2.355 | 1 | 0.25 |
| FR_1 | 0.804111 | 0.641 | 0.688 | 1 | 0.5 |
| FR_2 | -1.626428 | 6.047 | 6.086 | 1 | 0.5 |
| FL_0 | 0.053167 | -2.668 | -2.688 | 1 | 0.25 |
| FL_1 | 0.808531 | 0.469 | 0.512 | 1 | 0.5 |
| FL_2 | -1.622997 | 5.711 | 5.766 | 1 | 0.5 |
| RR_0 | -0.021436 | 2.695 | 2.734 | 1 | 0.25 |
| RR_1 | 0.814647 | 0.566 | 0.617 | 1 | 0.5 |
| RR_2 | -1.633614 | 6.492 | 6.543 | 1 | 0.5 |
| RL_0 | 0.044508 | -2.695 | -2.715 | 1 | 0.25 |
| RL_1 | 0.830210 | 0.270 | 0.305 | 1 | 0.5 |
| RL_2 | -1.628689 | 6.121 | 6.164 | 1 | 0.5 |

## Consequence for the no-lift route

Factory standing in this recording uses Kp = 1, Kd = 0.25 on the hip ab/adduction
joints and 0.5 on the other joints, with substantial feedforward support torque.
Calf feedforward medians are approximately 5.77–6.54 N·m. This makes a
zero-feedforward position hold a materially different controller. A small
excitation amplitude must not be confused with the total support torque needed
to carry the body. These measurements alone do not justify copying factory
commands, gains, or selecting a custom torque limit.

The standing baseline is accepted for offline observation. It does not resolve
the prone calf command-limit conflict, establish a floor endpoint, or validate
custom-controller takeover. The next useful observation is the factory return
to prone followed by damping (experiment 2.1.8). This will show the command and
feedback changes near the floor, without external lifting or a custom motor
command. There is no need to repeat this standing capture.
