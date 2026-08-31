# Robotic-Dog-Tracking-Interface
This GitHub repo is a tutorial and a waypoint/trajectory tracking interface for [Unitree Go1 Robot](https://shop.unitree.com/), please refer to official [sdk](https://github.com/unitreerobotics/unitree_legged_sdk/tree/go1) for details. 
* The tutorial includes the setup of the robot, network connections, low-level-control and high-level-control examples.
* Robotic-Dog-Tracking-Interface is able to obtain state estimation from motion capture systems [Qualisys](https://www.qualisys.com/), and you're welcome to write functions for your own motion capture system. The purpose of this repository is that prople can foucs on motion planning / path planning / trajectory planning and **Robotic-Dog--Tracking-Interface** can take over actual waypoint/trajectory tracking as long as your planner outputs waypoints/trajectory as csv files in a specific directory.

**Future updates**
* Direct low-level-control examples
* Trajectory tracking

Dependencies
============
This repo has been tested with:
* Ubuntu 20.04 LTS, Python 3.8
* Raspberry Pi

Requuired packages:
```
$ sudo apt install cmake libmsgpack* libboost-all-dev
```

Build
=====
To download and build this repo,
```
$ git clone https://github.com/tianyuzhou-sam/Robotic-Dog-Tracking-Interface.git
$ cd <MAIN_DIRECTORY>
$ git submodule update --init --recursive
$ cd <MAIN_DIRECTORY>
$ mkdir build
$ cd build
$ cmake ..
$ make
```

If you want to build the python wrapper, then replace the cmake line `cmake ..` with:
```
$ cmake -DPYTHON_BUILD=TRUE ..
```

Connection
==========
**1. To run on the robot's system (raspberry pi):** 

First connect your computer to the robot's wifi `Unitree_Go429210A` (5 GHz only) with password `00000000`. Then open a new terminal
```
$ ssh <USERNAME>@192.168.12.1
```
```
$ ssh pi@192.168.12.1
```
Enter the password and then you are in the robot's system. The default password is `123`.


**2. To run on network device, a bridge is needed:**
On the robot's system (in the same terminal that you just ssh-ed):
```
sudo sysctl -p
sudo iptables -F
sudo iptables -t nat -F
sudo iptables -t nat -A POSTROUTING -o wlan1 -j MASQUERADE
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
sudo iptables -A FORWARD -i wlan1 -o eth0 -j ACCEPT
sudo iptables -A FORWARD -i eth0 -o wlan1 -j ACCEPT

```
On network device (Open a new terminal in your computer):
```
$ sudo route add default gw 192.168.12.1
```


Example
=======

The same computer needs to connect with the motion capture system with Ethernet cables.

To run an open-loop walking example:
```
$ cd build
$ ./example_walk
```


To run NMPC waypoint tracking with the motion capture system:

```
$ python3 experiment/run_waypoints.py
```

Ground low-level experiment
===========================

The repository includes a conservative 500 Hz C++ experiment runner with
remote-stop handling, command watchdog, foot kinematics, force/CoP support
gates, and an offline Python analyzer. Non-Linux systems build a dry-run-only
executable; Unitree hardware access must be built and run on Linux.

```bash
cmake -S . -B build
cmake --build build --target go1_lowlevel_experiment

./build/go1_lowlevel_experiment --dry-run --mode leg-lift \
  --leg auto --lift-height-m 0.02 --tau-overlay-nm 0.10 \
  --tau-overlay-hz 0.5 --log /tmp/go1_leg_lift_dry.csv

python3 experiment/analyze_lowlevel_log.py /tmp/go1_leg_lift_dry.csv
```

The staged ground path is an explicitly confirmed prone-damping remote
preflight, standing handover, squat, one auto-selected leg, then the four-leg
sequence. The preflight actively sends zero-torque damping so the SDK/NAT
return path is established. Leg lifting uses position impedance plus a small
torque overlay; single-joint pure torque still requires a load-bearing stand.
See
[docs/GO1_LOWLEVEL_EXPERIMENT.md](docs/GO1_LOWLEVEL_EXPERIMENT.md) for stop
semantics, safety gates, commands, log schema, and the MOCAP boundary.
