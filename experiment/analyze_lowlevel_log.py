#!/usr/bin/env python3
"""Analyze CSV logs produced by go1_lowlevel_experiment.

The script intentionally stays offline: it never imports robot_interface and
cannot send commands to a Go1.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    print(
        "error: NumPy is required for log analysis. Install the project's "
        "Python dependencies before running this offline tool.",
        file=sys.stderr,
    )
    raise SystemExit(2)


JOINTS: Tuple[str, ...] = (
    "FR_0", "FR_1", "FR_2",
    "FL_0", "FL_1", "FL_2",
    "RR_0", "RR_1", "RR_2",
    "RL_0", "RL_1", "RL_2",
)

ACTION_PHASES: Tuple[str, ...] = (
    "GROUND_HANDOVER",
    "TORQUE_EXCITE",
    "SQUAT",
    "WEIGHT_SHIFT",
    "LIFT",
    "AIR_HOLD",
    "LOWER",
    "CONTACT_VERIFY",
    "RECENTER",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a Go1 low-level experiment CSV without robot access."
    )
    parser.add_argument("log", type=Path, help="CSV produced by go1_lowlevel_experiment")
    parser.add_argument(
        "--joint",
        action="append",
        choices=JOINTS,
        help="Joint to analyze; repeat this option. Defaults to all joints.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary CSV path (default: <log>.summary.csv)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Plot directory (default: <log stem>_plots next to the log)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Compute metrics without matplotlib output"
    )
    return parser.parse_args()


def load_log(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"log does not exist: {path}")

    with path.open("r", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        fieldnames = list(reader.fieldnames)
        columns: Dict[str, List[object]] = {name: [] for name in fieldnames}
        row_count = 0
        text_columns = {"phase", "abort_reason", "stop_source", "active_leg"}
        for row in reader:
            row_count += 1
            for column in fieldnames:
                value = row[column]
                if column in text_columns:
                    columns[column].append(value)
                else:
                    try:
                        columns[column].append(float(value))
                    except (TypeError, ValueError):
                        columns[column].append(math.nan)

    if row_count == 0:
        raise ValueError("CSV has no samples")

    data: Dict[str, np.ndarray] = {}
    for column, values in columns.items():
        data[column] = np.asarray(
            values, dtype=str if column in text_columns else float
        )
    return data


def require_columns(data: Dict[str, np.ndarray], columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError("missing expected log columns: " + ", ".join(missing))


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size < 3 or np.std(left) < 1.0e-9 or np.std(right) < 1.0e-9:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def linear_gain(command: np.ndarray, estimate: np.ndarray) -> float:
    mask = np.isfinite(command) & np.isfinite(estimate)
    command = command[mask]
    estimate = estimate[mask]
    if command.size < 3:
        return math.nan
    command = command - np.mean(command)
    estimate = estimate - np.mean(estimate)
    denominator = float(np.dot(command, command))
    if denominator < 1.0e-12:
        return math.nan
    return float(np.dot(command, estimate) / denominator)


def lag_by_correlation(
    command: np.ndarray, estimate: np.ndarray, sample_dt_s: float, max_lag_s: float = 0.5
) -> Tuple[float, float]:
    mask = np.isfinite(command) & np.isfinite(estimate)
    command = command[mask]
    estimate = estimate[mask]
    if command.size < 10 or sample_dt_s <= 0:
        return math.nan, math.nan
    command = command - np.mean(command)
    estimate = estimate - np.mean(estimate)
    if np.std(command) < 1.0e-9 or np.std(estimate) < 1.0e-9:
        return math.nan, math.nan

    max_samples = min(int(max_lag_s / sample_dt_s), command.size // 3)
    best_lag = 0
    best_corr = -math.inf
    for lag in range(-max_samples, max_samples + 1):
        if lag < 0:
            left = command[-lag:]
            right = estimate[: estimate.size + lag]
        elif lag > 0:
            left = command[: command.size - lag]
            right = estimate[lag:]
        else:
            left = command
            right = estimate
        corr = safe_correlation(left, right)
        if math.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag
    if not math.isfinite(best_corr):
        return math.nan, math.nan
    return best_lag * sample_dt_s, best_corr


def dominant_frequency_response(
    command: np.ndarray, estimate: np.ndarray, sample_dt_s: float
) -> Tuple[float, float, float]:
    mask = np.isfinite(command) & np.isfinite(estimate)
    command = command[mask]
    estimate = estimate[mask]
    if command.size < 16 or sample_dt_s <= 0:
        return math.nan, math.nan, math.nan
    command = command - np.mean(command)
    estimate = estimate - np.mean(estimate)
    spectrum_cmd = np.fft.rfft(command)
    spectrum_est = np.fft.rfft(estimate)
    frequencies = np.fft.rfftfreq(command.size, sample_dt_s)
    if spectrum_cmd.size <= 1:
        return math.nan, math.nan, math.nan
    index = int(np.argmax(np.abs(spectrum_cmd[1:])) + 1)
    if abs(spectrum_cmd[index]) < 1.0e-12:
        return math.nan, math.nan, math.nan
    response = spectrum_est[index] / spectrum_cmd[index]
    return (
        float(frequencies[index]),
        float(abs(response)),
        float(np.angle(response, deg=True)),
    )


def network_metrics(
    data: Dict[str, np.ndarray], action_mask: np.ndarray
) -> Dict[str, float]:
    host_s = data["host_monotonic_ns"] * 1.0e-9
    recv_mask = data["recv_ok"] > 0.5
    recv_times = host_s[recv_mask & np.isfinite(host_s)]
    elapsed = float(host_s[-1] - host_s[0]) if host_s.size > 1 else math.nan
    gaps = np.diff(recv_times)
    hold_indices = np.flatnonzero(data["phase"] == "HOLD")
    if hold_indices.size and np.any(action_mask):
        roll_reference = data["imu_roll"][hold_indices[0]]
        pitch_reference = data["imu_pitch"][hold_indices[0]]
        roll_excursion = float(
            np.nanmax(np.abs(data["imu_roll"][action_mask] - roll_reference))
        )
        pitch_excursion = float(
            np.nanmax(np.abs(data["imu_pitch"][action_mask] - pitch_reference))
        )
    else:
        roll_excursion = math.nan
        pitch_excursion = math.nan

    temperatures = np.concatenate([data[f"{joint}_temperature"] for joint in JOINTS])
    speeds = np.concatenate([data[f"{joint}_state_dq"] for joint in JOINTS])
    remote_valid_ratio = math.nan
    remote_l2_b_seen = math.nan
    if "remote_valid" in data and "remote_buttons" in data:
        fresh_count = int(np.count_nonzero(recv_mask))
        if fresh_count:
            remote_valid = data["remote_valid"] > 0.5
            remote_valid_ratio = float(np.count_nonzero(recv_mask & remote_valid) /
                                       fresh_count)
            buttons = np.nan_to_num(data["remote_buttons"], nan=0.0).astype(np.int64)
            required = (1 << 5) | (1 << 9)
            remote_l2_b_seen = float(np.any(
                recv_mask & remote_valid & ((buttons & required) == required)
            ))
    return {
        "samples": float(host_s.size),
        "elapsed_s": elapsed,
        "feedback_rate_hz": (
            float(max(recv_times.size - 1, 0) / elapsed)
            if math.isfinite(elapsed) and elapsed > 0
            else math.nan
        ),
        "feedback_gap_p99_ms": (
            float(np.percentile(gaps, 99) * 1000.0) if gaps.size else math.nan
        ),
        "feedback_gap_max_ms": float(np.max(gaps) * 1000.0) if gaps.size else math.nan,
        "loop_dt_p99_ms": float(np.nanpercentile(data["loop_dt_us"], 99) / 1000.0),
        "loop_dt_max_ms": float(np.nanmax(data["loop_dt_us"]) / 1000.0),
        "imu_roll_excursion_rad": roll_excursion,
        "imu_pitch_excursion_rad": pitch_excursion,
        "max_motor_temperature_c": float(np.nanmax(temperatures)),
        "max_abs_joint_speed_rad_s": float(np.nanmax(np.abs(speeds))),
        "remote_valid_fresh_ratio": remote_valid_ratio,
        "remote_l2_b_seen": remote_l2_b_seen,
    }


def ground_support_metrics(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    defaults = {
        "min_active_support_margin_m": math.nan,
        "min_airborne_force_ratio": math.nan,
        "final_contact_force_ratio": math.nan,
        "remote_estop_latency_ms": math.nan,
        "watchdog_cycles": 0.0,
    }
    optional = {
        "active_leg",
        "stop_source",
        "stop_request_ns",
        "damping_command_ns",
        "watchdog_active",
    }
    if not optional.issubset(data):
        return defaults

    airborne = np.isin(data["phase"], ("LIFT", "AIR_HOLD", "LOWER"))
    margins: List[float] = []
    force_ratios: List[float] = []
    for leg_index, leg in enumerate(("FR", "FL", "RR", "RL")):
        leg_mask = airborne & (data["active_leg"] == leg)
        margin_column = f"support_margin_{leg}_m"
        baseline_column = f"foot_force_baseline_{leg_index}"
        force_column = f"foot_force_{leg_index}"
        if margin_column in data:
            margins.extend(data[margin_column][leg_mask].tolist())
        if baseline_column in data and force_column in data:
            baseline = data[baseline_column][leg_mask]
            force = data[force_column][leg_mask]
            valid = np.isfinite(baseline) & (baseline > 0) & np.isfinite(force)
            force_ratios.extend((force[valid] / baseline[valid]).tolist())

    contact_ratios: List[float] = []
    contact = data["phase"] == "CONTACT_VERIFY"
    for leg_index, leg in enumerate(("FR", "FL", "RR", "RL")):
        leg_mask = contact & (data["active_leg"] == leg)
        baseline_column = f"foot_force_baseline_{leg_index}"
        force_column = f"foot_force_{leg_index}"
        if baseline_column not in data or force_column not in data:
            continue
        baseline = data[baseline_column][leg_mask]
        force = data[force_column][leg_mask]
        valid = np.isfinite(baseline) & (baseline > 0) & np.isfinite(force)
        if np.any(valid):
            contact_ratios.append(float((force[valid] / baseline[valid])[-1]))

    stop_mask = (data["stop_source"] == "remote_l2_b") & (
        data["damping_command_ns"] > 0
    )
    latency_ms = math.nan
    if np.any(stop_mask):
        request = data["stop_request_ns"][stop_mask][0]
        damping = data["damping_command_ns"][stop_mask][0]
        latency_ms = float((damping - request) * 1.0e-6)

    finite_margins = np.asarray(margins, dtype=float)
    finite_margins = finite_margins[np.isfinite(finite_margins)]
    finite_ratios = np.asarray(force_ratios, dtype=float)
    finite_ratios = finite_ratios[np.isfinite(finite_ratios)]
    return {
        "min_active_support_margin_m": (
            float(np.min(finite_margins)) if finite_margins.size else math.nan
        ),
        "min_airborne_force_ratio": (
            float(np.min(finite_ratios)) if finite_ratios.size else math.nan
        ),
        "final_contact_force_ratio": (
            float(np.min(contact_ratios)) if contact_ratios else math.nan
        ),
        "remote_estop_latency_ms": latency_ms,
        "watchdog_cycles": float(np.count_nonzero(data["watchdog_active"] > 0.5)),
    }


def joint_metrics(
    data: Dict[str, np.ndarray], joint: str, action_mask: np.ndarray, sample_dt_s: float
) -> Dict[str, object]:
    command = data[f"{joint}_tau_cmd_total"][action_mask]
    feedforward = data[f"{joint}_tau_ff"][action_mask]
    estimate = data[f"{joint}_tau_est"][action_mask]
    error = estimate - command
    finite_error = error[np.isfinite(error)]
    lag_s, lag_corr = lag_by_correlation(command, estimate, sample_dt_s)
    frequency_hz, frequency_gain, phase_deg = dominant_frequency_response(
        command, estimate, sample_dt_s
    )

    cmd_q = data[f"{joint}_cmd_q"]
    state_q = data[f"{joint}_state_q"]
    kp = data[f"{joint}_cmd_kp"]
    position_mask = action_mask & (cmd_q < 1.0e8) & (kp > 0)
    position_error = state_q[position_mask] - cmd_q[position_mask]
    position_error = position_error[np.isfinite(position_error)]

    safe_stop_mask = np.isin(data["phase"], ("SAFE_STOP", "SAFE_HOLD"))
    return_error = math.nan
    if np.any(safe_stop_mask):
        initial_candidates = state_q[data["phase"] == "HOLD"]
        final_candidates = state_q[safe_stop_mask]
        if initial_candidates.size and final_candidates.size:
            return_error = float(final_candidates[-1] - initial_candidates[0])

    total_correlation = safe_correlation(command, estimate)
    command_has_variation = (
        command.size > 0
        and np.any(np.isfinite(command))
        and float(np.nanstd(command)) > 1.0e-9
    )
    return {
        "joint": joint,
        "samples": int(command.size),
        "tau_bias_nm": float(np.mean(finite_error)) if finite_error.size else math.nan,
        "tau_rmse_nm": (
            float(np.sqrt(np.mean(finite_error ** 2))) if finite_error.size else math.nan
        ),
        "tau_corr_total": total_correlation,
        "tau_corr_feedforward": safe_correlation(feedforward, estimate),
        "tau_gain_total": linear_gain(command, estimate),
        "lag_s": lag_s,
        "lag_corr": lag_corr,
        "dominant_frequency_hz": frequency_hz,
        "frequency_gain": frequency_gain,
        "frequency_phase_deg": phase_deg,
        "position_rms_rad": (
            float(np.sqrt(np.mean(position_error ** 2)))
            if position_error.size
            else math.nan
        ),
        "return_error_rad": return_error,
        "max_abs_tau_ff_nm": float(np.nanmax(np.abs(feedforward))) if feedforward.size else 0.0,
        "max_abs_tau_est_nm": float(np.nanmax(np.abs(estimate))) if estimate.size else math.nan,
        "go_no_go_corr_pass": bool(
            command_has_variation
            and math.isfinite(total_correlation)
            and total_correlation >= 0.8
        ),
    }


def write_summary(
    path: Path, metrics: Sequence[Dict[str, object]], network: Dict[str, float],
    support: Dict[str, float], abort_reasons: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = list(metrics[0].keys()) if metrics else ["joint"]
    extra_fields = [
        "feedback_rate_hz",
        "feedback_gap_p99_ms",
        "feedback_gap_max_ms",
        "loop_dt_p99_ms",
        "loop_dt_max_ms",
        "imu_roll_excursion_rad",
        "imu_pitch_excursion_rad",
        "max_motor_temperature_c",
        "max_abs_joint_speed_rad_s",
        "remote_valid_fresh_ratio",
        "remote_l2_b_seen",
        "min_active_support_margin_m",
        "min_airborne_force_ratio",
        "final_contact_force_ratio",
        "remote_estop_latency_ms",
        "watchdog_cycles",
        "abort_reasons",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields + extra_fields)
        writer.writeheader()
        for metric in metrics:
            row = dict(metric)
            row.update(
                {
                    key: network.get(key, support.get(key, math.nan))
                    for key in extra_fields[:-1]
                }
            )
            row["abort_reasons"] = ";".join(abort_reasons)
            writer.writerow(row)


def make_plots(
    data: Dict[str, np.ndarray], joints: Sequence[str], output_dir: Path
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"warning: plots skipped because matplotlib is unavailable: {error}", file=sys.stderr)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    time_s = (data["host_monotonic_ns"] - data["host_monotonic_ns"][0]) * 1.0e-9

    for joint in joints:
        figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(time_s, data[f"{joint}_tau_ff"], label="tau_ff", linewidth=1)
        axes[0].plot(
            time_s, data[f"{joint}_tau_cmd_total"], label="tau_cmd_total", linewidth=1
        )
        axes[0].plot(time_s, data[f"{joint}_tau_est"], label="tauEst", linewidth=1)
        axes[0].set_ylabel("torque [Nm]")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.25)

        valid_q = data[f"{joint}_cmd_q"].copy()
        valid_q[valid_q > 1.0e8] = np.nan
        axes[1].plot(time_s, valid_q, label="q_des", linewidth=1)
        axes[1].plot(time_s, data[f"{joint}_state_q"], label="q", linewidth=1)
        axes[1].set_ylabel("position [rad]")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(time_s, data[f"{joint}_state_dq"], label="dq", linewidth=1)
        axes[2].set_ylabel("velocity [rad/s]")
        axes[2].set_xlabel("time [s]")
        axes[2].grid(True, alpha=0.25)
        figure.suptitle(joint)
        figure.tight_layout()
        figure.savefig(output_dir / f"{joint}.png", dpi=150)
        plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time_s, data["loop_dt_us"] / 1000.0)
    axes[0].axhline(2.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("loop dt [ms]")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time_s, data["imu_roll"], label="roll")
    axes[1].plot(time_s, data["imu_pitch"], label="pitch")
    axes[1].plot(time_s, data["imu_yaw"], label="yaw")
    axes[1].set_ylabel("IMU [rad]")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)
    for index in range(4):
        axes[2].plot(time_s, data[f"foot_force_{index}"], label=f"foot {index}")
    axes[2].set_ylabel("foot force [raw]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "timing_imu_foot_force.png", dpi=150)
    plt.close(figure)

    support_columns = [f"support_margin_{leg}_m" for leg in ("FR", "FL", "RR", "RL")]
    if {"cop_x_m", "cop_y_m", *support_columns}.issubset(data):
        figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(time_s, data["cop_x_m"], label="CoP x")
        axes[0].plot(time_s, data["cop_y_m"], label="CoP y")
        axes[0].set_ylabel("CoP [m]")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.25)
        for leg, column in zip(("FR", "FL", "RR", "RL"), support_columns):
            axes[1].plot(time_s, data[column], label=f"exclude {leg}")
        event_styles = {
            "LIFT": ("tab:green", "lift starts"),
            "AIR_HOLD": ("tab:orange", "air hold"),
            "CONTACT_VERIFY": ("tab:purple", "touchdown check"),
            "RECENTER": ("tab:brown", "contact confirmed"),
        }
        previous = ""
        for index, phase in enumerate(data["phase"]):
            if phase in event_styles and phase != previous:
                color, label = event_styles[phase]
                leg = data["active_leg"][index] if "active_leg" in data else ""
                event_label = f"{label} {leg}".strip()
                for axis in axes:
                    axis.axvline(
                        time_s[index], color=color, linestyle=":", linewidth=0.8,
                        alpha=0.75, label=event_label if axis is axes[1] else None,
                    )
            previous = phase
        axes[1].axhline(0.010, color="black", linestyle="--", linewidth=1)
        axes[1].set_ylabel("support margin [m]")
        axes[1].set_xlabel("time [s]")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "cop_support_margin.png", dpi=150)
        plt.close(figure)
    return True


def main() -> int:
    args = parse_args()
    data = load_log(args.log)
    required = [
        "host_monotonic_ns",
        "state_tick_ms",
        "phase",
        "recv_ok",
        "loop_dt_us",
        "abort_reason",
        "imu_roll",
        "imu_pitch",
        "imu_yaw",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accel_x",
        "accel_y",
        "accel_z",
        "foot_force_0",
        "foot_force_1",
        "foot_force_2",
        "foot_force_3",
    ]
    for joint in JOINTS:
        required.extend(
            [
                f"{joint}_tau_ff",
                f"{joint}_tau_cmd_total",
                f"{joint}_tau_est",
                f"{joint}_cmd_q",
                f"{joint}_cmd_kp",
                f"{joint}_state_q",
                f"{joint}_state_dq",
                f"{joint}_temperature",
            ]
        )
    require_columns(data, required)

    joints: List[str] = args.joint or list(JOINTS)
    host_s = data["host_monotonic_ns"] * 1.0e-9
    host_gaps = np.diff(host_s)
    finite_gaps = host_gaps[np.isfinite(host_gaps) & (host_gaps > 0)]
    sample_dt_s = float(np.median(finite_gaps)) if finite_gaps.size else 0.002
    action_mask = np.isin(data["phase"], ACTION_PHASES)
    if not np.any(action_mask):
        # Prone-damping remote preflight has no motion-action phase. Retain its
        # timing and remote diagnostics while leaving motion metrics as NaN.
        action_mask = np.zeros(data["phase"].shape, dtype=bool)

    metrics = [joint_metrics(data, joint, action_mask, sample_dt_s) for joint in joints]
    network = network_metrics(data, action_mask)
    support = ground_support_metrics(data)
    abort_reasons = sorted(
        {reason for reason in data["abort_reason"].tolist() if reason.strip()}
    )

    summary_path = args.summary or args.log.with_suffix(args.log.suffix + ".summary.csv")
    plot_dir = args.plot_dir or args.log.with_name(args.log.stem + "_plots")
    write_summary(summary_path, metrics, network, support, abort_reasons)
    plots_written = False
    if not args.no_plots:
        plots_written = make_plots(data, joints, plot_dir)

    print(f"summary: {summary_path}")
    if plots_written:
        print(f"plots: {plot_dir}")
    print(
        "network: "
        f"rate={network['feedback_rate_hz']:.2f} Hz, "
        f"p99_gap={network['feedback_gap_p99_ms']:.3f} ms, "
        f"max_gap={network['feedback_gap_max_ms']:.3f} ms"
    )
    print(
        "motion: "
        f"roll_excursion={network['imu_roll_excursion_rad']:.4f} rad, "
        f"pitch_excursion={network['imu_pitch_excursion_rad']:.4f} rad, "
        f"max_joint_speed={network['max_abs_joint_speed_rad_s']:.4f} rad/s, "
        f"max_temperature={network['max_motor_temperature_c']:.1f} C"
    )
    print(
        "remote: "
        f"valid_fresh_ratio={network['remote_valid_fresh_ratio']:.3f}, "
        f"L2+B_seen={network['remote_l2_b_seen']:.0f}"
    )
    print(
        "support: "
        f"min_margin={support['min_active_support_margin_m']:.4f} m, "
        f"min_air_force_ratio={support['min_airborne_force_ratio']:.3f}, "
        f"contact_ratio={support['final_contact_force_ratio']:.3f}, "
        f"remote_estop_latency={support['remote_estop_latency_ms']:.3f} ms, "
        f"watchdog_cycles={support['watchdog_cycles']:.0f}"
    )
    if abort_reasons:
        print("abort reasons: " + ", ".join(abort_reasons))
    for metric in metrics:
        if metric["max_abs_tau_ff_nm"] > 1.0e-6 or metric["position_rms_rad"] > 0:
            print(
                f"{metric['joint']}: corr={metric['tau_corr_total']:.3f}, "
                f"gain={metric['tau_gain_total']:.3f}, "
                f"rmse={metric['tau_rmse_nm']:.4f} Nm, "
                f"lag={metric['lag_s']:.4f} s, "
                f"q_rms={metric['position_rms_rad']:.4f} rad"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
