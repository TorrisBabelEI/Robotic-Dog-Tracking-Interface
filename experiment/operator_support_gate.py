#!/usr/bin/env python3
"""Offline-only prototype of a time-limited operator support confirmation.

Both sides bind/connect to 127.0.0.1. This program never imports the Go1 SDK,
opens a robot socket, or changes the C++ controller's hardware lock. A later
review must separately approve transport to the Pi and control-loop wiring.
"""

from __future__ import annotations

import argparse
import select
import socket
import sys
import time
from dataclasses import dataclass


LEASE_S = 0.10
SEND_INTERVAL_MS = 25
PULSE_S = 1.50
MAX_FRAME = 32


@dataclass
class ConfirmationPulse:
    """A single click grants one short, non-extendable confirmation window."""

    deadline_s: float | None = None

    def start(self, now_s: float) -> bool:
        if self.active(now_s):
            return False
        self.deadline_s = now_s + PULSE_S
        return True

    def active(self, now_s: float) -> bool:
        if self.deadline_s is None:
            return False
        if now_s < self.deadline_s:
            return True
        self.deadline_s = None
        return False

    def cancel(self) -> None:
        self.deadline_s = None


@dataclass
class SupportLease:
    """Fail-closed receiver state; timestamps are local arrival times."""

    last_sequence: int = -1
    last_hold_s: float | None = None
    held: bool = False

    def reset(self) -> None:
        self.last_sequence = -1
        self.last_hold_s = None
        self.held = False

    def receive(self, frame: bytes, now_s: float) -> bool:
        try:
            text = frame.decode("ascii")
            command, sequence_text = text.split(" ")
            if command not in ("H", "R") or not sequence_text.isdecimal():
                raise ValueError("invalid frame")
            sequence = int(sequence_text)
            if sequence > 2**63 - 1 or sequence <= self.last_sequence:
                raise ValueError("stale or overflowing sequence")
        except (UnicodeDecodeError, ValueError):
            self.reset()
            return False

        # Once a heartbeat gap exceeds the lease, old state cannot be revived
        # as a continuation of the previous hold. A new hold starts at now_s.
        self.last_sequence = sequence
        if command == "R":
            self.held = False
            self.last_hold_s = None
        else:
            self.held = True
            self.last_hold_s = now_s
        return self.active(now_s)

    def active(self, now_s: float) -> bool:
        if not self.held or self.last_hold_s is None:
            return False
        if now_s < self.last_hold_s or now_s - self.last_hold_s > LEASE_S:
            self.held = False
            self.last_hold_s = None
            return False
        return True


def run_probe(port: int) -> int:
    lease = SupportLease()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        listener.setblocking(False)
        print(f"Offline probe listening on 127.0.0.1:{port}; no robot access")
        print("Click the GUI once; its 1.5-second confirmation expires automatically.")
        print("Press Ctrl-C to close the offline probe.")
        connection: socket.socket | None = None
        pending = bytearray()
        previous = False
        try:
            while True:
                readable, _, _ = select.select(
                    [listener] + ([connection] if connection else []), [], [], 0.02
                )
                if listener in readable:
                    candidate, _ = listener.accept()
                    if connection is not None:
                        candidate.close()
                    else:
                        connection = candidate
                        connection.setblocking(False)
                        pending.clear()
                        lease.reset()
                        print("Sender connected; support starts false", flush=True)
                if connection is not None and connection in readable:
                    chunk = connection.recv(1024)
                    if not chunk:
                        connection.close()
                        connection = None
                        pending.clear()
                        lease.reset()
                        print("Sender disconnected; support=false", flush=True)
                    else:
                        pending.extend(chunk)
                        if len(pending) > MAX_FRAME * 4:
                            pending.clear()
                            lease.reset()
                        while b"\n" in pending:
                            frame, _, rest = pending.partition(b"\n")
                            pending = bytearray(rest)
                            if len(frame) > MAX_FRAME:
                                lease.reset()
                            else:
                                lease.receive(frame, time.monotonic())
                active = lease.active(time.monotonic()) if connection else False
                if active != previous:
                    previous = active
                    print(f"support={'true' if active else 'false'}", flush=True)
        except KeyboardInterrupt:
            print("Probe stopped; support=false")
        finally:
            if connection is not None:
                connection.close()
    return 0


def run_sender(port: int) -> int:
    try:
        import tkinter as tk
    except ImportError as error:
        print(f"Tkinter unavailable: {error}", file=sys.stderr)
        return 2

    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=2)
    except OSError as error:
        print(f"Cannot connect to the offline probe: {error}", file=sys.stderr)
        return 2
    connection.settimeout(0.2)
    root = tk.Tk()
    root.title("Offline prone-support input test")
    root.geometry("520x240")
    sequence = 0
    pulse = ConfirmationPulse()

    label = tk.Label(root, text="NOT CONFIRMED", font=("Sans", 18), fg="red")
    label.pack(pady=25)
    button = tk.Button(root, text="Click once after visually confirming belly contact",
                       font=("Sans", 13), width=42, height=3)
    button.pack()

    def send(command: str) -> None:
        nonlocal sequence
        sequence += 1
        try:
            connection.sendall(f"{command} {sequence}\n".encode("ascii"))
        except OSError:
            pulse.cancel()
            label.configure(text="CONNECTION LOST", fg="red")
            button.configure(state="disabled")

    def confirm() -> None:
        if pulse.start(time.monotonic()):
            label.configure(text="CONFIRMING FOR 1.5 SECONDS", fg="green")
            send("H")

    def cancel(_event: object = None) -> None:
        if pulse.deadline_s is not None:
            pulse.cancel()
            label.configure(text="NOT CONFIRMED", fg="red")
            send("R")

    def heartbeat() -> None:
        pending = pulse.deadline_s is not None
        if pulse.active(time.monotonic()):
            send("H")
        elif pending:
            label.configure(text="NOT CONFIRMED", fg="red")
            send("R")
        root.after(SEND_INTERVAL_MS, heartbeat)

    def close() -> None:
        cancel()
        connection.close()
        root.destroy()

    button.configure(command=confirm)
    root.bind("<FocusOut>", cancel)
    root.protocol("WM_DELETE_WINDOW", close)
    heartbeat()
    try:
        root.mainloop()
    finally:
        connection.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "sender"))
    parser.add_argument("--port", type=int, default=18092)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be in 1024..65535")
    return run_probe(args.port) if args.mode == "probe" else run_sender(args.port)


if __name__ == "__main__":
    raise SystemExit(main())
