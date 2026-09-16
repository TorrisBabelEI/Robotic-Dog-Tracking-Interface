import unittest

from experiment.operator_support_gate import (
    LEASE_S,
    PULSE_S,
    ConfirmationPulse,
    SupportLease,
)


class ConfirmationPulseTests(unittest.TestCase):
    def test_single_click_expires_without_being_held(self):
        pulse = ConfirmationPulse()
        self.assertTrue(pulse.start(10.0))
        self.assertTrue(pulse.active(10.0 + PULSE_S - 0.001))
        self.assertFalse(pulse.active(10.0 + PULSE_S))

    def test_repeat_click_cannot_extend_active_pulse(self):
        pulse = ConfirmationPulse()
        self.assertTrue(pulse.start(10.0))
        self.assertFalse(pulse.start(10.5))
        self.assertFalse(pulse.active(10.0 + PULSE_S))
        self.assertTrue(pulse.start(12.0))
        pulse.cancel()
        self.assertFalse(pulse.active(12.1))


class SupportLeaseTests(unittest.TestCase):
    def test_hold_expires_without_heartbeat(self):
        lease = SupportLease()
        self.assertTrue(lease.receive(b"H 1", 10.0))
        self.assertTrue(lease.active(10.0 + LEASE_S / 2))
        self.assertFalse(lease.active(10.0 + LEASE_S + 0.001))

    def test_release_is_immediate(self):
        lease = SupportLease()
        lease.receive(b"H 1", 10.0)
        self.assertFalse(lease.receive(b"R 2", 10.02))
        self.assertFalse(lease.active(10.03))

    def test_old_or_malformed_frames_fail_closed(self):
        for frame in (b"H 1", b"H -1", b"H 999999999999999999999",
                      b"H 2 extra", b"Q 2", b"\xff"):
            lease = SupportLease()
            lease.receive(b"H 1", 10.0)
            self.assertFalse(lease.receive(frame, 10.02), frame)
            self.assertFalse(lease.active(10.03), frame)

    def test_reset_or_backward_clock_fails_closed(self):
        lease = SupportLease()
        lease.receive(b"H 1", 10.0)
        self.assertFalse(lease.active(9.0))
        lease.receive(b"H 2", 11.0)
        lease.reset()
        self.assertFalse(lease.active(11.0))


if __name__ == "__main__":
    unittest.main()
