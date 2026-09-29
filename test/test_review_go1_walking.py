import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from experiment import review_go1_walking as review


def frame(state=False, q=0.1, tick=1):
    b = bytearray(820 if state else 614)
    b[:3] = b'\xfe\xef\xff'
    for i in range(12):
        if state:
            struct.pack_into('<Bff', b, 75+32*i, 10, q, 0.2)
        else:
            struct.pack_into('<BffhHH', b, 22+27*i, 10, q, 0, 128, 80*32, 2*16)
    if state:
        struct.pack_into('<I', b, 755, tick)
    offset = 803 if state else 610
    struct.pack_into('<I', b, offset, review.sdk_crc(b[:offset]))
    return b


class WalkingReviewTests(unittest.TestCase):
    def test_wire_scales_crc_and_servo_validation(self):
        b = frame(q=0.2)
        np.testing.assert_allclose(review.decode_frame(b)[0], [.2,0,.5,80,2])
        b[30] ^= 1
        with self.assertRaisesRegex(ValueError, 'CRC'):
            review.decode_frame(b)
        b = frame(); b[22] = 0
        struct.pack_into('<I', b, 610, review.sdk_crc(b[:610]))
        with self.assertRaisesRegex(ValueError, 'non-servo'):
            review.decode_frame(b)

    def test_nonfinite_and_active_sentinel_rejected(self):
        for q in (float('nan'), 2.146e9):
            with self.subTest(q=q), self.assertRaises(ValueError):
                review.decode_frame(frame(q=q))

    def test_only_past_fresh_commands_and_pd_effort(self):
        states = np.tile([.1,.2], (4,12,1))
        commands = np.tile([.2,0,.5,80,2], (1,12,1))
        result = review.tracking_metrics([999,1000,5000,5001], states,
                                         [1000], commands, 0, 6000)
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['rejected_stale_or_missing'], 2)
        self.assertAlmostEqual(result['pooled_q_rmse_rad'], .1)
        self.assertAlmostEqual(result['max_abs_predicted_effort_nm'], 8.1)
        self.assertEqual(result['max_pair_age_us'], 4000)

    def test_no_pairs_is_an_error(self):
        with self.assertRaisesRegex(ValueError, 'No fresh'):
            review.tracking_metrics([0], np.zeros((1,12,2)), [1],
                                    np.zeros((1,12,5)), 0, 2)

    def test_exit_order_and_incomplete_restoration(self):
        names = ['guardian_exit_proved','sender_exit_proved','workers_exit_proved',
                 'actors_resume_begin','actors_resumed']
        events = [dict(event=n, monotonic_ns=i+1) for i,n in enumerate(names)]
        self.assertTrue(review.exit_review(events)['exit_before_restore'])
        events[1]['monotonic_ns'] = 9
        self.assertFalse(review.exit_review(events)['exit_before_restore'])
        self.assertFalse(review.exit_review(events[:-1])['actors_resumed'])
        self.assertFalse(review.exit_review([])['exit_before_restore'])

    def make_package(self, root):
        (root/'policy').mkdir()
        (root/'policy/weights.npz').write_bytes(b'fixture weights')
        (root/'policy/manifest.json').write_text(json.dumps({'weights_sha256':review.sha256(root/'policy/weights.npz')}))
        # Arbitrary package code is data to the reviewer, never imported.
        (root/'run_trial.py').write_text('raise RuntimeError("must not execute")')
        tuning = dict(kp=[80]*12,kd=[2]*12,command=[.5,0,0],duration_s=5,gain_ramp_s=.3)
        (root/'tuning.json').write_text(json.dumps(tuning))
        native = 'GO1_WALKING_TUNING_V1\n'
        for n in ('kp','kd','command'):
            native += n+' '+' '.join(format(float(x),'.17g') for x in tuning[n])+'\n'
        for n in ('duration_s','gain_ramp_s'):
            native += n+' '+format(float(tuning[n]),'.17g')+'\n'
        (root/'walking_gains.conf').write_text(native)
        manifest = {'active_tuning':tuning,'files':{str(p.relative_to(root)):review.sha256(p) for p in root.rglob('*') if p.is_file()}}
        (root/'click_release.json').write_text(json.dumps(manifest))
        return manifest

    def test_package_hashes_config_and_no_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.make_package(root)
            result=review.package_review(root)
            self.assertTrue(result['integrity_passed'])
            self.assertFalse(result['hardware_ready'])
            self.assertEqual(result['source'], root.name)
            self.assertNotIn(str(root.parent), json.dumps(result))
            (root/'walking_gains.conf').write_text('changed')
            result=review.package_review(root)
            self.assertFalse(result['integrity_passed'])
            self.assertTrue(any('native gain config' in x for x in result['problems']))

    def test_manifest_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=self.make_package(root)
            m['files']['../outside']='0'*64
            (root/'click_release.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'escapes'):
                review.package_review(root)

    def test_duplicate_manifest_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'x.json';p.write_text('{"files":{},"files":{}}')
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                review.read_json(p)

    def test_run_endpoint_isolation_and_capture_count_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'test/run').mkdir(parents=True)
            for name, value in [('release_manifest.json',{'files':{}}),('tuning.json',{})]:
                (root/name).write_text(json.dumps(value))
            (root/'capture.log').write_text('0 packets dropped by kernel')
            (root/'traffic.pcap').write_bytes(b'fixture')
            (root/'test/run/watchdog.log').write_text('')
            (root/'test/run/terminal.stdout').write_text('POLICY_SEND_SPAN:{"count":2,"span_ns":2000000}\nSDK_AUDIT:{"sent":2}\n')
            rows=[(1_000_000,'192.168.123.161',8080,'192.168.123.10',8007,frame(q=.2)),
                  (1_000_500,'192.168.123.10',8007,'192.168.123.161',8008,frame(True,q=99)),
                  (1_001_000,'192.168.123.10',8007,'192.168.123.161',8080,frame(True)),
                  (1_001_001,'192.168.123.10',8007,'192.168.123.161',8080,frame(True)),
                  (1_002_000,'192.168.123.161',8080,'192.168.123.10',8007,frame(q=.2))]
            with patch.object(review,'packets',return_value=iter(rows)):
                result=review.run_review(root,window=(0,.01))
            self.assertEqual(result['source'],root.name)
            self.assertNotIn(str(root.parent),json.dumps(result))
            self.assertEqual(result['duplicate_state_ticks'],1)
            self.assertEqual(result['fresh_custom_states'],1)
            self.assertAlmostEqual(result['tracking']['pooled_q_rmse_rad'],.1)
            with patch.object(review,'packets',return_value=iter(rows[:-1])):
                with self.assertRaisesRegex(ValueError,'do not match'):
                    review.run_review(root)


if __name__=='__main__':unittest.main()
