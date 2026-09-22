import struct,unittest
from experiment.review_factory_effort import total_effort,sdk_crc,command_crc
class EffortTests(unittest.TestCase):
    def frames(self,qtarget=.2,kp=5,kd=1):
        c=bytearray(614);s=bytearray(820);c[:3]=s[:3]=b'\xfe\xef\xff'
        for i in range(12):
            struct.pack_into('<Bff',s,75+32*i,10,.1,.3)
            struct.pack_into('<BffhHH',c,22+27*i,10,qtarget,0,128,round(kp*32),round(kd*16))
        struct.pack_into('<I',s,803,sdk_crc(s[:803]));struct.pack_into('<I',c,610,command_crc(c[:610]));return c,s
    def test_pd_plus_ff(self):
        c,s=self.frames();self.assertTrue(all(abs(v-.7)<1e-6 for v in total_effort(c,s)))
    def test_crc_rejected(self):
        c,s=self.frames();c[24]^=1
        with self.assertRaises(ValueError):total_effort(c,s)
    def test_active_sentinel_rejected(self):
        c,s=self.frames(2.146e9)
        with self.assertRaises(ValueError):total_effort(c,s)
    def test_zero_gain_sentinel(self):
        c,s=self.frames(2.146e9,0,0);self.assertEqual(total_effort(c,s),[.5]*12)
    def test_nan_rejected(self):
        c,s=self.frames(float('nan'))
        with self.assertRaises(ValueError):total_effort(c,s)


    def test_sdk_profile_requires_explicit_selection(self):
        c,s=self.frames()
        struct.pack_into('<I',c,610,sdk_crc(c[:610]))
        self.assertTrue(all(abs(v-.7)<1e-6 for v in total_effort(c,s,profile='sdk')))
        with self.assertRaises(ValueError):total_effort(c,s)

    def test_state_crc_and_nonservo_rejected(self):
        c,s=self.frames();s[100]^=1
        with self.assertRaises(ValueError):total_effort(c,s)
        c,s=self.frames();c[22]=0
        struct.pack_into('<I',c,610,command_crc(c[:610]))
        with self.assertRaises(ValueError):total_effort(c,s)

    def test_pairing_age_and_source_isolation(self):
        from experiment import review_factory_effort as module
        from unittest.mock import patch
        from pathlib import Path
        import tempfile
        c,s=self.frames()
        def state(t,port=8008):return (t,'192.168.123.10',8007,'192.168.123.161',port,s)
        def command(t):return (t,'192.168.123.161',8008,'192.168.123.10',8007,c)
        frames=[command(1),state(100,8090),command(101),state(200),command(4200),command(4201)]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'capture';path.write_bytes(b'fixture')
            with patch.object(module,'packets',return_value=iter(frames)):
                report,rows=module.review(path)
            self.assertEqual(report['paired_frames'],1)
            self.assertEqual(rows[0]['state_age_us'],4000)
            self.assertEqual(sum(report['rejected'].values()),3)
            with patch.object(module,'packets',return_value=iter([state(200),command(199)])):
                report,rows=module.review(path)
                self.assertEqual(report['out_of_order_packets'],1)
                self.assertEqual(rows,[])

if __name__=='__main__':unittest.main()
