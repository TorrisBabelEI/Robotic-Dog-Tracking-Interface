import struct
import tempfile
import unittest
from pathlib import Path
from experiment.decode_native_go1_pcap import packets


class LinkLayerTests(unittest.TestCase):
    def capture(self, linktype):
        payload = b'example'
        udp = struct.pack('!HHHH', 8007, 8008, 8+len(payload), 0) + payload
        ip = struct.pack('!BBHHHBBH4s4s', 0x45, 0, 20+len(udp), 0, 0,
                         64, 17, 0, bytes([192,168,123,10]), bytes([192,168,123,161])) + udp
        link = bytes(12 if linktype == 1 else 14) + b'\x08\x00'
        frame = link+ip
        return (struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, linktype)
                + struct.pack('<IIII', 1, 123, len(frame), len(frame)) + frame)

    def read(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.pcap'
            path.write_bytes(content)
            return list(packets(path))

    def test_ethernet_and_cooked_yield_identical_udp(self):
        expected = [(1000123, '192.168.123.10', 8007, '192.168.123.161', 8008, b'example')]
        self.assertEqual(self.read(self.capture(1)), expected)
        self.assertEqual(self.read(self.capture(113)), expected)

    def test_unknown_link_type_rejected(self):
        with self.assertRaises(ValueError):
            self.read(self.capture(999))

    def test_truncated_packet_rejected(self):
        for link in (1, 113):
            with self.assertRaises(ValueError):
                self.read(self.capture(link)[:-1])


if __name__ == '__main__':
    unittest.main()
