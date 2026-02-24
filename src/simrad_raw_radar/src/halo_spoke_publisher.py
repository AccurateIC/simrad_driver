#!/usr/bin/env python3
"""
Simrad Halo Spoke Publisher — ROS 1 (rospy)
============================================
Source reference: opencpn-radar-pi/radar_pi
  src/navico/NavicoReceive.cpp  (ProcessFrame logic)
  include/navico/haloatype.h    (br4g_header struct — used by RT_HaloA/RT_HaloB)

Subscribes to 236.6.7.8:6678 (Halo multicast) and publishes every decoded
spoke as a RadarSpoke message on /radar/spoke.

Also detects rotation boundaries (angle_raw wrap) and prints a summary to
the terminal once per full antenna revolution.

ROS setup
---------
  1. Copy RadarSpoke.msg to <your_package>/msg/RadarSpoke.msg
  2. In CMakeLists.txt add:
       find_package(catkin REQUIRED COMPONENTS rospy std_msgs message_generation)
       add_message_files(FILES RadarSpoke.msg)
       generate_messages(DEPENDENCIES std_msgs)
  3. In package.xml add:
       <build_depend>message_generation</build_depend>
       <exec_depend>message_runtime</exec_depend>
  4. catkin_make  (or catkin build)
  5. source devel/setup.bash
  6. rosrun <your_package> halo_spoke_publisher.py

Usage
-----
  rosrun <your_package> halo_spoke_publisher.py
  rosrun <your_package> halo_spoke_publisher.py _interface:=192.168.7.105
  rosrun <your_package> halo_spoke_publisher.py _interface:=192.168.7.105 _every:=4

Topics published
----------------
  /radar/spoke   (RadarSpoke)   — every decoded spoke
"""

import socket
import struct
import argparse

import rospy
from std_msgs.msg import Header

# Import the custom message.
# Replace 'your_package' with the actual ROS package name.
from simrad_raw_radar.msg import RadarSpoke

# ─────────────────────────────────────────────────────────────────────────────
# Constants — verified directly from NavicoReceive.cpp
# ─────────────────────────────────────────────────────────────────────────────

MCAST_IP   = "236.6.7.8"
MCAST_PORT = 6678

# Both br24_header and br4g_header are /* total size = 24 */ = 0x18
# Halo (RT_HaloA / RT_HaloB) uses br4g_header.
HEADER_LEN_BR24 = 0x18   # 24 bytes — br24_header (BR24 / 3G)
HEADER_LEN_BR4G = 0x18   # 24 bytes — br4g_header (Halo)

# ── VERIFIED br4g_header byte layout (Halo uses line->br4g, NOT line->br24) ──
# (from NavicoReceive.cpp ProcessFrame, case RT_HaloA / RT_HaloB)
#
#  Offset  Size  Field          Notes
#  0       1     headerLen      0x18
#  1       1     status         0x02 = transmitting
#  2       2     scan_number    LE uint16, 0–4095, spoke counter
#  4       2     u00            Always 0x4400
#  6       2     largerange     LE uint16
#  8       2     angle          LE uint16, 0–4095
#  10      2     heading        LE uint16, masked — see HEADING_* flags
#  12      2     smallrange     LE uint16
#  14      2     rotation       LE uint16
#  16      4     u02            Always 0xFFFFFFFF
#  20      4     u03            Mostly 0xFFFFFFFF
#                               Total = 24 bytes ✓

OFF_STATUS      = 1
OFF_SCAN_NUMBER = 2
OFF_LARGERANGE  = 6
OFF_ANGLE       = 8
OFF_HEADING     = 10
OFF_SMALLRANGE  = 12

NAVICO_SPOKES_RAW = 4096
DEG_PER_SPOKE     = 360.0 / NAVICO_SPOKES_RAW

# Heading flags (from NavicoReceive.cpp):
#   HEADING_TRUE_FLAG = 0x4000
#   HEADING_MASK      = 0x0FFF  (NAVICO_SPOKES_RAW - 1)
#   HEADING_VALID(x)  = ((x & ~(0x4000|0x0FFF)) == 0)
HEADING_TRUE_FLAG = 0x4000
HEADING_MASK      = NAVICO_SPOKES_RAW - 1   # 0x0FFF

STATUS_TX = 0x02

# ── Sample encoding ───────────────────────────────────────────────────────────
# Wire: 512 nibble-packed bytes → 1024 decoded intensity values (0–255)
# lookupNibbleToByte from NavicoReceive.cpp:
NAVICO_SPOKE_LEN  = 1024
NAVICO_DATA_BYTES = NAVICO_SPOKE_LEN // 2   # 512 bytes on the wire

NIBBLE_TO_BYTE = [
    0x00, 0x32, 0x40, 0x4e, 0x5c, 0x6a, 0x78, 0x86,
    0x94, 0xa2, 0xb0, 0xbe, 0xcc, 0xda, 0xe8, 0xff,
]

# ── Packet framing ────────────────────────────────────────────────────────────
# radar_frame_pkt: frame_hdr[8] + radar_line[120]
# Each radar_line = br4g_header(24) + data(512) = 536 bytes
FRAME_HDR_LEN  = 8
SPOKE_LINE_LEN = HEADER_LEN_BR4G + NAVICO_DATA_BYTES   # 536 bytes

# Rotation wrap threshold: drop > 512 raw steps (~45°) → new rotation
ROTATION_WRAP_THRESHOLD = 512


# ─────────────────────────────────────────────────────────────────────────────
# Interface auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def auto_interface():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((MCAST_IP, MCAST_PORT))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Socket: join multicast
# ─────────────────────────────────────────────────────────────────────────────

def join_multicast(interface_ip):
    """
    Mirrors startUDPMulticastReceiveSocket() in radar_pi:
      socket() → SO_REUSEADDR → bind(INADDR_ANY:port) → IP_ADD_MEMBERSHIP
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        pass  # not available on Windows

    sock.bind(("", MCAST_PORT))
    mreq = socket.inet_aton(MCAST_IP) + socket.inet_aton(interface_ip)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(2.0)
    return sock


# ─────────────────────────────────────────────────────────────────────────────
# Decoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def unpack_samples(raw_bytes):
    """
    Unpack 512 nibble-packed wire bytes → 1024 intensity values (0–255).
    low nibble  → sample[2i],  high nibble → sample[2i+1]
    """
    out = bytearray(NAVICO_SPOKE_LEN)
    for i, b in enumerate(raw_bytes):
        out[2 * i]     = NIBBLE_TO_BYTE[b & 0x0F]
        out[2 * i + 1] = NIBBLE_TO_BYTE[(b >> 4) & 0x0F]
    return bytes(out)   # bytes — directly usable as uint8[] in ROS msg


def decode_range(large_raw, small_raw):
    if large_raw == 0x80:
        if small_raw == 0xFFFF or small_raw == 0:
            return 0
        return (small_raw * 8) // 55   # ← correct
    else:
        return (large_raw * small_raw) // 512


def decode_heading(raw):
    """
    Returns (valid: bool, is_true: bool, degrees: float).
    HEADING_VALID(x) = ((x & ~(0x4000 | 0x0FFF)) == 0)
    """
    valid = ((raw & ~(HEADING_TRUE_FLAG | HEADING_MASK)) == 0)
    if not valid:
        return False, False, -1.0
    is_true = bool(raw & HEADING_TRUE_FLAG)
    deg = (raw & HEADING_MASK) * DEG_PER_SPOKE
    return True, is_true, deg


# ─────────────────────────────────────────────────────────────────────────────
# Packet decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_packet(udp_payload):
    """
    Decode a full UDP payload into a list of spoke dicts.

    Frame: frame_hdr[8] + radar_line[N]
    Each radar_line: br4g_header(24) + nibble-data(512) = 536 bytes
    Typically 32 spokes per packet; up to 120.
    """
    spokes = []
    if len(udp_payload) < FRAME_HDR_LEN + 2:
        return spokes

    offset = FRAME_HDR_LEN   # skip 8-byte frame header

    while offset < len(udp_payload):
        header_len = udp_payload[offset]

        if header_len != HEADER_LEN_BR4G:
            break  # unrecognised — stop

        end = offset + header_len + NAVICO_DATA_BYTES
        if end > len(udp_payload):
            break  # truncated

        hdr = udp_payload[offset: offset + header_len]

        status      = hdr[OFF_STATUS]
        scan_number = struct.unpack_from('<H', hdr, OFF_SCAN_NUMBER)[0]
        large_range = struct.unpack_from('<H', hdr, OFF_LARGERANGE)[0]
        angle_raw   = struct.unpack_from('<H', hdr, OFF_ANGLE)[0]
        heading_raw = struct.unpack_from('<H', hdr, OFF_HEADING)[0]
        small_range = struct.unpack_from('<H', hdr, OFF_SMALLRANGE)[0]

        range_m   = decode_range(large_range, small_range)
        angle_deg = angle_raw * DEG_PER_SPOKE
        hdg_valid, hdg_true, heading_deg = decode_heading(heading_raw)

        raw_data = udp_payload[offset + header_len: end]
        samples  = unpack_samples(raw_data)   # bytes, length 1024

        spokes.append({
            "spoke_number": scan_number,
            "angle_raw":    angle_raw,
            "angle_deg":    angle_deg,
            "range_m":      range_m,
            "heading_deg":  heading_deg,   # -1.0 if invalid
            "heading_true": hdg_true,
            "status":       status,
            "samples":      samples,       # bytes, 1024 values 0–255
        })

        offset = end

    return spokes


# ─────────────────────────────────────────────────────────────────────────────
# Rotation tracker
# ─────────────────────────────────────────────────────────────────────────────

class RotationTracker:
    """
    Detects rotation boundary when angle_raw drops > ROTATION_WRAP_THRESHOLD.
    Fires on_rotation() summary to terminal once per full sweep.
    rotation_number is embedded in each RadarSpoke message (uint8, wraps 0–255).
    """

    def __init__(self):
        self.rotation_number  = 0
        self.current_rotation = []   # spokes buffered for current sweep
        self._prev_angle_raw  = None

    def feed(self, spoke):
        angle_raw = spoke["angle_raw"]

        if self._prev_angle_raw is not None:
            delta = angle_raw - self._prev_angle_raw
            if delta < -ROTATION_WRAP_THRESHOLD:
                self._on_rotation_complete()
                self.rotation_number = (self.rotation_number + 1) & 0xFF
                self.current_rotation = []

        self._prev_angle_raw = angle_raw
        self.current_rotation.append(spoke)

    def _on_rotation_complete(self):
        spokes = self.current_rotation
        if not spokes:
            return
        angles = [s["angle_deg"] for s in spokes]
        peaks  = [max(s["samples"]) for s in spokes]
        ranges = [s["range_m"] for s in spokes if s["range_m"] > 0]
        rng_str = f"{ranges[0]} m" if ranges else "n/a"
        rospy.loginfo(
            f"Rotation {self.rotation_number:>3d} complete | "
            f"{len(spokes):>4d} spokes | "
            f"{min(angles):.1f}°–{max(angles):.1f}° | "
            f"range {rng_str} | "
            f"max peak {max(peaks)}/255"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ROS publisher
# ─────────────────────────────────────────────────────────────────────────────

def make_spoke_msg(spoke, rotation_number):
    """Build a RadarSpoke ROS message from a decoded spoke dict."""
    msg = RadarSpoke()

    msg.header              = Header()
    msg.header.stamp        = rospy.Time.now()
    msg.header.frame_id     = "radar"

    msg.spoke_number        = spoke["spoke_number"]
    msg.angle_raw           = spoke["angle_raw"]
    msg.angle_deg           = float(spoke["angle_deg"])
    msg.range_m             = int(spoke["range_m"])
    msg.heading_deg         = float(spoke["heading_deg"])   # -1.0 = invalid
    msg.heading_true        = bool(spoke["heading_true"])
    msg.status              = spoke["status"]
    msg.rotation_number     = rotation_number & 0xFF

    # samples is already bytes (length 1024) — assign directly to uint8[]
    msg.samples             = spoke["samples"]

    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rospy.init_node("halo_radar_spoke_publisher", anonymous=False)

    # ROS params (set with _param:=value on the command line, or via launch file)
    iface = rospy.get_param("~interface", None) or auto_interface()
    every = int(rospy.get_param("~every", 1))

    pub = rospy.Publisher("/radar/spoke", RadarSpoke, queue_size=256)

    rospy.loginfo("Simrad Halo Spoke Publisher")
    rospy.loginfo(f"  Multicast   : {MCAST_IP}:{MCAST_PORT}")
    rospy.loginfo(f"  Interface   : {iface}")
    rospy.loginfo(f"  Publishing  : /radar/spoke  (RadarSpoke)")
    rospy.loginfo(f"  Every       : {every} spoke(s)")

    sock    = join_multicast(iface)
    tracker = RotationTracker()

    rospy.loginfo("Joined multicast group. Receiving spokes...")

    pkt_count   = 0
    spoke_count = 0
    pub_count   = 0

    while not rospy.is_shutdown():
        try:
            payload, _ = sock.recvfrom(65536)
        except socket.timeout:
            continue
        except OSError:
            break

        pkt_count += 1

        for spoke in decode_packet(payload):
            spoke_count += 1
            if spoke_count % every != 0:
                continue

            # Update rotation tracker (fires terminal summary on wrap)
            tracker.feed(spoke)

            # Build and publish the ROS message
            msg = make_spoke_msg(spoke, tracker.rotation_number)
            pub.publish(msg)
            pub_count += 1

    sock.close()
    rospy.loginfo(
        f"Stopped. pkts={pkt_count} spokes={spoke_count} "
        f"published={pub_count} rotations={tracker.rotation_number}"
    )


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass