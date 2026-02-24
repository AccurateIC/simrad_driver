#!/usr/bin/env python3
"""
Simrad Halo — PPI Radar Visualiser
====================================
Subscribes to /radar/spoke (RadarSpoke) and renders a real-time
Plan Position Indicator (PPI) scope using OpenCV.

Erase behaviour
---------------
A spoke is erased ONLY when the antenna sweeps back to the same bearing
and a new spoke is about to be written in its place — exactly like a
real radar display.  There is no time-based fade.

  - 4096 spoke slots (one per raw angle step) are maintained.
  - When spoke N arrives, the pixels belonging to the PREVIOUS spoke N
    are zeroed in the pixel buffer, then the new pixels are written.
  - The pixel buffer is uint8 — no float accumulation needed.

Features
--------
  - Full 360° PPI display
  - Sweep line tracking current antenna bearing
  - Range rings and cardinal / intercardinal bearing labels
  - Live HUD: rotation, spoke rate, range, heading, gain
  - Gain control (+/-)
  - Clear (C)

Dependencies
------------
  pip install opencv-python numpy

ROS setup
---------
  Replace 'your_package' with your actual catkin package name.
  source devel/setup.bash
  rosrun <your_package> halo_radar_viz.py
  rosrun <your_package> halo_radar_viz.py _topic:=/radar/spoke _gain:=1.5

Keyboard controls (OpenCV window must have focus)
-------------------------------------------------
  Q / ESC   quit
  + / =     increase gain
  -         decrease gain
  C         clear display
"""

import math
import threading
import time

import cv2
import numpy as np
import rospy

from simrad_raw_radar.msg import RadarSpoke   # ← replace with your package name

# ─────────────────────────────────────────────────────────────────────────────
# Display configuration
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_SIZE      = 900          # window width & height in pixels
CENTRE            = DISPLAY_SIZE // 2
RADIUS            = CENTRE - 10  # usable PPI radius in pixels

NAVICO_SPOKE_LEN  = 1024         # intensity samples per spoke
NAVICO_SPOKES_RAW = 4096         # raw angle steps per full rotation

# Colours (BGR)
SWEEP_COLOUR  = (0, 255, 80)     # bright green — sweep line
RING_COLOUR   = (0, 70, 0)       # dim green    — range rings
LABEL_COLOUR  = (0, 160, 0)      # mid green    — bearing labels
HUD_COLOUR    = (0, 210, 0)      # mid green    — HUD text
SWEEP_WIDTH   = 2

RING_COUNT    = 4                # number of range rings
FPS_TARGET    = 30               # display refresh rate


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute pixel coordinates for every possible spoke angle
# ─────────────────────────────────────────────────────────────────────────────
# For each of the 4096 raw angle slots we store the (px, py) arrays of pixels
# that a spoke at that angle would illuminate.  This is done once at startup
# and reused every time a spoke arrives, avoiding per-callback trig overhead.

def _precompute_spoke_pixels():
    """
    Returns a list of 4096 entries.
    Entry i = (px, py) — int32 arrays of pixel coords for angle_raw == i.
    """
    t = np.linspace(0.0, 1.0, NAVICO_SPOKE_LEN, dtype=np.float32)
    slots = []
    for angle_raw in range(NAVICO_SPOKES_RAW):
        angle_deg = angle_raw * 360.0 / NAVICO_SPOKES_RAW
        rad = math.radians(angle_deg - 90.0)   # 0° north = up in image
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        px = (CENTRE + t * RADIUS * cos_a).astype(np.int32)
        py = (CENTRE + t * RADIUS * sin_a).astype(np.int32)

        # Clip to display bounds
        mask = (px >= 0) & (px < DISPLAY_SIZE) & (py >= 0) & (py < DISPLAY_SIZE)
        slots.append((px[mask], py[mask], mask))   # store mask too for sample indexing

    return slots

rospy.loginfo("Pre-computing spoke pixel coordinates (4096 slots)...")
SPOKE_PIXELS = _precompute_spoke_pixels()
rospy.loginfo("Pre-computation done.")


# ─────────────────────────────────────────────────────────────────────────────
# PPI display
# ─────────────────────────────────────────────────────────────────────────────

class PPIDisplay:
    """
    Maintains:
      _canvas   — uint8 BGR pixel buffer (DISPLAY_SIZE × DISPLAY_SIZE × 3)
      _slots    — dict mapping angle_raw → (px, py) pixel arrays currently
                  painted for that slot, so they can be erased on next hit.

    The old spoke is erased and the new one is written atomically under
    a single lock acquisition, so there is never a visible blank frame.
    """

    def __init__(self, initial_gain=1.0):
        self._canvas = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 3), dtype=np.uint8)
        self._lock   = threading.Lock()

        # Per-slot memory: angle_raw → (px, py) that were last painted there.
        # None means the slot has never been written.
        self._slot_pixels = [None] * NAVICO_SPOKES_RAW   # (px, py) or None

        # HUD state
        self._sweep_angle_deg = 0.0
        self._rotation_count  = 0
        self._spoke_count     = 0
        self._spoke_rate      = 0.0
        self._range_m         = 0
        self._heading_deg     = -1.0

        # Tuneable
        self.gain = float(initial_gain)

        # Rate tracking
        self._rate_t0     = time.time()
        self._rate_spokes = 0

        # Static overlays pre-computed once
        self._ring_radii = [
            int(RADIUS * (i + 1) / RING_COUNT)
            for i in range(RING_COUNT)
        ]
        self._bearing_labels = []
        for deg, label in [(0, "N"), (45, "NE"), (90, "E"), (135, "SE"),
                           (180, "S"), (225, "SW"), (270, "W"), (315, "NW")]:
            rad = math.radians(deg - 90.0)
            r   = RADIUS + 14
            x   = int(CENTRE + r * math.cos(rad))
            y   = int(CENTRE + r * math.sin(rad))
            self._bearing_labels.append((x, y, label))

    # ── Called from ROS subscriber thread ────────────────────────────────

    def feed_spoke(self, angle_raw, angle_deg, range_m, samples,
                   heading_deg, rotation_number):
        """
        Erase the previous spoke at this angle slot, then paint the new one.
        Everything happens under a single lock — no blank-frame artefacts.

        Parameters
        ----------
        angle_raw       : int     raw antenna angle 0–4095
        angle_deg       : float   angle in degrees
        range_m         : int     current range setting in metres
        samples         : bytes   1024 intensity values 0–255
        heading_deg     : float   vessel heading; -1.0 = unavailable
        rotation_number : int     rotation counter (uint8 from publisher)
        """
        if len(samples) != NAVICO_SPOKE_LEN:
            return
        if not (0 <= angle_raw < NAVICO_SPOKES_RAW):
            return

        # Retrieve pre-computed pixel coords and in-bounds mask for this slot
        px, py, mask = SPOKE_PIXELS[angle_raw]

        # Map intensity samples through gain → green channel
        raw_int = np.frombuffer(samples, dtype=np.uint8)[mask]
        scaled  = np.clip(
            (raw_int.astype(np.float32) * self.gain), 0, 255
        ).astype(np.uint8)

        # Build colour columns: (N, 3) BGR — green channel only
        colours       = np.zeros((len(px), 3), dtype=np.uint8)
        colours[:, 1] = scaled

        # Update spoke rate counter
        self._rate_spokes += 1
        now = time.time()
        dt  = now - self._rate_t0
        if dt >= 1.0:
            self._spoke_rate  = self._rate_spokes / dt
            self._rate_spokes = 0
            self._rate_t0     = now

        with self._lock:
            # 1. ERASE old spoke pixels at this slot (if any)
            prev = self._slot_pixels[angle_raw]
            if prev is not None:
                opx, opy = prev
                self._canvas[opy, opx] = 0

            # 2. PAINT new spoke pixels
            self._canvas[py, px] = colours

            # 3. Remember which pixels we just painted so we can erase them next time
            self._slot_pixels[angle_raw] = (px, py)

            # 4. Update HUD state
            self._sweep_angle_deg = angle_deg
            self._rotation_count  = rotation_number
            self._spoke_count    += 1
            self._range_m         = range_m
            self._heading_deg     = heading_deg

    # ── Called from main / display thread ────────────────────────────────

    def render_frame(self):
        """
        Compose one display frame (uint8 BGR) without modifying the canvas.
        Overlays (rings, labels, sweep line, HUD) are drawn onto a copy.
        """
        with self._lock:
            frame        = self._canvas.copy()
            sweep_deg    = self._sweep_angle_deg
            rot_count    = self._rotation_count
            spoke_count  = self._spoke_count
            spoke_rate   = self._spoke_rate
            range_m      = self._range_m
            heading_deg  = self._heading_deg

        # ── Range rings ───────────────────────────────────────────────────
        for r in self._ring_radii:
            cv2.circle(frame, (CENTRE, CENTRE), r, RING_COLOUR, 1, cv2.LINE_AA)

        # ── Bearing labels ────────────────────────────────────────────────
        for (x, y, label) in self._bearing_labels:
            cv2.putText(frame, label, (x - 8, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, LABEL_COLOUR, 1, cv2.LINE_AA)

        # ── Sweep line ────────────────────────────────────────────────────
        rad = math.radians(sweep_deg - 90.0)
        sx  = int(CENTRE + RADIUS * math.cos(rad))
        sy  = int(CENTRE + RADIUS * math.sin(rad))
        cv2.line(frame, (CENTRE, CENTRE), (sx, sy), SWEEP_COLOUR, SWEEP_WIDTH, cv2.LINE_AA)

        # ── Centre dot ────────────────────────────────────────────────────
        cv2.circle(frame, (CENTRE, CENTRE), 3, SWEEP_COLOUR, -1)

        # ── HUD ───────────────────────────────────────────────────────────
        rng_str = f"{range_m} m" if range_m > 0 else "---"
        hdg_str = f"{heading_deg:.1f} deg" if heading_deg >= 0.0 else "---"

        hud_lines = [
            f"Rotation : {rot_count}",
            f"Spokes   : {spoke_count}",
            f"Rate     : {spoke_rate:.1f} spk/s",
            f"Range    : {rng_str}",
            f"Heading  : {hdg_str}",
            f"Gain     : {self.gain:.2f}x",
            f"[+/-] gain   [C] clear   [Q] quit",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (8, 18 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, HUD_COLOUR, 1, cv2.LINE_AA)

        return frame

    def clear(self):
        """Erase canvas and all slot memories."""
        with self._lock:
            self._canvas[:] = 0
            self._slot_pixels = [None] * NAVICO_SPOKES_RAW


# ─────────────────────────────────────────────────────────────────────────────
# ROS subscriber callback
# ─────────────────────────────────────────────────────────────────────────────

def make_callback(display):
    def callback(msg):
        display.feed_spoke(
            angle_raw       = int(msg.angle_raw),
            angle_deg       = float(msg.angle_deg),
            range_m         = int(msg.range_m),
            samples         = bytes(msg.samples),
            heading_deg     = float(msg.heading_deg),
            rotation_number = int(msg.rotation_number),
        )
    return callback


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rospy.init_node("halo_radar_viz", anonymous=False)

    topic        = rospy.get_param("~topic", "/radar/spoke")
    initial_gain = float(rospy.get_param("~gain", 1.0))

    rospy.loginfo(f"Halo PPI Visualiser")
    rospy.loginfo(f"  Topic  : {topic}")
    rospy.loginfo(f"  Gain   : {initial_gain}")
    rospy.loginfo(f"  Erase  : per-slot (spoke replaced only when antenna revisits bearing)")

    display = PPIDisplay(initial_gain=initial_gain)

    rospy.Subscriber(topic, RadarSpoke, make_callback(display), queue_size=512)

    win_name = "Halo Radar — PPI"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, DISPLAY_SIZE, DISPLAY_SIZE)

    frame_delay = max(1, int(1000 / FPS_TARGET))

    rospy.loginfo("Display open. Press Q or ESC to quit.")

    while not rospy.is_shutdown():
        frame = display.render_frame()
        cv2.imshow(win_name, frame)

        key = cv2.waitKey(frame_delay) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            display.gain = min(display.gain + 0.1, 5.0)
        elif key == ord('-'):
            display.gain = max(display.gain - 0.1, 0.1)
        elif key in (ord('c'), ord('C')):
            display.clear()

    cv2.destroyAllWindows()
    rospy.loginfo("Visualiser closed.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass