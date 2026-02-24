# Simrad Halo radar spoke
# Place in: <your_package>/msg/RadarSpoke.msg
# Add to CMakeLists.txt:  add_message_files(FILES RadarSpoke.msg)
# Published on topic:     /radar/spoke

std_msgs/Header header        # stamp = rospy.Time.now(), frame_id = "radar"

uint16  spoke_number          # scan_number from br4g_header (0–4095)
uint16  angle_raw             # raw angle (0–4095)
float32 angle_deg             # angle in degrees (0.0–359.9)
int32   range_m               # range in metres (0 = invalid)
float32 heading_deg           # heading in degrees (-1.0 = invalid/unavailable)
bool    heading_true          # true = True North, false = Magnetic

uint8   status                # 0x02 = transmitting
uint8   rotation_number       # which rotation this spoke belongs to (wraps at 255)

uint8[] samples               # 1024 intensity values, 0–255
                              # index 0 = nearest range cell, 1023 = furthest