import serial
import struct
import time
'''
Servo 1: Horizontal movement
Servo 2: Vertical movement
'''
port = serial.Serial("/dev/rrc", baudrate=1000000, timeout=5)
port.rts = False
port.dtr = False
#port.open()

crc8_table = [
    0, 94, 188, 226, 97, 63, 221, 131, 194, 156, 126, 32, 163, 253, 31, 65,
    157, 195, 33, 127, 252, 162, 64, 30, 95, 1, 227, 189, 62, 96, 130, 220,
    35, 125, 159, 193, 66, 28, 254, 160, 225, 191, 93, 3, 128, 222, 60, 98,
    190, 224, 2, 92, 223, 129, 99, 61, 124, 34, 192, 158, 29, 67, 161, 255,
    70, 24, 250, 164, 39, 121, 155, 197, 132, 218, 56, 102, 229, 187, 89, 7,
    219, 133, 103, 57, 186, 228, 6, 88, 25, 71, 165, 251, 120, 38, 196, 154,
    101, 59, 217, 135, 4, 90, 184, 230, 167, 249, 27, 69, 198, 152, 122, 36,
    248, 166, 68, 26, 153, 199, 37, 123, 58, 100, 134, 216, 91, 5, 231, 185,
    140, 210, 48, 110, 237, 179, 81, 15, 78, 16, 242, 172, 47, 113, 147, 205,
    17, 79, 173, 243, 112, 46, 204, 146, 211, 141, 111, 49, 178, 236, 14, 80,
    175, 241, 19, 77, 206, 144, 114, 44, 109, 51, 209, 143, 12, 82, 176, 238,
    50, 108, 142, 208, 83, 13, 239, 177, 240, 174, 76, 18, 145, 207, 45, 115,
    202, 148, 118, 40, 171, 245, 23, 73, 8, 86, 180, 234, 105, 55, 213, 139,
    87, 9, 235, 181, 54, 104, 138, 212, 149, 203, 41, 119, 244, 170, 72, 22,
    233, 183, 85, 11, 136, 214, 52, 106, 43, 117, 151, 201, 74, 20, 246, 168,
    116, 42, 200, 150, 21, 75, 169, 247, 182, 232, 10, 84, 215, 137, 107, 53
]

def checksum_crc8(data):
    check = 0
    for b in data:
        check = crc8_table[check ^ b]
    return check & 0x00FF

def buf_write(func, data):
    buf = [0xAA, 0x55, int(func)]
    buf.append(len(data))
    buf.extend(data)
    buf.append(checksum_crc8(bytes(buf[2:])))
    port.write(buf)

def pwm_servo_set_position(duration, positions):
    duration = int(duration * 1000)
    data = [0x01, duration & 0xFF, 0xFF & (duration >> 8), len(positions)]
    for i in positions:
        data.extend(struct.pack("<BH", i[0], i[1]))
    buf_write(4, data)

if __name__ == "__main__":
    time.sleep(1)
    while True:
        pwm_servo_set_position(2, [[1,500]]) #Moves servo 1 one direction
        time.sleep(2)
        pwm_servo_set_position(2, [[1,2500]]) #Moves servo 1 in the other direction
        time.sleep(2)
        pwm_servo_set_position(2, [[1,1500]]) #Recenter the camera
        time.sleep(2)
        pwm_servo_set_position(2, [[2,1500]]) #Moves servo 2 in one direction
        time.sleep(2)
        pwm_servo_set_position(2, [[2,2500]]) #Moves servo 2 in the other direction
        time.sleep(2)
        pwm_servo_set_position(2, [[2,2200]]) #Recenter the camera
        time.sleep(2)
        pwm_servo_set_position(2, [[1,500],[2,1500]])
        time.sleep(2)
        pwm_servo_set_position(2, [[1,2500],[2,2500]])
        time.sleep(2)
        pwm_servo_set_position(2, [[1,1500],[2,2200]]) #Recenter the camera
        time.sleep(2)
