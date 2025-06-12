from pynput.keyboard import Controller, Key
import time

keyboard = Controller()

def press(keys):
    for key in keys:
        keyboard.press(key)
    time.sleep(0.1)
    for key in keys:
        keyboard.release(key)
    
def perform(yaw, pitch, roll, shoulder_roll):
    is_ready = False


    yaw_sens = 25

    if yaw >= 90 + yaw_sens:
        if is_ready:
            print("yaw right")
            press([Key.cmd, Key.tab])
            is_ready = False
    elif yaw <= 90 - yaw_sens:
        if is_ready:
            print("yaw left")
            press([Key.cmd, Key.shift, Key.tab])
            is_ready = False
    else:
        is_ready = True