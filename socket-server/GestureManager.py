import Quartz
import AppKit
import time

def press_cmd_tab():
    # Press Cmd down
    Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                       Quartz.CGEventCreateKeyboardEvent(None, AppKit.kVK_Command, True))
    # Press Tab
    Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                       Quartz.CGEventCreateKeyboardEvent(None, AppKit.kVK_Tab, True))
    time.sleep(0.05)
    # Release Tab
    Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                       Quartz.CGEventCreateKeyboardEvent(None, AppKit.kVK_Tab, False))
    # Release Cmd
    Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                       Quartz.CGEventCreateKeyboardEvent(None, AppKit.kVK_Command, False))

def press_cmd_shift_tab():
    # Press Cmd + Shift down
    Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                       Quartz.CGEventCreateKeyboardEvent(None, AppKit.kVK_Command, True))
    Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                       Quartz.CGEventCreateKeyboardEvent(None, AppKit.kVK_Shift, True))
    # Press Tab
    Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                       Quartz.CGEventCreateKeyboardEvent(None, AppKit.kVK_Tab, True))
    time.sleep(0.05)
    # Release Tab
    Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                       Quartz.CGEventCreateKeyboardEvent(None, AppKit.kVK_Tab, False))
    # Release Shift and Cmd
    Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                       Quartz.CGEventCreateKeyboardEvent(None, AppKit.kVK_Shift, False))
    Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                       Quartz.CGEventCreateKeyboardEvent(None, AppKit.kVK_Command, False))