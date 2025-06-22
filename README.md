# takealook-sdk
takealook의 웹캠 데이터 수집 및 자세 판별 함수들 제공

## 0. Require env.
| **라이브러리**   | **버전**       |
|-------------|------------|
| OpenCV (cv2) | 4.11.0     |
| MediaPipe   | 0.10.21    |
| NumPy       | 1.26.4     |
| Python3      | 3.10.13  |

## 1. How to run
1. satisfy the require env.
2. run `SocketServer.py`

## 2. about return value
### For 10 sec.
For 10 seconds, this server doesn't return anything.  
This server get a data for avg. of user's pose(roll, yaw, pitch, etc.)  
The average move a new data by applying ???.  

### After 10 sec.
After 10 seconds, this server returns a result value that `[brightness, face distance, blink per 10 sec., shoulder roll, turtle neck, gesture]`  
The corresponding value is data from 0 to 3 that has been determined.  

## for debuger
**Do it step-by-step**
1. run `SocketClient.py`
2. run `SocketServer.py`
