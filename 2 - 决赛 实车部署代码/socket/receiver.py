import cv2
from npsocket import NumpySocket
import datetime
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / 'out_videos'
SAVE_DIR.mkdir(exist_ok=True)

sock_receiver = NumpySocket()
sock_receiver.initalize_receiver('192.168.2.1', 7777)

def out_name():
    return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '.mp4'



if __name__ == '__main__':

    first_frame = True
    while True:
        conn_state, frame = sock_receiver.receive_array()

        if conn_state and first_frame: # 连接状态,并为视频第一帧, 新建保存文件
            first_frame = False
            out_file = str(SAVE_DIR / out_name())
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = frame.shape[:2]
            out_writer = cv2.VideoWriter(out_file, fourcc, 24, (width, height), True)

        if conn_state: # 写入帧
            cv2.imshow('frame', frame)
            cv2.waitKey(2)
            out_writer.write(frame)

        else:   # 连接断开，保存视频文件
            first_frame = True
            try:
                out_writer.release()
                print(f'Save video at: {out_file}')
            except:
                pass

            # 是否继续等待下一个视频
            # c = input("Continue? ([y]/n)? ")
            # if c in ['n', 'N']:
            #     exit()




