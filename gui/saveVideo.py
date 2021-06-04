import cv2

def captureVideo(cap, writer):
    while True:
        ret,frame = cap.read()

        writer.write(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


def recordVideo(inputCam):
    cap = cv2.VideoCapture(f'udpsrc port=5{inputCam}00 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter('underwaterVideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

    captureVideo(cap, writer)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()