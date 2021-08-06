import cv2

vidcap = cv2.VideoCapture('./data/videos/80.mp4')

count = 1
while(vidcap.isOpened()):
        ret, img = vidcap.read()
        cv2.imwrite('./data/videos/img80/%d.png' % count, img)
        cv2.imshow('frame', img)
        cv2.waitKey(25)
        print('saved image %d.png' % count)
        count += 1
vidcap.release()