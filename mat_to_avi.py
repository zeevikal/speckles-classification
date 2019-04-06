import scipy.io
import cv2

# cap = cv2.VideoCapture('data/VideoFile_fps100_0001.mat')
# count = 0
# while cap.isOpened():
#     ret,frame = cap.read()
#     cv2.imshow('window-name',frame)
#     # cv2.imwrite("frame%d.jpg" % count, frame)
#     # count = count + 1
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()  # destroy all the opened windows

mat = scipy.io.loadmat('data/24032019/VideoFile_fps100_0001.mat')
img_reshaped = mat['Video_fps100_0001'][:, :]
print(img_reshaped.shape)
for i in range(img_reshaped.shape[2]):
    cv2.imshow("segmented_map", mat['Video_fps100_0001'][:, :, i])
    cv2.waitKey(0)

