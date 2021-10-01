import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

# plt.figure()

#orb = cv2.ORB_create()

mire0 = cv2.imread('./capture_mire_0.png')
mire1 = cv2.imread('./capture_mire_1.png')
# cv2.imshow('mire0',mire0)
# cv2.imshow('mire1',mire1)

found_0, coord_px0 = cv2.findChessboardCorners(mire0,(7,7))
found_1, coord_px1 = cv2.findChessboardCorners(mire1,(7,7))


for i in range(len(coord_px0)):
    mire0[int(coord_px0[i][0][1])-5:int(coord_px0[i][0][1])+5,int(coord_px0[i][0][0])-5:int(coord_px0[i][0][0])+5] = [255,0,0]

for i in range(len(coord_px1)):
    mire1[int(coord_px1[i][0][1])-5:int(coord_px1[i][0][1])+5,int(coord_px1[i][0][0])-5:int(coord_px1[i][0][0])+5] = [255,0,0]

coord_px = np.zeros([98,2])
for i in range(98):
    if i < 49 :
        coord_px[i][0] = coord_px0[i][0][1]
        coord_px[i][1] = coord_px0[i][0][0]
    else :
        coord_px[i][0] = coord_px1[i-49][0][1]
        coord_px[i][1] = coord_px1[i-49][0][0]

coord_mm = np.zeros([98,3])
delta_z = -120
for i in range(98):
    if i<49:
        x = i//7 * 20
        y = i%7 * 20
        coord_mm[i] = [x, y, 0]
    else:
        x = (i-49)//7 * 20
        y = (i-49)%7 * 20
        coord_mm[i] = [x, y, delta_z]
    # coord_mm[i][0][1] = i/7 * 20
    # coord_mm[i][0][2] = 0

    # coord_mm[i+49][0][0] = i%7 * 20
    # coord_mm[i+49][0][1] = i/7 * 20
    # coord_mm[i+49][0][2] = delta_z


# print(coord_px)
print(coord_mm)


# print(coord_px0)

while(True):
    # ret, frame = cap.read() #1 frame acquise à chaque iteration
    # cv2.imshow('Capture_Video', frame) #affichage

    cv2.imshow('mire0',mire0)
    cv2.imshow('mire1',mire1)

    key = cv2.waitKey(1) #on évalue la touche pressée
    if key & 0xFF == ord('q'): #si appui sur 'q'
        break #sortie de la boucle while


cap.release()
cv2.destroyAllWindows()
