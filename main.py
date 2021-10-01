import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

# plt.figure()

#orb = cv2.ORB_create()

mire0 = cv2.imread('./capture_mire_0.png')
mire1 = cv2.imread('./capture_mire_1.png')

#Parameters
nbInterX = 7
nbInterY = 7
nbInter1mat = nbInterX*nbInterY
nbInterTot = nbInter1mat*2
################################

#Intersection detection
found_0, coord_px0 = cv2.findChessboardCorners(mire0,(nbInterX,nbInterY))
found_1, coord_px1 = cv2.findChessboardCorners(mire1,(nbInterX,nbInterY))
################################

#Intersection display
for i in range(len(coord_px0)):
    mire0[int(coord_px0[i][0][1])-5:int(coord_px0[i][0][1])+5,int(coord_px0[i][0][0])-5:int(coord_px0[i][0][0])+5] = [255,0,0]
for i in range(len(coord_px1)):
    mire1[int(coord_px1[i][0][1])-5:int(coord_px1[i][0][1])+5,int(coord_px1[i][0][0])-5:int(coord_px1[i][0][0])+5] = [255,0,0]
################################

#Creation of coord_px
coord_px = np.zeros([nbInterTot,2])
for i in range(nbInterTot):
    if i < nbInter1mat :
        coord_px[i][0] = coord_px0[i][0][1]
        coord_px[i][1] = coord_px0[i][0][0]
    else :
        coord_px[i][0] = coord_px1[i-nbInter1mat][0][1]
        coord_px[i][1] = coord_px1[i-nbInter1mat][0][0]
################################

#Creation of coord_mm
coord_mm = np.zeros([nbInterTot,3])
delta_z = -120
for i in range(nbInterTot):
    if i<nbInter1mat:
        x = i//7 * 20
        y = i%7 * 20
        coord_mm[i] = [x, y, 0]
    else:
        x = (i-nbInter1mat)//7 * 20
        y = (i-nbInter1mat)%7 * 20
        coord_mm[i] = [x, y, delta_z]
################################


# print(coord_px)
print(coord_mm)


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
