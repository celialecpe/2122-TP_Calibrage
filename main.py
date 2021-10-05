import numpy as np
import cv2
from matplotlib import pyplot as plt

# cap = cv2.VideoCapture(0)

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
def add_square(im, u1, u2, c, w=5):
    im[int(u1)-w:int(u1)+w,int(u2)-w:int(u2)+w] = c

# for i in range(len(coord_px0)):
#     mire0[int(coord_px0[i][0][1])-5:int(coord_px0[i][0][1])+5,int(coord_px0[i][0][0])-5:int(coord_px0[i][0][0])+5] = [255,0,0]
for i in range(len(coord_px1)):
    mire1[int(coord_px1[i][0][1])-5:int(coord_px1[i][0][1])+5,int(coord_px1[i][0][0])-5:int(coord_px1[i][0][0])+5] = [255,0,0]
################################

#Creation of coord_px
coord_px = np.zeros([nbInterTot,2])
for i in range(nbInterTot):
    if i < nbInter1mat :
        #notre repère
        coord_px[i][0] = int(coord_px0[i][0][1])
        coord_px[i][1] = int(coord_px0[i][0][0])
        

        # #Leur repere
        # coord_px[i][0] = int(coord_px0[i][0][0])
        # coord_px[i][1] = int(coord_px0[i][0][1])
    else :
        #notre repère
        coord_px[i][0] = int(coord_px1[i-nbInter1mat][0][1])
        coord_px[i][1] = int(coord_px1[i-nbInter1mat][0][0])

        # #leur repère
        # coord_px[i][0] = int(coord_px1[i-nbInter1mat][0][0])
        # coord_px[i][1] = int(coord_px1[i-nbInter1mat][0][1])

################################


#Creation of coord_mm
coord_mm = np.zeros([nbInterTot,3])
delta_z = -120
for i in range(nbInterTot):
    if i<nbInter1mat:
        #Notre repere
        x = i//7 * 20
        y = i%7 * 20

        # #Leur repere
        # y = i//7 * 20
        # x = i%7 * 20

        
        coord_mm[i] = [x, y, 0]
    else:
        #notre repere
        x = (i-nbInter1mat)//7 * 20
        y = (i-nbInter1mat)%7 * 20
        coord_mm[i] = [x, y, delta_z]

        # #Leur repere
        # y = (i-nbInter1mat)//7 * 20
        # x = (i-nbInter1mat)%7 * 20
        # coord_mm[i] = [x, y, -delta_z]
################################

#Tsai method
#Notre repere
i1, i2 = mire0.shape[:-1]
i1 = i1/2
i2 = i2/2

#Leur repere
# i2, i1 = mire0.shape[:-1]
# i2 = i2/2
# i1 = i1/2

u_tilde = np.zeros([nbInterTot,2])
for i in range(nbInterTot):
    u_tilde[i][0] = coord_px[i][0] - i1
    u_tilde[i][1] = coord_px[i][1] - i2

U1 = np.zeros((nbInterTot, 1))
for i in range(nbInterTot):
    U1[i] = u_tilde[i][0]

A = np.zeros((nbInterTot, 7))
for i in range(nbInterTot):
    A[i] = [u_tilde[i][1]*coord_mm[i][0], u_tilde[i][1]*coord_mm[i][1], u_tilde[i][1]*coord_mm[i][2],
            u_tilde[i][1],
            -u_tilde[i][0]*coord_mm[i][0], -u_tilde[i][0]*coord_mm[i][1], -u_tilde[i][0]*coord_mm[i][2]]


L = np.dot(np.linalg.pinv(A), U1)

norme_oc2 = 1 / (np.sqrt(L[4]*L[4] + L[5]*L[5] + L[6]*L[6]))
beta = norme_oc2 * np.sqrt(L[0]*L[0] + L[1]*L[1] + L[2]*L[2])
oc2 = -norme_oc2
oc1 = L[3] * oc2 / beta
r11 = L[0] * oc2 / beta
r12 = L[1] * oc2 / beta
r13 = L[2] * oc2 / beta
r21 = L[4] * oc2
r22 = L[5] * oc2
r23 = L[6] * oc2


colonne3 = np.cross(np.transpose(np.array([r11,r12,r13])),
                    np.transpose(np.array([r21,r22,r23])))


r31 = colonne3[0][0]
r32 = colonne3[0][1]
r33 = colonne3[0][2]
phi = -np.arctan(r23/r33)
gamma = -np.arctan(r12/r11)
omega = np.arctan(r13/(-r23*np.sin(phi)+r33*np.cos(phi)))


B = np.zeros((nbInterTot, 2))
for i in range(nbInterTot):
    B[i] = [u_tilde[i][1], 
            -(r21*coord_mm[i][0] + r22*coord_mm[i][1] + r23*coord_mm[i][2] + oc2)]

R = np.zeros((nbInterTot, 1))
for i in range(nbInterTot):
    R[i] = -u_tilde[i][1] * (r31*coord_mm[i][0] + r32*coord_mm[i][1] + r33*coord_mm[i][2])

M = np.dot(np.linalg.pinv(B), R)
oc3 = M[0]
f2 = M[1]
f = 4 #mm
f1 = beta * f2

s2 = f/f2
s1 = f/f1


print("")
print("beta :", beta)
print("oc1 :", oc1)
print("oc2 :", oc2)
print("oc3 :", oc3)
print("r11 :", r11)
print("r12 :", r12)
print("r13 :", r13)
print("r21 :", r21)
print("r22 :", r22)
print("r23 :", r23)
print("r31 :", r31)
print("r32 :", r32)
print("r33 :", r33)
print("f :", f)
print("s1 :", s1)
print("s2 :", s2)

print(phi/3.14*180, gamma/3.14*180, omega/3.14*180)

# Projecton
Mint = np.array([[f1, 0, i1, 0],
                 [0, f2, i2, 0], 
                 [0, 0, 1, 0]])
Mext = np.array([[r11, r12, r13, oc1], 
                 [r21, r22, r23, oc2],
                 [r31, r32, r33, oc3],
                 [0, 0, 0, 1]])

M = np.dot(Mint, Mext)

print("Mint :", Mint.shape)
print("Mext :", Mext.shape)
print("M :", M.shape)

# x = np.array([coord_mm[1][0], coord_mm[1][1], coord_mm[1][2], 1])
for i in range(nbInter1mat):
    x = np.array([coord_mm[i][0], coord_mm[i][1], coord_mm[i][2], 1])
    u = np.dot(M, x)

    print(u)
    u1 = u[0]/u[2]
    u2 = u[1]/u[2]

    print(u1, u2)
    print(coord_mm[i], coord_px[i])


    add_square(mire0, u1, u2, [0, 255, 0])



while(True):
    # ret, frame = cap.read() #1 frame acquise à chaque iteration
    # cv2.imshow('Capture_Video', frame) #affichage

    cv2.imshow('mire0',mire0)
    plt.show()

    key = cv2.waitKey(1) #on évalue la touche pressée
    if key & 0xFF == ord('q'): #si appui sur 'q'
        break #sortie de la boucle while



cv2.destroyAllWindows()
