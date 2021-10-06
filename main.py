#%% Import
import numpy as np
import cv2
from matplotlib import pyplot as plt

#%% Paramater declaration & intersection detection

# Parameters and image reading
mire0 = cv2.imread('./capture_mire_0.png')
mire1 = cv2.imread('./capture_mire_1.png')

nbInterX = 7
nbInterY = 7
nbInter1mat = nbInterX*nbInterY
nbInterTot = nbInter1mat*2

f = 4 # focal in mm

# Intersection detection
found_0, coord_px0 = cv2.findChessboardCorners(mire0,(nbInterX,nbInterY))
found_1, coord_px1 = cv2.findChessboardCorners(mire1,(nbInterX,nbInterY))

# Intersection display
def add_square(im, u1, u2, c, w=5):
    im[int(u1)-w:int(u1)+w,int(u2)-w:int(u2)+w] = c

for i in range(len(coord_px0)):
    add_square(mire0, coord_px0[i][0][1], coord_px0[i][0][0], [255,0,0], w=5)
for i in range(len(coord_px1)):
    add_square(mire1, coord_px1[i][0][1], coord_px1[i][0][0], [255,0,0], w=5)

# Creation of coord_px (pixel coordinate)
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


# Creation of coord_mm (world coordinate in mm)
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

#%% Tsai method

### STEP 1 : Solving AL = U1 ###

# Image center
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

# U1 vector (Nx1)
U1 = np.zeros((nbInterTot, 1))
for i in range(nbInterTot):
    U1[i] = u_tilde[i][0]

# A matrix (Nx7)
A = np.zeros((nbInterTot, 7))
for i in range(nbInterTot):
    A[i] = [u_tilde[i][1]*coord_mm[i][0], u_tilde[i][1]*coord_mm[i][1], u_tilde[i][1]*coord_mm[i][2],
            u_tilde[i][1],
            -u_tilde[i][0]*coord_mm[i][0], -u_tilde[i][0]*coord_mm[i][1], -u_tilde[i][0]*coord_mm[i][2]]

# Solution : L vector (Nx1)
L = np.dot(np.linalg.pinv(A), U1)

# Computing some intrinsect and extrasect parameters 
abs_oc2 = 1 / (np.sqrt(L[4]*L[4] + L[5]*L[5] + L[6]*L[6]))

beta = float(abs_oc2 * np.sqrt(L[0]*L[0] + L[1]*L[1] + L[2]*L[2])) # beta = f1/f2 = s2:s1

# 2 first coordinates of the world coordinate system's origin
oc2 = float(-abs_oc2)
oc1 = float(L[3] * oc2 / beta)

# Rotation matrix
r11 = float(L[0] * oc2 / beta)
r12 = float(L[1] * oc2 / beta)
r13 = float(L[2] * oc2 / beta)
r21 = float(L[4] * oc2)
r22 = float(L[5] * oc2)
r23 = float(L[6] * oc2)

# Last line of rotation matrix by the cross product of the two first lines
ligne3 = np.cross(np.transpose(np.array([r11,r12,r13])),
                    np.transpose(np.array([r21,r22,r23])))
r31 = ligne3[0]
r32 = ligne3[1]
r33 = ligne3[2]

# Rotation angles (in rad)
phi = float(-np.arctan(r23/r33))
gamma = float(-np.arctan(r12/r11))
omega = float(np.arctan(r13/(-r23*np.sin(phi)+r33*np.cos(phi))))

### STEP 2 : Solving BM = R ###

# B matrix (Nx2)
B = np.zeros((nbInterTot, 2))
for i in range(nbInterTot):
    B[i] = [u_tilde[i][1], 
            -(r21*coord_mm[i][0] + r22*coord_mm[i][1] + r23*coord_mm[i][2] + oc2)]

# R vector (Nx1)
R = np.zeros((nbInterTot, 1))
for i in range(nbInterTot):
    R[i] = -u_tilde[i][1] * (r31*coord_mm[i][0] + r32*coord_mm[i][1] + r33*coord_mm[i][2])

# Solution : M vector (Nx1)
M = np.dot(np.linalg.pinv(B), R)

# Computing the remaining intrinsect and extrasect parameters 

oc3 = float(M[0]) # Last coordinate

f2 = float(M[1])
f1 = beta * f2

# Pixel dimension
s1 = f/f1
s2 = f/f2

#%% Printing intrinsect and extrasect parameters

print("------INTRINSECT PARAMETERS------")
print("focal (in mm) :", f)

print("\nPIXEL DIMENSION")
print("beta (s2/s1):", beta)
print("s1 :", s1)
print("s2 :", s2)

print("\n------EXTRINSECT PARAMETERS------")

print("\nWORLD COORDINATE SYSTEM ORIGIN")
print("oc1 :", oc1)
print("oc2 :", oc2)
print("oc3 :", oc3)

print("\nROTATION MATRIX")
print("r11 :", r11, ", r12 :", r12, ", r13 :", r13)
print("r21 :", r21, ", r22 :", r22, ", r23 :", r23)
print("r31 :", r31, ", r32 :", r32, ", r33 :", r33)

print("\nphi (in deg):", phi/3.14*180)
print("gamma (in deg):", gamma/3.14*180)
print("omega (in deg):", omega/3.14*180)

#%% Projecton

# Intrinsect parameters matrix (3x4)
Mint = np.array([[f1, 0, i1, 0],
                 [0, f2, i2, 0], 
                 [0, 0, 1, 0]])

# Extrasect parameters (4x4)
Mext = np.array([[r11, r12, r13, oc1], 
                 [r21, r22, r23, oc2],
                 [r31, r32, r33, oc3],
                 [0, 0, 0, 1]])

# M matrix (3x4)
M = np.dot(Mint, Mext)

# Projecting world coordinate to pixel coordinate
print("\nProjection")
for i in range(nbInter1mat): 
    x = np.array([coord_mm[i][0],
                  coord_mm[i][1],
                  coord_mm[i][2],
                  1]) # world coordinate

    # pixel coordinate
    alpha_u = np.dot(M, x)
    u = np.array([float(alpha_u[0]/alpha_u[2]), float(alpha_u[1]/alpha_u[2])])

    print("World coordinates :", x[:-1], "| Pixel coordinates :", u)

    add_square(mire0, u[0], u[1], [0, 255, 0])

for i in range(nbInter1mat): 
    x = np.array([coord_mm[i+nbInter1mat][0],
                  coord_mm[i+nbInter1mat][1],
                  coord_mm[i+nbInter1mat][2],
                  1]) # world coordinate

    # pixel coordinate
    alpha_u = np.dot(M, x)
    u = np.array([float(alpha_u[0]/alpha_u[2]), float(alpha_u[1]/alpha_u[2])])

    print("World coordinates :", x[:-1], "| Pixel coordinates :", u)

    add_square(mire1, u[0], u[1], [0, 255, 0])


#%% Image display
while(True):
    cv2.imshow('mire0',mire0)
    cv2.imshow('mire1',mire1)
    plt.show()

    key = cv2.waitKey(1) #on évalue la touche pressée
    if key & 0xFF == ord('q'): #si appui sur 'q'
        break #sortie de la boucle while

cv2.destroyAllWindows()

#%% cv2.CalibrateCamera
print(coord_mm.shape)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(coord_mm, coord_px, mire0.shape[:-1], None, None)

print(rvecs, tvecs)