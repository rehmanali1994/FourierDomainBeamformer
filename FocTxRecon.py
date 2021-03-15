# Setting up all folders we can import from by adding them to python path
import sys, os, pdb
curr_path = os.getcwd();
sys.path.append(curr_path+'/beamforming_functions/');

# Importing stuff from all folders in python path
import numpy as np
from beamforming_functions import *
import scipy.io as sio
from scipy.signal import hilbert
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Load Channel Data
FocTxDataset = sio.loadmat('FocTxDataset.mat');
t = FocTxDataset['t'][0];
rxAptPos = FocTxDataset['rxAptPos'];
rcvdata = FocTxDataset['rcvdata'];
apod = FocTxDataset['apod'];
txBeamOrigins = FocTxDataset['txBeamOrigins'];
tx_dir = FocTxDataset['tx_dir'][0];
tx_focDepth = FocTxDataset['tx_focDepth'][0][0];
c = FocTxDataset['c'][0][0];

# Get All Dimension Information
nt, nRx, nTx = rcvdata.shape;

# Points to Focus and Get Image At
dBrange = np.array([-80, 0]); c = 1540;
num_r = 600; rlims = np.array([4e-3, 36e-3]);
r_img = np.linspace(rlims[0], rlims[1], num_r);

# Focused Transmit Image Reconstruction
tx_origin_x = txBeamOrigins[:,0]; # Transmit Beam Origin [m]
img_h = np.zeros((r_img.size, tx_origin_x.size), dtype = np.dtype('complex64'));
print("Beginning Dynamic Receive Beamforming");
for kk in np.arange(tx_origin_x.size):
    X = tx_origin_x[kk]*np.ones(r_img.shape);
    Y = np.zeros(X.shape); Z = r_img;
    foc_pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()));
    focData = focus_data(t, rcvdata[:,:,kk], foc_pts, rxAptPos, \
        tx_center = txBeamOrigins[kk,:], dc = 0, speed_of_sound = c);
    img_h[:, kk] = np.sum(focData, axis = 1);
    print("Image line at x = "+str(tx_origin_x[kk]));

# Upsample the Image
upsamp_x = 4; upsamp_z = 2;
x_img = np.linspace(tx_origin_x[0],tx_origin_x[-1],(tx_origin_x.size-1)*upsamp_x+1);
z_img = np.linspace(r_img[0],r_img[-1],(r_img.size-1)*upsamp_z+1);
TX_ORIG_X, R_IMG = np.meshgrid(tx_origin_x, r_img);
X_IMG, Z_IMG = np.meshgrid(x_img, z_img);
IMG = np.clip(griddata((TX_ORIG_X.flatten(), R_IMG.flatten()), \
    np.abs(img_h.flatten()), (X_IMG, Z_IMG), method = 'linear'),0,None);

# Show Scan Converted Image
plt.figure(); imagesc(1000*x_img, 1000*z_img, \
    20*np.log10(IMG/np.max(IMG)), dBrange);
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('DAS Beamforming'); plt.colorbar(); plt.show();
