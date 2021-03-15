# Setting up all folders we can import from by adding them to python path
import sys, os, pdb
curr_path = os.getcwd();
sys.path.append(curr_path+'/beamforming_functions/');

# Importing stuff from all folders in python path
import numpy as np
from beamforming_functions import *
import scipy.io as sio
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Load Channel Data
FocTxDataset = sio.loadmat('FocTxDataset.mat');
t = FocTxDataset['t'][0];
rxAptPos = FocTxDataset['rxAptPos'];
txAptPos = FocTxDataset['txAptPos'];
rcvdata = FocTxDataset['rcvdata'];
apod = FocTxDataset['apod'];
txBeamOrigins = FocTxDataset['txBeamOrigins'];
tx_dir = FocTxDataset['tx_dir'][0];
tx_focDepth = FocTxDataset['tx_focDepth'][0][0];
c = FocTxDataset['c'][0][0];

# Options for Virtual Source Reconstruction
opt = 2; # 1/2 are with/without edge waves, respectively
tx_idx = 63; # which transmit beam to show partial image of

# Get All Dimension Information
nt, nRx, nTx = rcvdata.shape;
no_active_elements = np.max(np.sum(apod!=0,axis=0));
pitch = np.mean(np.diff(txAptPos[:,0]));

# Subsample Transmit Events
dwnsmp = 1; # Downsampling Factor
rcvdata = rcvdata[:,:,::dwnsmp];
txBeamOrigins = txBeamOrigins[::dwnsmp,:];

# Points to Focus and Get Image At
dBrange = np.array([-80, 0]); num_r = 600;
rlims = np.array([4e-3, 36e-3]); # Imaging Depths
r_img = np.linspace(rlims[0], rlims[1], num_r);
tx_origin_x = txBeamOrigins[:,0]; # Transmit Beam Origin [m]

# Scan Lines and Image Coordinates
X, R_IMG = np.meshgrid(txAptPos[:,0], r_img);
Y = np.zeros(X.shape); Z = R_IMG;
foc_pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()));

# Virtual Source Synthetic Aperture Focusing
img_retroTx = np.zeros((r_img.size, (txAptPos[:,0]).size, tx_origin_x.size), dtype = np.dtype('complex64'));
includePtInRecon = np.zeros((r_img.size, (txAptPos[:,0]).size, tx_origin_x.size));
print("Beginning Virtual Source Synthetic Aperture Beamforming");
for kk in np.arange(tx_origin_x.size):
    focData, includePoint = \
        focus_retroTx(t, rcvdata[:,:,kk], foc_pts, rxAptPos, \
        txBeamOrigins[kk,:], tx_dir, tx_focDepth, \
        no_active_elements*pitch/2, 0, c);
    img_retroTx[:,:,kk] = np.sum(np.reshape(focData, \
        (r_img.size, txAptPos[:,0].size, rxAptPos.shape[0])), axis=2);
    includePtInRecon[:,:,kk] = np.reshape(includePoint, \
        (r_img.size, txAptPos[:,0].size)) | \
        (np.abs(X-tx_origin_x[kk]) < (1-(1/(2*dwnsmp)))*np.mean(np.diff(tx_origin_x)));
    print("Transmit Beam at x = "+str(tx_origin_x[kk]));
numTxForReconPt = np.sum(includePtInRecon, axis=2);

# High-Pass Filter Each Image Axially
img_retroTx_orig = img_retroTx;
b, a = butter(10, 0.1, 'highpass');
img_retroTx = filtfilt(b, a, img_retroTx_orig, axis=0);

## Virtual Source Synthetic Aperture Reconstruction
if opt == 1: # Virtual Source Reconstruction with Edge Waves
    img_h = np.sum(img_retroTx, axis=2);
    # L1 Coherence Factor
    l1cf = np.abs(np.sum(img_retroTx,axis=2))/np.sum(np.abs(img_retroTx),axis=2);
elif opt == 2: # Virtual Source Reconstruction without Edge Waves
    img_h = np.sum(img_retroTx*includePtInRecon, axis=2) / (numTxForReconPt);
    # L1 Coherence Factor
    l1cf = np.abs(np.sum(img_retroTx*includePtInRecon, axis=2)) / \
        np.sum(np.abs(img_retroTx*includePtInRecon), axis=2);

# Need to Perform Scan Conversion:
x_img = np.linspace(np.min(X), np.max(X), 300);
z_img = np.linspace(np.min(Z), np.max(Z), 1000);
X_IMG, Z_IMG = np.meshgrid(x_img, z_img);
X, R_IMG = np.meshgrid(txAptPos[:,0], r_img);
IMG = griddata((X.flatten(), R_IMG.flatten()), np.abs(img_h.flatten()), (X_IMG, Z_IMG), method = 'linear');
L1CF = griddata((X.flatten(), R_IMG.flatten()), l1cf.flatten(), (X_IMG, Z_IMG), method = 'linear');

# Show Scan Converted Image
plt.figure(); imagesc(1000*x_img, 1000*z_img, 20*np.log10(IMG/np.max(IMG)), dBrange);
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('Virtual Source Reconstruction'); plt.colorbar(); plt.show();

# Show CF Image
plt.figure(); imagesc(1000*txAptPos[:,0], 1000*r_img, L1CF, np.array([0,1]));
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('L1 Coherence Factor'); plt.colorbar(); plt.show();

## Show Rx Focused Image for Individual Tx Beams
# Isolating Rx Focused Image for Single Tx Beam
img_h_singleTx = img_retroTx[:,:,int(np.round(tx_idx/dwnsmp))];
img_h_singleTx_NoEdgeWaves = img_h_singleTx * \
    includePtInRecon[:,:,int(np.round(tx_idx/dwnsmp))];

# Need to Perform Scan Conversion:
x_img = np.linspace(np.min(X), np.max(X), 300);
z_img = np.linspace(np.min(Z), np.max(Z), 1000);
X_IMG, Z_IMG = np.meshgrid(x_img, z_img);
X, R_IMG = np.meshgrid(txAptPos[:,0], r_img);
img_h_singleTx_scan_conv = griddata((X.flatten(), R_IMG.flatten()), \
    np.abs(img_h_singleTx.flatten()), (X_IMG, Z_IMG), method = 'linear');
img_h_singleTx_NoEdgeWaves_scan_conv = griddata((X.flatten(), R_IMG.flatten()), \
    np.abs(img_h_singleTx_NoEdgeWaves.flatten()), (X_IMG, Z_IMG), method = 'linear');

# Show Scan Converted Image
plt.figure(); plt.subplot(1,2,1); imagesc(1000*x_img, 1000*z_img, \
    20*np.log10(img_h_singleTx_scan_conv/ \
    np.max(img_h_singleTx_scan_conv)), dBrange);
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('Single TX with Edge Waves'); plt.colorbar();
plt.subplot(1,2,2); imagesc(1000*x_img, 1000*z_img, \
    20*np.log10(img_h_singleTx_NoEdgeWaves_scan_conv/ \
    np.max(img_h_singleTx_NoEdgeWaves_scan_conv)), dBrange);
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('Single TX without Edge Waves'); plt.colorbar(); plt.show();
