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
txAptPos = FocTxDataset['txAptPos'];
rcvdata = FocTxDataset['rcvdata'];
apod = FocTxDataset['apod'];
txBeamOrigins = FocTxDataset['txBeamOrigins'];
tx_dir = FocTxDataset['tx_dir'][0];
tx_focDepth = FocTxDataset['tx_focDepth'][0][0];
c = FocTxDataset['c'][0][0];

# Points to Focus and Get Image At
dov = 36e-3; # Max Depth [m]
num_x = 201; num_z = 601;
xlims = (12.7e-3)*np.array([-1, 1]);
zlims = np.array([4e-3, 36e-3]);
cbfm = 1540; # sound speed [m/s]

# Select Subset of Transmit Elements
tx_evts = np.arange(0,128,1);
txBeamOrigins = txBeamOrigins[tx_evts,:];
apod = apod[tx_evts,:];
rxdata_h = rcvdata[:,:,tx_evts];
del rcvdata;

# Aperture Definition
fTx = 6e6; # frequency [Hz]
fBW = 6e6; # bandwidth [Hz]
lmbda = cbfm/fTx; # wavelength [m]
pitch = np.mean(np.diff(rxAptPos[:,0])); # element spacing [m]
no_elements = rxAptPos.shape[0]; # number of elements
xpos = np.arange(-(no_elements-1)/2,1+(no_elements-1)/2)*pitch; # element position [m]

# Transmit Impulse Response in Frequency Domain
nt = t.size; # [s]
fs = 1/np.mean(np.diff(t)); # [Hz]

# Construct Transmit Beamforming Delays
delays = np.zeros((txBeamOrigins.shape[0],txAptPos.shape[0]));
for tx_idx in np.arange(tx_evts.size):
    # transmit aperture locations
    txAptPosRelToCtr = txAptPos - \
        np.ones((txAptPos.shape[0],1)) * txBeamOrigins[tx_idx,:];
    txFocRelToCtr = tx_focDepth * \
        np.ones((txAptPos.shape[0],1)) * tx_dir/np.linalg.norm(tx_dir);
    txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;
    # positive value is time delay, negative is time advance
    delays[tx_idx,:] = (np.sqrt(np.sum(txFocRelToCtr**2, axis=1)) - \
        np.sqrt(np.sum(txFocRelToAptPos**2, axis=1)))/c;

# Recovered Multistatic Dataset
rf_decoded = refocus_decode(rxdata_h,fs*delays,apod=apod,fHPF=(1e6)/fs);

# Points to Focus and Get Image At
x_img = np.linspace(xlims[0], xlims[1], num_x);
z_img = np.linspace(zlims[0], zlims[1], num_z);
dBrange = np.array([-80, 0]); c = 1540;

# Multistatic Synthetic Aperture Image Reconstruction
X, Y, Z = np.meshgrid(x_img, 0, z_img);
foc_pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()));
focData = bfm_fs(t, rf_decoded, foc_pts, rxAptPos, txAptPos=txAptPos, dc_rx=0, dc_tx=0, speed_of_sound=c);
img_h = np.reshape(focData, (x_img.size, z_img.size)).T;
plt.figure(); imagesc(1000*x_img, 1000*z_img, \
    20*np.log10(np.abs(img_h)/np.max(np.abs(img_h))), dBrange);
plt.title('Adjoint-Based REFoCUS Reconstruction')
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]'); plt.colorbar(); plt.show();
