# Setting up all folders we can import from by adding them to python path
import sys, os, pdb
curr_path = os.getcwd();
sys.path.append(curr_path+'/beamforming_functions/');

# Importing stuff from all folders in python path
import numpy as np
from beamforming_functions import *
import scipy.io as sio
from scipy.signal import hilbert
from scipy.interpolate import griddata, interp1d
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

# Load File and Set Imaging Grid
dov = 36e-3; # Max Depth [m]
upsamp_x = 2; # Upsampling in x (assuming dx = pitch)
upsamp_z = 2; # Upsampling in z (assuming dz = pitch)
Nx0 = 192; # Number of Points Laterally in x
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

# Simulation Space
x = np.arange(-(upsamp_x*Nx0-1)/2,1+(upsamp_x*Nx0-1)/2)*(pitch/upsamp_x); # m
Nu1 = np.round(dov/((lmbda/2)/upsamp_z));
z = np.arange(Nu1)*(lmbda/2)/upsamp_z; # m

# Image Reconstruction Parameters and Anti-Aliasing Window
dBrange = np.array([-80, 0]); reg = 1e-3; ord = 50;
xmax = (np.max(np.abs(xpos))+np.max(np.abs(x)))/2; # m
aawin = 1/np.sqrt(1+(x/xmax)**ord);

# Transmit Impulse Response in Frequency Domain
nt = t.size; # [s]
fs = 1/np.mean(np.diff(t)); # [Hz]
f = (fs/2)*np.arange(-1,1,2/nt); # [Hz]
P_Tx = lambda f: 1.0*((f>=fTx-fBW/2) & (f<=fTx+fBW/2)); # Pulse Spectrum
P_Tx_f = P_Tx(f); # Pulse Definition

# Only Keep Positive Frequencies within Passband
passband_f_idx = np.argwhere((P_Tx_f > reg) & (f > 0)).flatten();
f = f[passband_f_idx]; P_Tx_f = P_Tx_f[passband_f_idx];
P_Tx_f = np.ones(P_Tx_f.shape); # Assume Flat Passband

# Get Receive Channel Data in the Frequency Domain
P_Rx_f = np.fft.fftshift(np.fft.fft(rxdata_h, n=nt, axis=0), axes=0);
P_Rx_f = P_Rx_f[passband_f_idx,:,:]; del rxdata_h;
X, F, N = np.meshgrid(np.arange(P_Rx_f.shape[1]), f, np.arange(P_Rx_f.shape[2]));
P_Rx_f = P_Rx_f * np.exp(-1j*2*np.pi*F*t[0]);
rxdata_f = interp1d(xpos, np.transpose(P_Rx_f, (1,0,2)), \
    kind='nearest', axis=0, fill_value=0, bounds_error=False)(x);

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

# Pulsed-Wave Frequency Response on Transmit
txdata_f = np.zeros((x.size, f.size, tx_evts.size), dtype=np.dtype('complex64'));
for tx_idx in np.arange(tx_evts.size):
    # Construct Transmit Responses for Each Element
    apod_x = interp1d(xpos, apod[tx_idx,:], \
        kind='nearest', axis=0, fill_value=0, bounds_error=False)(x);
    delayIdeal = interp1d(xpos, delays[tx_idx,:], \
        kind='nearest', axis=0, fill_value=0, bounds_error=False)(x);
    txdata_f[:,:,tx_idx] = np.outer(apod_x,P_Tx_f) * \
        np.exp(-1j*2*np.pi*np.outer(delayIdeal,f));

# Define Triangle-Inequality Coherence Factor (CF)
TICF = lambda signals: (np.abs(np.sum(signals,axis=1))/(np.sum(np.abs(signals),axis=1)));

# Create Image and Gain Compensation Maps
img = np.zeros((z.size, x.size), dtype=np.dtype('complex64'));
img_ticf = np.zeros((z.size, x.size));
img[0,:] = np.sum(np.sum(txdata_f*np.conj(rxdata_f),axis=1),axis=1);
img_ticf[0,:] = TICF(np.sum(txdata_f*np.conj(rxdata_f),axis=1));

# Propagate Ultrasound Signals in Depth
rxdata_f_nxt = np.zeros(rxdata_f.shape);
txdata_f_nxt = np.zeros(txdata_f.shape);
for z_idx in np.arange(z.size-1):
    # Propagate Signals in Depth
    rxdata_f_nxt, txdata_f_nxt = \
        propagate(x, z[z_idx], z[z_idx+1], cbfm, f, rxdata_f, txdata_f, aawin);
    # Compute Image at this Depth
    img[z_idx+1,:] = np.sum(np.sum(txdata_f_nxt*np.conj(rxdata_f_nxt),axis=1),axis=1);
    img_ticf[z_idx+1,:] = TICF(np.sum(txdata_f_nxt*np.conj(rxdata_f_nxt),axis=1));
    # Setup Next Depth Step
    rxdata_f = rxdata_f_nxt; txdata_f = txdata_f_nxt;
    print("z = "+str(z[z_idx])+" m / "+str(dov)+" m");

# Reconstruct and Plot Ultrasound Image
plt.figure(); imagesc(1000*x, 1000*z, 20*np.log10(np.abs(img)/np.max(np.abs(img))), dBrange);
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('Full Waveform Reconstruction'); plt.colorbar(); plt.show();

# Triangle Inequality Coherence Factor
plt.figure(); imagesc(1000*x, 1000*z, img_ticf, np.array([0,1]));
plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]');
plt.title('Triangle-Inequality Coherence Factor (TICF)'); plt.colorbar(); plt.show();
