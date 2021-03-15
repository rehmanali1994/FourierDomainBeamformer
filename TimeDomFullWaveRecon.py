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
zlims = np.array([0e-3, 36e-3]);
cbfm = 1540; # sound speed [m/s]

# Select Which Transmit Elements
tx_evt = 64;
txBeamOrigin = txBeamOrigins[tx_evt,:];
apod = apod[tx_evt,:];
rxdata_h = rcvdata[:,:,tx_evt];
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
dBrange = np.array([-80, 0]); reg = 5e-2; ord = 50;
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
P_Tx_f = np.hanning(P_Tx_f.size);

# Get Receive Channel Data in the Frequency Domain
P_Rx_f = np.fft.fftshift(np.fft.fft(rxdata_h, n=nt, axis=0), axes=0);
P_Rx_f = P_Rx_f[passband_f_idx,:,]; del rxdata_h;
X, F = np.meshgrid(np.arange(P_Rx_f.shape[1]), f);
P_Rx_f = P_Rx_f * np.exp(-1j*2*np.pi*F*t[0]);
rxdata_f = interp1d(xpos, np.transpose(P_Rx_f, (1,0)), \
    kind='nearest', axis=0, fill_value=0, bounds_error=False)(x);

# Construct Transmit Beamforming Delays
# Transmit Aperture Locations
txAptPosRelToCtr = txAptPos - \
    np.ones((txAptPos.shape[0],1)) * txBeamOrigin;
txFocRelToCtr = tx_focDepth * \
    np.ones((txAptPos.shape[0],1)) * tx_dir/np.linalg.norm(tx_dir);
txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;
# Positive Value is Time Delay, Negative is Time Advance
delay = (np.sqrt(np.sum(txFocRelToCtr**2, axis=1)) - \
    np.sqrt(np.sum(txFocRelToAptPos**2, axis=1)))/c;

# Pulsed-Wave Frequency Response on Transmit
apod_x = interp1d(xpos, apod, kind='nearest', axis=0, fill_value=0, bounds_error=False)(x);
delayIdeal = interp1d(xpos, delay, kind='nearest', axis=0, fill_value=0, bounds_error=False)(x);
txdata_f = np.outer(apod_x,P_Tx_f) * np.exp(-1j*2*np.pi*np.outer(delayIdeal,f));

# Propagate Ultrasound Signals in Depth
rx_wf_x_z_f = np.zeros((z.size,)+rxdata_f.shape, dtype=np.dtype('complex64'));
rx_wf_x_z_f[0,:,:] = rxdata_f;
tx_wf_x_z_f = np.zeros((z.size,)+txdata_f.shape, dtype=np.dtype('complex64'));
tx_wf_x_z_f[0,:,:] = txdata_f;
rxdata_f_nxt = np.zeros(rxdata_f.shape, dtype=np.dtype('complex64'));
txdata_f_nxt = np.zeros(txdata_f.shape, dtype=np.dtype('complex64'));
for z_idx in np.arange(z.size-1):
    # Propagate Signals in Depth
    rxdata_f_nxt, txdata_f_nxt = propagate(x, z[z_idx], z[z_idx+1], cbfm, f, \
        rxdata_f[:,:,np.newaxis], txdata_f[:,:,np.newaxis], aawin);
    rx_wf_x_z_f[z_idx+1,:,:] = rxdata_f_nxt[:,:,0];
    tx_wf_x_z_f[z_idx+1,:,:] = txdata_f_nxt[:,:,0];
    # Setup Next Depth Step
    rxdata_f = rxdata_f_nxt[:,:,0]; txdata_f = txdata_f_nxt[:,:,0];
    print("z = "+str(z[z_idx])+" m / "+str(dov)+" m");

# Compute Wavefield vs Time
tstart = 0; tend = 25e-6; Nt = 201;
t = np.linspace(tstart, tend, Nt);
ff, tt = np.meshgrid(f, t);
delays = np.exp(1j*2*np.pi*ff*tt);
Nx = x.size; Nz = z.size;
tx_wf_x_z_t = np.transpose(np.reshape(delays.dot(np.reshape(np.transpose(tx_wf_x_z_f, \
    (2,0,1)), (f.size, Nz*Nx))), (Nt, Nz, Nx)), (1,2,0));
rx_wf_x_z_t = np.transpose(np.reshape(delays.dot(np.reshape(np.transpose(rx_wf_x_z_f, \
    (2,0,1)), (f.size, Nz*Nx))), (Nt, Nz, Nx)), (1,2,0));
img_x_z_t = np.cumsum(rx_wf_x_z_t*np.conj(tx_wf_x_z_t), axis=2);

## Plot Cross-Correlation of Tx and Rx Wavefields
plt.figure(); tpause = 1e-9;
while True:
    # Image Reconstructed at Each Time Step
    img = np.zeros((z.size, x.size),dtype=np.dtype('complex64'));
    for t_idx in np.arange(t.size):
        # Plot Transmit Wavefield
        plt.subplot(1,3,1); imagesc(1000*x, 1000*z, \
            np.real(tx_wf_x_z_t[:,:,t_idx]), \
            reg*np.max(np.abs(tx_wf_x_z_t[:,:,t_idx]))*np.array([-1,1]));
        plt.xlabel('x Azimuthal Distance (mm)');
        plt.ylabel('z Axial Distance (mm)');
        plt.title('Transmit Wavefield');
        # Plot Receive Wavefield
        plt.subplot(1,3,2); imagesc(1000*x, 1000*z, \
            np.real(rx_wf_x_z_t[:,:,t_idx]), \
            reg*np.max(np.abs(rx_wf_x_z_t[:,:,t_idx]))*np.array([-1,1]));
        plt.xlabel('x Azimuthal Distance (mm)');
        plt.ylabel('z Axial Distance (mm)');
        plt.title('Backpropagated Received Signals');
        # Accumulate Cross Corrleation
        img = img_x_z_t[:,:,t_idx];
        # Plot Accumulated Image
        plt.subplot(1,3,3); imagesc(1000*x, 1000*z, \
            20*np.log10(np.abs(img)/np.max(np.abs(img))), dBrange);
        plt.xlabel('x Azimuthal Distance (mm)');
        plt.ylabel('z Axial Distance (mm)');
        plt.title('Time Domain Cross-Correlation');
        # Set Spacing between Subplots
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        # Animate
        plt.draw(); plt.pause(tpause); plt.clf();
