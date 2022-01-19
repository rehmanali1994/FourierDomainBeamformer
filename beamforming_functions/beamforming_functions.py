import numpy as np
from scipy import linalg
from scipy.interpolate import RectBivariateSpline, interpn
import pdb

def calc_times(foci, elempos, dc = 0, speed_of_sound = 1540):
    ''' foc_times = calc_times(foci, elempos, dc = 0, speed_of_sound = 1540)

    CALC_TIMES - computes focusing times

    The function computes the (Tx or Rx) time of arrival for specified focal points
    given the array element positions.

    NOTE: Primarily intended when Tx and Rx apertures are the same (i.e. no full synthetic aperture)

    INPUTS:
    foci              - M x 3 matrix with position of focal points of interest [m]
    elempos           - N x 3 matrix with element positions [m]
    dc                - time offset [s]; scalar, N x 1 vector, or M x N array
    speed_of_sound    - speed of sounds [m/s]; default 1540 m/s

    OUTPUT:
    foc_times         - M x N matrix with times of flight for all foci and all array elements '''

    if type(dc).__module__ == 'builtins':
        dc = np.array([dc]);
    if not(np.isscalar(dc)) and sum(np.array(dc.shape)==1) <= 1:
        np.tile(dc, (foci.shape[0], 1));

    foci_tmp = np.tile(np.reshape(foci,(foci.shape[0],1,3)), (1,elempos.shape[0],1));
    elempos_tmp = np.tile(np.reshape(elempos,(1,elempos.shape[0],3)), (foci_tmp.shape[0],1,1));

    r = foci_tmp - elempos_tmp;

    distance = np.sqrt(np.sum(r**2, axis = 2));
    foc_times = distance/speed_of_sound + dc;

    return foc_times;


def focus_fs_to_TxBeam(t, signal, rxAptPos, txAptPos, tx_center, tx_dir, tx_focDepth, tx_apod, dc_tx = 0, speed_of_sound = 1540):
    ''' foc_data = focus_fs_to_TxBeam(t, signal, rxAptPos, txAptPos, tx_center, tx_dir, tx_focDepth, dc_tx = 0, speed_of_sound = 1540)

    FOCUS_FS_TO_TXBEAM - Focuses the RF data at desired locations

    The function interpolates the RF signals collected using the full synthetic sequence
    to focus the data at desired locations

    INPUTS:
    t                  - T x 1 time vector for samples of the input signal
    signal             - T x N x M matrix containing input RF data to be interpolated
    rxAptPos           - N x 3 matrix with positions of the Rx apertures (elements) [m]
    txAptPos           - M x 3 matrix with positions of the Tx apertures (elements) [m]
    tx_center          - 1 x 3 vector with the position of the center of the Tx aperture [m]
    tx_dir             - 1 x 3 matrix with direction of transmit beam
    tx_focDepth        - Depth of transmit focus along transmit direction [m]
    tx_apod            - M x 1 vector of apodizations for transmit beam
    dc_tx              - time offsets [s] for Tx; scalars, M x 1 vectors
    speed_of_sound     - speed of sounds [m/s]; default 1540 m/s

    OUTPUT:
    foc_data           - T x N vector with transmit-beamformed data points '''

    if np.isscalar(dc_tx): dc_tx = np.array(dc_tx); # make dc_tx have array type

    # calculate all these relative distances to do retrospective transmit focusing
    txAptPosRelToCtr = txAptPos - np.tile(tx_center, (txAptPos.shape[0], 1));
    txFocRelToCtr = tx_focDepth * np.tile(tx_dir/np.linalg.norm(tx_dir), (txAptPos.shape[0], 1));
    txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;

    # positive value is time delay, negative is time advance
    if np.isinf(tx_focDepth): # Plane Wave Option
        tx_delay = (np.mat(-txAptPosRelToCtr)*np.mat(tx_dir/np.linalg.norm(tx_dir)).T)/speed_of_sound;
    else: # Column Vector
        tx_delay = (np.sqrt(np.sum(txFocRelToCtr**2, axis = 1))-np.sqrt(np.sum(txFocRelToAptPos**2, axis = 1)))/speed_of_sound;
    tx_delay = tx_delay + dc_tx;

    # transmit beamforming on full-synthetic aperture dataset: delayed-and-summed
    foc_data = np.zeros((t.shape[0], rxAptPos.shape[0])).astype('complex64');
    for i in np.arange(rxAptPos.shape[0]):
        for j in np.arange(txAptPos.shape[0]):
            foc_data[:,i] = foc_data[:,i] + tx_apod[j]*np.interp(t-tx_delay[j], t, signal[:,i,j], left=0, right=0);

    return foc_data;


def focus_data(t, signal, foci_rx, elempos, tx_center = np.zeros(3), dc = 0, speed_of_sound = 1540):
    ''' foc_data = focus_data(t, signal, foci_rx, elempos, tx_center = np.zeros(3), dc = 0, speed_of_sound = 1540)

    FOCUS_DATA - Focuses the RF data at desired locations

    The function interpolates the RF signals received on individual array elements
    to focus the data at desired locations.

    INPUTS:
    t                  - T x 1 time vector for samples of the input signal
    signal             - T x N matrix containing input RF data to be interpolated
    foci_rx            - M x 3 matrix with position of Rx focal points of interest [m]
    elempos            - N x 3 matrix with element positions [m]
    tx_center          - 1 x 3 vector with the position of the center of the Tx aperture
                       - (Tx center); [0,0,0] by default
    dc                 - time offset [s]; scalar, N x 1 vector, M x N matrix
    speed_of_sound     - speed of sounds [m/s]; default 1540 m/s

    OUTPUT:
    foc_data - M x N vector with interpolated (RF) data points

    NOTE: Intended to focus the data from diverging (transmit) waves or
    to focus the data along the transmit beam (a single A-line per Tx event) '''

    tx_center = np.ndarray.flatten(tx_center);

    # time from the Rx focus to array elements
    rx_times = calc_times(foci_rx, elempos, dc = dc, speed_of_sound = speed_of_sound);

    # time from the array elements to Rx focus
    tx_distance = foci_rx-np.tile(tx_center,(foci_rx.shape[0],1));
    tx_times = np.tile((np.sqrt(np.sum(tx_distance**2,axis=1))/speed_of_sound).T,(elempos.shape[0],1)).T;

    # two-way travel times
    foc_times = rx_times + tx_times;

    # interpolate channel data to get focused data
    elmts = np.arange(elempos.shape[0]);
    E_interp = np.tile(elmts,(foci_rx.shape[0],1));
    foc_data = interpn((elmts, t), signal.T, (E_interp, foc_times), bounds_error = False, fill_value = 0);

    return foc_data;


def focus_retroTx(t, signal, foc_pts, rxAptPos, tx_center, tx_dir, tx_focDepth, tx_radius, dc = 0, speed_of_sound = 1540):
    '''foc_data = focus_retroTx(t, signal, foc_pts, rxAptPos, tx_center, tx_dir, tx_focDepth, tx_radius, dc = 0, speed_of_sound = 1540)

    FOCUS_RETROTX - Focuses the RF data at desired locations

    The function interpolates the RF signals collected using the full synthetic sequence
    to focus the data at desired locations

    INPUTS:
    t                  - T x 1 time vector for samples of the input signal
    signal             - T x N matrix containing input RF data to be interpolated
    foc_pts            - P x 3 matrix with position of focal points [m]
    rxAptPos           - N x 3 matrix with positions of the Rx apertures (elements) [m]
    tx_center          - 1 x 3 vector with the position of the center of the Tx aperture [m]
    tx_dir             - 1 x 3 matrix with direction of transmit beam
    tx_focDepth        - Depth of transmit focus along transmit direction [m]
    tx_radius          - Radius of transmit aperture around center in x-y plane [m]
    dc                 - time offsets [s] for Tx and Rx; scalars, N x 1 vectors, or P x N matrix
    speed_of_sound     - speed of sounds [m/s]; default 1540 m/s

    OUTPUT:
    foc_data           - P x N vector with interpolated (RF) data points
    includePoint       - P vector of whether point will contribute coherently to transmit beam sum'''

    tx_dir = tx_dir/np.linalg.norm(tx_dir); # normalize direction vector
    tx_focus = tx_center + tx_focDepth * tx_dir; # transmit focus

    # Retrospective Transmit Focal Times
    tFoc = np.linalg.norm(tx_center - tx_focus)/speed_of_sound; # time at Tx focus
    focToReconDisplacement = foc_pts - np.tile(tx_focus, (foc_pts.shape[0], 1));
    tFocToRecon = np.sign(np.array(np.mat(focToReconDisplacement)*np.mat(tx_dir).T)).flatten() * \
        np.sqrt(np.sum(focToReconDisplacement**2, 1)) / speed_of_sound; # focus to recon point time
    tRecon = tFoc + tFocToRecon; # time at recon point

    # Calculation of Focal Times to Interpolate
    tx_times = np.tile(np.array([tRecon]).T, (1,rxAptPos.shape[0]))
    rx_times = calc_times(foc_pts, rxAptPos, dc = dc, speed_of_sound = speed_of_sound);
    foc_times = rx_times + tx_times;

    # For Excluding Reconstruction Points Outside Validity Region
    x_intercept = np.tile(tx_focus[0], (foc_pts.shape[0], 1)).flatten() - \
        np.tile(tx_focus[2], (foc_pts.shape[0], 1)).flatten() * \
        (focToReconDisplacement[:, 0] / focToReconDisplacement[:, 2]);
    excludePoint = ((((x_intercept - tx_center[0])**2) > tx_radius**2) | \
        (x_intercept < np.min(rxAptPos[:,0])) | \
        (x_intercept > np.max(rxAptPos[:,0])));
    includePoint = ~excludePoint;

    # Interpolation of Receive Data
    if True:
        # reconstruction for points inside validity region
        elmts = np.arange(rxAptPos.shape[0]);
        E_interp = np.tile(elmts,(foc_pts.shape[0],1));
        foc_data = np.zeros((includePoint.size,rxAptPos.shape[0]), dtype = np.dtype('complex64'));
        foc_data[includePoint,:] = interpn((elmts, t), signal.T, \
            (E_interp[includePoint,:], foc_times[includePoint,:]), bounds_error = False, fill_value = 0);
        # reconstruction with edge-waves
        xTxLeft = np.max([tx_center[0]-tx_radius, np.min(rxAptPos[:,0])]);
        xTxRight = np.min([tx_center[0]+tx_radius, np.max(rxAptPos[:,0])]);
        dc_txLeft = tFoc-np.sqrt(np.sum((tx_focus-np.array([xTxLeft,0,0]))**2))/speed_of_sound;
        dc_txRight = tFoc-np.sqrt(np.sum((tx_focus-np.array([xTxRight,0,0]))**2))/speed_of_sound;
        tTxLeft = calc_times(foc_pts, np.array([xTxLeft,0,0])[np.newaxis,:], dc_txLeft, speed_of_sound);
        tTxRight = calc_times(foc_pts, np.array([xTxRight,0,0])[np.newaxis,:], dc_txRight, speed_of_sound);
        tx_times_left = np.tile(tTxLeft,(1,rxAptPos.shape[0]));
        tx_times_right = np.tile(tTxRight,(1,rxAptPos.shape[0]));
        foc_times_left = rx_times + tx_times_left;
        foc_times_right = rx_times + tx_times_right
        foc_data = foc_data + interpn((elmts, t), signal.T, \
            (E_interp, foc_times_left), bounds_error = False, fill_value = 0);
        foc_data = foc_data + interpn((elmts, t), signal.T, \
            (E_interp, foc_times_right), bounds_error = False, fill_value = 0);
    else:
        # reconstruction for points inside validity region
        elmts = np.arange(rxAptPos.shape[0]);
        E_interp = np.tile(elmts,(foc_pts.shape[0],1));
        foc_data = interpn((elmts, t), signal.T, (E_interp, foc_times), bounds_error = False, fill_value = 0);

    return foc_data, includePoint;


def bfm_fs(t, signal, foc_pts, rxAptPos, rxApod = None, txAptPos = None, txApod = None, dc_rx = 0, dc_tx = 0, speed_of_sound = 1540):
    '''bfm_data = bfm_fs(t, signal, foc_pts, rxAptPos, txAptPos = None, dc_rx = 0, dc_tx = 0, speed_of_sound = 1540)

    BFM_FS - Beamforms (Delay and Sum) the RF data at desired locations

    The function interpolates the RF signals collected using the full synthetic sequence
    to focus the data at desired locations.

    INPUTS:
    t                  - T x 1 time vector for samples of the input signal
    signal             - T x N x M matrix containing input RF data to be interpolated
    foc_pts            - P x 3 matrix with position of focal points [m]
    rxAptPos           - N x 3 matrix with positions of the Rx apertures (elements) [m]
    rxApod             - P x N matrix of receive apodizations for each Rx element and focal point
                       - matrix of ones by default
    txAptPos           - M x 3 matrix with positions of the Tx apertures (elements) [m]
                       - txAptPos = rxAptPos by default
    txApod             - P x M matrix of transmit apodizations for each Tx element and focal point

    dc_rx, dc_tx       - time offsets [s] for Tx and Rx; scalars, N (M) x 1 vectors, or P x N (M) matrix
    speed_of_sound     - speed of sounds [m/s]; default 1540 m/s

    OUTPUT:
    foc_data - vector with dimension P for beamformed image '''

    # Set variables to defaults if not set
    if rxApod is None: rxApod = np.ones((foc_pts.shape[0], rxAptPos.shape[0]));
    if txAptPos is None: txAptPos = rxAptPos;
    if txApod is None: txApod = rxApod;

    # time from the focus to receive  apertures (array elements)
    rx_times = calc_times(foc_pts, rxAptPos, dc = dc_rx, speed_of_sound = speed_of_sound);

    # time from the transmit apertures (array elements) to focus
    tx_times = calc_times(foc_pts, txAptPos, dc = dc_tx, speed_of_sound = speed_of_sound);

    # focused but not summed rf data
    bfm_data = np.zeros(foc_pts.shape[0]).astype('complex64');
    for i in np.arange(rx_times.shape[1]):
        for j in np.arange(tx_times.shape[1]):
            bfm_data = bfm_data + rxApod[:,i] * txApod[:,j] * \
                np.interp(rx_times[:,i]+tx_times[:,j], t, signal[:,i,j], left=0, right=0);
        print('Rx Elem '+str(i))

    return bfm_data;


def refocus_decode(rf_encoded, delays, apod=None, fHPF=None):
    ''' REFOCUS_DECODE Decode focused beams using the applied delays
    rf_decoded = REFOCUS_DECODE(rf_encoded,s_shift)
    Parameters:
        rf_encoded - RF data - samples x receive channel x transmit event
        delays - Applied delays in samples - transmit event x transmit element
    Name/value pairs:
        'apod' - Apodization applied for each transmit (same size as delays)
        'fHPF' - Ratio of High-Pass Filter Cutoff Frequency/Sampling-Frequency '''

    # Get input dimensions
    n_samples, n_receives, n_transmits = rf_encoded.shape;
    n_elements = delays.shape[1];
    assert(delays.shape[0] == n_transmits), 'Transmit count inconsistent between rf_encoded and delays';

    # Default apodization is all ones
    if apod is None:
        apod = np.ones(delays.shape);
    else:
        assert(apod.shape==delays.shape), 'Apodization size should match delays size';

    # High-Pass Filter Cutoff
    if fHPF is None:
        fHPF = 0;
    else:
        assert(fHPF>=0), 'High-pass filter cutoff should be greater than 0';

    # 1-D FFT to convert time to frequency
    RF_encoded = np.fft.fft(rf_encoded, axis = 0);
    RF_encoded = np.transpose(RF_encoded, axes=(2,1,0)); # (transmit event x receive channel x time sample)
    frequency = np.arange(np.ceil(n_samples/2))/n_samples;

    # Apply encoding matrix at each frequency
    RF_decoded = np.zeros((int(np.ceil(n_samples/2)), n_elements, n_receives), dtype = np.dtype('complex64'));
    for i in np.arange(1,int(np.ceil(n_samples/2))):
        H = (frequency[i]>fHPF)*apod*np.exp(-1j*2*np.pi*frequency[i]*delays);
        RF_decoded[i,:,:] = np.array(np.dot(np.conj(H).T, np.matrix(RF_encoded[:,:,i])));
    RF_decoded = np.transpose(RF_decoded, axes=(0,2,1)); # (frequency x receive channel x transmit element)

    # Inverse FFT for real signal
    rf_decoded = np.fft.ifft(RF_decoded, n=n_samples, axis = 0);
    return rf_decoded;


def propagate(x, z1, z2, c, f, rxdata_z1_f, txdata_z1_f, aawin):
    '''rxdata_z2_f, txdata_z2_f = propagate(x, z1, z2, c, f, rxdata_z1_f, txdata_z1_f, aawin)

    PROPAGATE - Angular Spectrum Propagation of TX/RX Signals into the Medium

    This function propagates transmit and receive wavefields at from one
    depth to another depth using the angular spectrum method

    INPUTS:
    x                  - 1 x X vector of x-grid positions for wavefield
    z1                 - depth of input TX and RX wavefields
    z2                 - depth of output TX and RX wavefields
    c                  - speed of sound [m/s] between z1 and z2; default 1540 m/s
    f                  - 1 x F vector of pulse frequencies in spectrum
    rxdata_z1_f        - X x F x N array of input RX wavefields at z1
    txdata_z1_f        - X x F x N array of input TX wavefields at z1
    aawin              - 1 x X vector of lateral taper to prevent wraparound

    OUTPUT:
    rxdata_z2_f        - X x F x N array of output RX wavefields at z2
    txdata_z2_f        - X x F x N array of output TX wavefields at z2'''

    # Verify the Number of Common Shot Gathers
    ns = txdata_z1_f.shape[2]; assert(rxdata_z1_f.shape[2] == ns), \
        'Number of sources must equal to number of common-source gathers';
    AAwin = np.tile(aawin[:,np.newaxis,np.newaxis], (1, f.size, ns));

    # Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
    ft = lambda sig: np.fft.fftshift(np.fft.fft(AAwin*sig, axis=0), axes=0);
    ift = lambda sig: AAwin*np.fft.ifft(np.fft.ifftshift(sig, axes=0), axis=0);

    # Spatial Grid
    dx = np.mean(np.diff(x)); nx = x.size;

    # FFT Axis for Lateral Spatial Frequency
    kx = np.mod(np.fft.fftshift(np.arange(nx)/(dx*nx))+1/(2*dx), 1/dx)-1/(2*dx);

    # Continuous Wave Response By Downward Angular Spectrum
    F, Kx = np.meshgrid(f,kx); # Create Grid in f-kx
    Kz = np.sqrt(((F/c)**2-Kx**2).astype('complex64')); # Axial Spatial Frequency
    H = np.exp(1j*2*np.pi*Kz*(z2-z1)); # Propagation Filter
    H[Kz**2 <= 0] = 0; # Remove Evanescent Components
    H = np.tile(H[:,:,np.newaxis], (1,1,ns)); # Replicate Across Shots
    # Apply Propagation Filter
    rxdata_z2_f = ift(H*ft(rxdata_z1_f));
    txdata_z2_f = ift(np.conj(H)*ft(txdata_z1_f));

    return rxdata_z2_f, txdata_z2_f;


# Python-Equivalent Command for IMAGESC in MATLAB
import matplotlib.pyplot as plt
def imagesc(x, y, img, rng, cmap='gray', numticks=(3, 3), aspect='equal'):
    exts = (np.min(x)-np.mean(np.diff(x)), np.max(x)+np.mean(np.diff(x)), \
        np.min(y)-np.mean(np.diff(y)), np.max(y)+np.mean(np.diff(y)));
    plt.imshow(np.flipud(img), cmap=cmap, extent=exts, vmin=rng[0], vmax=rng[1], aspect=aspect);
    plt.xticks(np.linspace(np.min(x), np.max(x), numticks[0]));
    plt.yticks(np.linspace(np.min(y), np.max(y), numticks[1]));
    plt.gca().invert_yaxis();
