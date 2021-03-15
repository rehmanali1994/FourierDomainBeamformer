clear
clc

% Load all Functions from Subdirectories
addpath(genpath(pwd));

% Load File and Set Imaging Grid
load('FocTxDataset.mat'); % Point Targets and Lesions Phantom
dov = 36e-3; % Max Depth [m]
upsamp_x = 2; % Upsampling in x (assuming dx = pitch)
upsamp_z = 2; % Upsampling in z (assuming dz = pitch)
Nx0 = 192; % Number of Points Laterally in x
xlims = (12.7e-3)*[-1, 1];
zlims = [4e-3, 36e-3];
cbfm = 1540; % sound speed [m/s]

% Select Subset of Transmit Elements
tx_evts = 1:1:128;
txBeamOrigins = txBeamOrigins(tx_evts,:);
apod = apod(tx_evts,:);
rxdata_h = rcvdata(:,:,tx_evts);
clearvars rcvdata;

% Aperture Definition
fTx = 6e6; % frequency [Hz]
fBW = 6e6; % bandwidth [Hz]
lambda = cbfm/fTx; % wavelength [m]
pitch = mean(diff(rxAptPos(:,1))); % element spacing [m]
no_elements = size(rxAptPos,1); % number of elements
xpos = (-(no_elements-1)/2:(no_elements-1)/2)*pitch; % element position [m]

% Simulation Space
x = (-(upsamp_x*Nx0-1)/2:(upsamp_x*Nx0-1)/2)*(pitch/upsamp_x); % m
Nu1 = round(dov/((lambda/2)/upsamp_z)); 
z = ((0:Nu1-1))*(lambda/2)/upsamp_z; % m

% Image Reconstruction Parameters and Anti-Aliasing Window
dBrange = [-80, 0]; reg = 1e-3; ord = 50; 
xmax = (max(abs(xpos))+max(abs(x)))/2; % m
aawin = 1./sqrt(1+(x/xmax).^ord);

% Transmit Impulse Response in Frequency Domain
nt = numel(t); % [s]
fs = 1/mean(diff(t)); % [Hz] 
f = (fs/2)*(-1:2/nt:1-2/nt); % [Hz]
P_Tx = @(f) 1.0*((f>=fTx-fBW/2) & (f<=fTx+fBW/2)); % Pulse Spectrum
P_Tx_f = P_Tx(f); % Pulse Definition

% Only Keep Positive Frequencies within Passband
passband_f_idx = find((P_Tx_f > reg) & (f > 0));
f = f(passband_f_idx); P_Tx_f = P_Tx_f(passband_f_idx);
P_Tx_f = ones(size(P_Tx_f)); % Assume Flat Passband

% Get Receive Channel Data in the Frequency Domain
P_Rx_f = fftshift(fft(rxdata_h, nt, 1), 1);
P_Rx_f = P_Rx_f(passband_f_idx,:,:); clearvars rxdata_h;
[~, F, ~] = meshgrid(1:size(P_Rx_f,2), f, 1:size(P_Rx_f,3)); 
P_Rx_f = P_Rx_f .* exp(-1i*2*pi*F*t(1));
rxdata_f = interp1(xpos, permute(P_Rx_f, [2,1,3]), x, 'nearest', 0);

% Construct Transmit Beamforming Delays
delay = zeros(size(txBeamOrigins,1),size(txAptPos,1));
for tx_idx = 1:numel(tx_evts)
    % transmit aperture locations
    txAptPosRelToCtr = txAptPos - ...
        ones(size(txAptPos,1),1) * txBeamOrigins(tx_idx,:);
    txFocRelToCtr = tx_focDepth * ...
        ones(size(txAptPos,1),1) * tx_dir/norm(tx_dir);
    txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;
    % positive value is time delay, negative is time advance
    delay(tx_idx,:) = (sqrt(sum(txFocRelToCtr.^2, 2)) - ...
        sqrt(sum(txFocRelToAptPos.^2, 2)))/c;
end

% Pulsed-Wave Frequency Response on Transmit
txdata_f = zeros(numel(x), numel(f), numel(tx_evts));
for tx_idx = 1:numel(tx_evts) 
    % Construct Transmit Responses for Each Element
    apod_x = interp1(xpos, apod(tx_idx,:), x, 'nearest', 0);
    delayIdeal = interp1(xpos, delay(tx_idx,:), x, 'nearest', 0);
    txdata_f(:,:,tx_idx) = (apod_x'*P_Tx_f).*exp(-1i*2*pi*delayIdeal'*f);
end

% Define Triangle-Inequality Coherence Factor (CF)
TICF = @(signals) (abs(sum(signals,3))./(sum(abs(signals),3)));

% Create Image and Gain Compensation Maps
img = zeros(numel(z), numel(x));
img_ticf = zeros(numel(z), numel(x));
img(1,:) = sum(sum(txdata_f .* conj(rxdata_f),2),3);
img_ticf(1,:) = TICF(sum(txdata_f .* conj(rxdata_f),2));

% Propagate Ultrasound Signals in Depth
rxdata_f_nxt = zeros(size(rxdata_f));
txdata_f_nxt = zeros(size(txdata_f));
for z_idx = 1:numel(z)-1
    % Propagate Signals in Depth
    [rxdata_f_nxt, txdata_f_nxt] = ...
        propagate(x, z(z_idx), z(z_idx+1), cbfm, f, rxdata_f, txdata_f, aawin);
    % Compute Image at this Depth
    img(z_idx+1,:) = sum(sum(txdata_f_nxt .* conj(rxdata_f_nxt),2),3);
    img_ticf(z_idx+1,:) = TICF(sum(txdata_f_nxt .* conj(rxdata_f_nxt),2));
    % Setup Next Depth Step
    rxdata_f = rxdata_f_nxt; txdata_f = txdata_f_nxt;
    disp(['z = ', num2str(z(z_idx)), ' m / ', num2str(dov), ' m']);
end

% Reconstruct and Plot Ultrasound Image
figure; imagesc(1000*x, 1000*z, ...
    db(abs(img)/max(abs(img(:)))), dBrange);
xlabel('Lateral [mm]'); ylabel('Axial [mm]'); 
title('Full Waveform Reconstruction'); 
zoom on; axis equal; axis xy; axis image; 
colormap gray; colorbar(); set(gca, 'YDir', 'reverse'); 
xlim(1000*xlims); ylim(1000*zlims);

% Triangle Inequality Coherence Factor
figure; imagesc(1000*x, 1000*z, img_ticf, [0,1]);
xlabel('Lateral [mm]'); ylabel('Axial [mm]');
title('Triangle-Inequality Coherence Factor (TICF)'); 
zoom on; axis equal; axis xy; axis image; 
colormap gray; colorbar(); set(gca, 'YDir', 'reverse'); 
xlim(1000*xlims); ylim(1000*zlims);