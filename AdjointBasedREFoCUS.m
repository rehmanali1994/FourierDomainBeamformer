clear
clc

% Load all Functions from Subdirectories
addpath(genpath(pwd));

% Load File and Set Imaging Grid
load('FocTxDataset.mat'); % Point Targets and Lesions Phantom
dov = 36e-3; % Max Depth [m]
num_x = 201; num_z = 601;
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

% Transmit Impulse Response in Frequency Domain
nt = numel(t); % [s]
fs = 1/mean(diff(t)); % [Hz] 

% Construct Transmit Beamforming Delays
delays = zeros(size(txBeamOrigins,1),size(txAptPos,1));
for tx_idx = 1:numel(tx_evts)
    % transmit aperture locations
    txAptPosRelToCtr = txAptPos - ...
        ones(size(txAptPos,1),1) * txBeamOrigins(tx_idx,:);
    txFocRelToCtr = tx_focDepth * ...
        ones(size(txAptPos,1),1) * tx_dir/norm(tx_dir);
    txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;
    % positive value is time delay, negative is time advance
    delays(tx_idx,:) = (sqrt(sum(txFocRelToCtr.^2, 2)) - ...
        sqrt(sum(txFocRelToAptPos.^2, 2)))/c;
end

% Recovered Multistatic Dataset
rf_decoded = refocus_decode(rxdata_h,fs*delays,'apod',apod,'fHPF',(1e6)/fs);

% Points to Focus and Get Image At
x_img = linspace(xlims(1), xlims(2), num_x);
z_img = linspace(zlims(1), zlims(2), num_z);
dBrange = [-80, 0]; c = 1540;

% Multistatic Synthetic Aperture Image Reconstruction
[X, Y, Z] = meshgrid(x_img, 0, z_img);
foc_pts = [X(:), Y(:), Z(:)];
tic; focData = bfm_fs_fast(t, rf_decoded, foc_pts, rxAptPos, txAptPos, 0, 0, c); toc;
img_h = reshape(focData, [numel(x_img), numel(z_img)])';
imagesc(1000*x_img, 1000*z_img, 20*log10(abs(img_h)/max(abs(img_h(:)))), dBrange); 
title('Adjoint-Based REFoCUS Reconstruction')
axis image; xlabel('Lateral [mm]'); ylabel('Axial [mm]');
colormap(gray); colorbar();