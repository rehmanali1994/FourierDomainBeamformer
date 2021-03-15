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
zlims = [0e-3, 36e-3];
cbfm = 1540; % sound speed [m/s]

% Select Which Transmit Elements
tx_evt = 64;
txBeamOrigin = txBeamOrigins(tx_evt,:);
apod = apod(tx_evt,:);
rxdata_h = rcvdata(:,:,tx_evt);
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
dBrange = [-80, 0]; reg = 1e-1; ord = 50; 
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
P_Tx_f = reshape(hanning(numel(P_Tx_f)),size(P_Tx_f)); 

% Get Receive Channel Data in the Frequency Domain
P_Rx_f = fftshift(fft(rxdata_h, nt, 1), 1);
P_Rx_f = P_Rx_f(passband_f_idx,:,:); clearvars rxdata_h;
[~, F] = meshgrid(1:size(P_Rx_f,2), f); 
P_Rx_f = P_Rx_f .* exp(-1i*2*pi*F*t(1));
rxdata_f = interp1(xpos, permute(P_Rx_f, [2,1]), x, 'nearest', 0);

% Construct Transmit Beamforming Delays
% Transmit Aperture Locations
txAptPosRelToCtr = txAptPos - ones(size(txAptPos,1),1) * txBeamOrigin;
txFocRelToCtr = tx_focDepth * ones(size(txAptPos,1),1) * tx_dir/norm(tx_dir);
txFocRelToAptPos = txFocRelToCtr - txAptPosRelToCtr;
% Positive Value is Time Delay, Negative is Time Advance
delay = (sqrt(sum(txFocRelToCtr.^2, 2))-sqrt(sum(txFocRelToAptPos.^2, 2)))/c;

% Pulsed-Wave Frequency Response on Transmit
apod_x = interp1(xpos, apod, x, 'nearest', 0);
delayIdeal = interp1(xpos, delay, x, 'nearest', 0);
txdata_f = (apod_x'*P_Tx_f).*exp(-1i*2*pi*delayIdeal'*f);

% Propagate Ultrasound Signals in Depth
rx_wf_x_z_f = zeros([numel(z),size(rxdata_f)]);
rx_wf_x_z_f(1,:,:) = rxdata_f;
tx_wf_x_z_f = zeros([numel(z),size(txdata_f)]);
tx_wf_x_z_f(1,:,:) = txdata_f;
rxdata_f_nxt = zeros(size(rxdata_f));
txdata_f_nxt = zeros(size(txdata_f));
for z_idx = 1:numel(z)-1
    % Propagate Signals in Depth
    [rxdata_f_nxt, txdata_f_nxt] = ...
        propagate(x, z(z_idx), z(z_idx+1), cbfm, f, rxdata_f, txdata_f, aawin);
    rx_wf_x_z_f(z_idx+1,:,:) = rxdata_f_nxt;
    tx_wf_x_z_f(z_idx+1,:,:) = txdata_f_nxt;
    % Setup Next Depth Step
    rxdata_f = rxdata_f_nxt; txdata_f = txdata_f_nxt;
    disp(['z = ', num2str(z(z_idx)), ' m / ', num2str(dov), ' m']);
end

% Compute Wavefield vs Time
tstart = 0; tend = 25e-6; Nt = 201;
t = linspace(tstart, tend, Nt);
[ff, tt] = meshgrid(f, t); 
delays = exp(1i*2*pi*ff.*tt);
Nx = numel(x); Nz = numel(z);
tx_wf_x_z_t = permute(reshape(delays*reshape(permute(tx_wf_x_z_f, ...
    [3, 1, 2]), [numel(f), Nz*Nx]), [Nt, Nz, Nx]), [2, 3, 1]);
rx_wf_x_z_t = permute(reshape(delays*reshape(permute(rx_wf_x_z_f, ...
    [3, 1, 2]), [numel(f), Nz*Nx]), [Nt, Nz, Nx]), [2, 3, 1]);
img_x_z_t = cumsum(rx_wf_x_z_t .* conj(tx_wf_x_z_t), 3);

%% Plot Cross-Correlation of Tx and Rx Wavefields
M = moviein(numel(t));
while true
    % Image Reconstructed at Each Time Step
    img = zeros(numel(z), numel(x));
    for t_idx = 1:numel(t)
        % Plot Transmit Wavefield
        subplot(1,3,1); imagesc(1000*x, 1000*z, real(tx_wf_x_z_t(:, :, t_idx)), ...
            reg*max(max(abs(tx_wf_x_z_t(:, :, t_idx))))*[-1,1]); 
        xlabel('x Azimuthal Distance (mm)'); ylabel('z Axial Distance (mm)'); 
        zoom on; axis equal; axis xy; axis image; set(gca, 'YDir', 'reverse'); 
        title('Transmit Wavefield');
        xlim(1000*xlims); ylim(1000*zlims); colormap gray; 
        % Plot Receive Wavefield
        subplot(1,3,2); imagesc(1000*x, 1000*z, real(rx_wf_x_z_t(:, :, t_idx)), ...
            reg*max(max(abs(rx_wf_x_z_t(:, :, t_idx))))*[-1,1]); 
        xlabel('x Azimuthal Distance (mm)'); ylabel('z Axial Distance (mm)'); 
        zoom on; axis equal; axis xy; axis image; set(gca, 'YDir', 'reverse'); 
        title('Backpropagated Received Signals');
        xlim(1000*xlims); ylim(1000*zlims); colormap gray; 
        % Accumulate Cross Corrleation
        img = img_x_z_t(:, :, t_idx);
        % Plot Accumulated Image
        subplot(1,3,3); imagesc(1000*x, 1000*z, ...
            db(abs(img)/max(abs(img(:)))), dBrange);
        xlabel('x Azimuthal Distance (mm)'); ylabel('z Axial Distance (mm)'); 
        title('Time Domain Cross-Correlation'); 
        zoom on; axis equal; axis xy; axis image; 
        colormap gray; xlim(1000*xlims); ylim(1000*zlims);  
        set(gca, 'YDir', 'reverse'); getframe(gca); 
        M(t_idx) = getframe(gcf); clf();
    end
end