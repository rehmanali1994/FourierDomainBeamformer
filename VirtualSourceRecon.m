clear
clc

% Setup path
addpath(genpath(pwd));

% Load File and Set Imaging Grid
load('FocTxDataset.mat'); % Point Targets and Lesions Phantom
c = 1540; % Beamforming Sound Speed [m/s]
rlims = [4e-3, 36e-3]; % Imaging Depths

% Options for Virtual Source Reconstruction
opt = 1; % 1/2 are with/without edge waves, respectively
tx_idx = 64; % which transmit beam to show partial image of
        
% Get All Dimension Information
[nt, nRx, nTx] = size(rcvdata);
no_active_elements = max(sum(apod~=0));
pitch = mean(diff(txAptPos(:,1)));

% Subsample Transmit Events
dwnsmp = 1; % Downsampling Factor
rcvdata = rcvdata(:,:,1:dwnsmp:end);
txBeamOrigins = txBeamOrigins(1:dwnsmp:end,:);

% Points to Focus and Get Image At
dBrange = [-80, 0]; num_r = 600;
r_img = linspace(rlims(1), rlims(2), num_r);
tx_origin_x = txBeamOrigins(:,1); % Transmit Beam Origin [m]

% Scan Lines and Image Coordinates
[X, R_IMG] = meshgrid(txAptPos(:,1), r_img);
Y = zeros(size(X)); Z = R_IMG;
foc_pts = [X(:), Y(:), Z(:)];

% Virtual Source Synthetic Aperture Focusing
img_retroTx = zeros(numel(r_img), numel(txAptPos(:,1)), numel(tx_origin_x));
includePtInRecon = zeros(numel(r_img), numel(txAptPos(:,1)), numel(tx_origin_x));
disp('Beginning Virtual Source Synthetic Aperture Beamforming');
for kk = 1:numel(tx_origin_x)
    [focData, includePoint] = ...
        focus_retroTx(t, rcvdata(:,:,kk), foc_pts, rxAptPos, ...
        txBeamOrigins(kk,:), tx_dir, tx_focDepth, ...
        no_active_elements*pitch/2, 0, c);
    img_retroTx(:,:,kk) = sum(reshape(focData, ...
        [numel(r_img), numel(txAptPos(:,1)), size(rxAptPos,1)]), 3);
    includePtInRecon(:,:,kk) = reshape(includePoint, ...
        [numel(r_img), numel(txAptPos(:,1))]) | ...
        (abs(X-tx_origin_x(kk)) < (1-(1/(2*dwnsmp)))*mean(diff(tx_origin_x)));
    disp(['Transmit Beam at x = ', num2str(tx_origin_x(kk))]);
end 
numTxForReconPt = sum(includePtInRecon, 3);

% High-Pass Filter Each Image Axially
img_retroTx_orig = img_retroTx;
[b, a] = butter(10, 0.1, 'high');
img_retroTx = filtfilt(b, a, img_retroTx_orig);

%% Virtual Source Synthetic Aperture Reconstruction
switch opt
    case 1 % Virtual Source Reconstruction with Edge Waves
        img_h = sum(img_retroTx, 3);
        % L1 Coherence Factor
        l1cf = abs(sum(img_retroTx,3))./sum(abs(img_retroTx),3); 
    case 2 % Virtual Source Reconstruction without Edge Waves
        img_h = sum(img_retroTx.*includePtInRecon, 3) ./ (numTxForReconPt);
        % L1 Coherence Factor
        l1cf = abs(sum(img_retroTx.*includePtInRecon,3))./...
            sum(abs(img_retroTx.*includePtInRecon),3); 
end

% Need to Perform Scan Conversion: 
x_img = linspace(min(X(:)), max(X(:)), 300);
z_img = linspace(min(Z(:)), max(Z(:)), 1000);
[X_IMG, Z_IMG] = meshgrid(x_img, z_img);
[X, R_IMG] = meshgrid(txAptPos(:,1), r_img);
IMG = interp2(X, R_IMG, abs(img_h), X_IMG, Z_IMG);
L1CF = interp2(X, R_IMG, l1cf, X_IMG, Z_IMG);

% Show Scan Converted Image
figure; imagesc(1000*x_img, 1000*z_img, 20*log10(IMG/max(IMG(:))), dBrange); 
axis image; xlabel('Lateral [mm]'); ylabel('Axial [mm]');
title('Virtual Source Reconstruction'); colormap(gray); colorbar();

% Show CF Image
figure; imagesc(1000*txAptPos(:,1), 1000*r_img, L1CF, [0,1]);
xlabel('Lateral [mm]'); ylabel('Axial [mm]');
title('L1 Coherence Factor'); zoom on; axis equal; axis xy; axis image; 
colormap gray; colorbar(); set(gca, 'YDir', 'reverse'); 

%% Show Rx Focused Image for Individual Tx Beams

% Isolating Rx Focused Image for Single Tx Beam
img_h_singleTx = img_retroTx(:,:,round(tx_idx/dwnsmp));
img_h_singleTx_NoEdgeWaves = img_h_singleTx .* ...
    includePtInRecon(:,:,round(tx_idx/dwnsmp));

% Need to Perform Scan Conversion: 
x_img = linspace(min(X(:)), max(X(:)), 300);
z_img = linspace(min(Z(:)), max(Z(:)), 1000);
[X_IMG, Z_IMG] = meshgrid(x_img, z_img);
[X, R_IMG] = meshgrid(txAptPos(:,1), r_img);
img_h_singleTx_scan_conv = ...
    interp2(X, R_IMG, abs(img_h_singleTx), X_IMG, Z_IMG);
img_h_singleTx_NoEdgeWaves_scan_conv = ...
    interp2(X, R_IMG, abs(img_h_singleTx_NoEdgeWaves), X_IMG, Z_IMG);

% Show Scan Converted Image
figure; subplot(1,2,1); imagesc(1000*x_img, 1000*z_img, ...
    20*log10(img_h_singleTx_scan_conv./...
    max(img_h_singleTx_scan_conv(:))), dBrange); 
axis image; xlabel('Lateral [mm]'); ylabel('Axial [mm]');
title('Single TX with Edge Waves'); colormap(gray); colorbar();
subplot(1,2,2); imagesc(1000*x_img, 1000*z_img, ...
    20*log10(img_h_singleTx_NoEdgeWaves_scan_conv./...
    max(img_h_singleTx_NoEdgeWaves_scan_conv(:))), dBrange); 
axis image; xlabel('Lateral [mm]'); ylabel('Axial [mm]');
title('Single TX without Edge Waves'); colormap(gray); colorbar();