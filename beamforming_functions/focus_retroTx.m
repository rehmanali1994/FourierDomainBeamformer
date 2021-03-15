function [foc_data, includePoint] = focus_retroTx(t, signal, foc_pts, ...
    rxAptPos, tx_center, tx_dir, tx_focDepth, tx_radius, dc, speed_of_sound)
% 
% FOCUS_RETROTX - Focuses the RF data at desired locations
% 
% The function interpolates the RF signals collected using the full synthetic sequence
% to focus the data at desired locations
% 
% INPUTS:
% t                  - T x 1 time vector for samples of the input signal
% signal             - T x N matrix containing input RF data to be interpolated
% foc_pts            - P x 3 matrix with position of focal points [m]
% rxAptPos           - N x 3 matrix with positions of the Rx apertures (elements) [m]
% tx_center          - 1 x 3 vector with the position of the center of the Tx aperture [m]
% tx_dir             - 1 x 3 matrix with direction of transmit beam
% tx_focDepth        - Depth of transmit focus along transmit direction [m] 
% tx_radius          - Radius of transmit aperture around center in x-y plane [m]
% dc                 - time offsets [s] for Tx and Rx; scalars, N (M) x 1 vectors, or P x N (M) matrix
% speed_of_sound     - speed of sounds [m/s]; default 1540 m/s
% 
% OUTPUT:
% foc_data           - P x N vector with interpolated (RF) data points
% includePoint       - P vector of whether point will contribute coherently to transmit beam sum
% 
% NOTE: this f-ion uses cubic spline interpolation (slower than linear interp in focus_fs_fast)
% 

tx_dir = tx_dir/norm(tx_dir); % normalize direction vector
tx_focus = tx_center + tx_focDepth * tx_dir; % transmit focus

tFoc = norm(tx_center - tx_focus)/speed_of_sound; % time at Tx focus
focToReconDisplacement = foc_pts - repmat(tx_focus, size(foc_pts,1), 1);
tFocToRecon = sign(focToReconDisplacement*tx_dir') * (1/speed_of_sound) ...
    .* sqrt(sum(focToReconDisplacement.^2, 2)); % focus to recon point time
tRecon = tFoc+tFocToRecon; % time at recon point

tx_times = repmat(tRecon, 1, size(rxAptPos, 1));
rx_times = calc_times(foc_pts, rxAptPos, dc, speed_of_sound);
foc_times = rx_times + tx_times;

% for excluding reconstruction points outside validity region
x_intercept = repmat(tx_focus(1), size(foc_pts,1), 1) - ...
    repmat(tx_focus(3), size(foc_pts,1), 1) .* ...
    (focToReconDisplacement(:, 1) ./ focToReconDisplacement(:, 3));
excludePoint = ((((x_intercept - tx_center(1)).^2) > tx_radius^2) | ...
    (x_intercept < min(rxAptPos(:,1))) | ...
    (x_intercept > max(rxAptPos(:,1))));
includePoint = ~excludePoint;

if 1
    % reconstruction for points inside validity region
    elmts = 1:size(rxAptPos,1);
    [E, T] = meshgrid(elmts, t);
    foc_data = zeros(numel(includePoint),size(rxAptPos,1));
    foc_data(includePoint,:) = interp2(E, T, signal, ...
        ones(size(foc_pts(includePoint,:),1),1)*elmts, ...
        foc_times(includePoint,:), 'linear', 0);
    % reconstruction with edge-waves 
    xTxLeft = max(tx_center(1)-tx_radius, min(rxAptPos(:,1)));
    xTxRight = min(tx_center(1)+tx_radius, max(rxAptPos(:,1)));
    dc_txLeft = tFoc-sqrt(sum((tx_focus-[xTxLeft,0,0]).^2,2))/speed_of_sound;
    dc_txRight = tFoc-sqrt(sum((tx_focus-[xTxRight,0,0]).^2,2))/speed_of_sound;
    tTxLeft = calc_times(foc_pts, [xTxLeft,0,0], dc_txLeft, speed_of_sound);
    tTxRight = calc_times(foc_pts, [xTxRight,0,0], dc_txRight, speed_of_sound);
    tx_times_left = repmat(tTxLeft, 1, size(rxAptPos, 1));
    tx_times_right = repmat(tTxRight, 1, size(rxAptPos, 1));
    foc_times_left = rx_times + tx_times_left;
    foc_times_right = rx_times + tx_times_right;
    foc_data = foc_data + interp2(E, T, signal, ...
        ones(size(foc_pts,1),1)*elmts, foc_times_left, 'linear', 0);
    foc_data = foc_data + interp2(E, T, signal, ...
        ones(size(foc_pts,1),1)*elmts, foc_times_right, 'linear', 0);
else
    % reconstruction for points inside validity region
    elmts = 1:size(rxAptPos,1);
    [E, T] = meshgrid(elmts, t);
    foc_data = interp2(E, T, signal, ...
        ones(size(foc_pts,1),1)*elmts, foc_times, 'linear', 0);
end

end

