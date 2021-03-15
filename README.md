# FourierDomainBeamformer
Generalized Fourier Beamformer Based on the Cross-Correlation of Transmitted and Received Wavefields

Fourier beamforming is generally used to form medical ultrasound images using specific transmission sequences. Fourier beamforming is distinct from delay-and-sum beamforming in that the diffraction process is used to focus ultrasound signals rather than selecting signal values based on times-of-flight. The Fourier beamformer shown here is applicable to any transmit sequence as long as the element-wise transmit apodizations and delays are known along with their corresponding receive channel data.

We provide sample data and algorithms presented in

> Ali, R. "Fourier-Based Synthetic-Aperture Imaging forArbitrary Transmissions by Cross-Correlation ofTransmitted and Received Wave-Fields". *Manuscript submitted for publication.*

for the reconstruction ultrasound images based on conventional dynamic-receive beamforming, virtual source synthetic aperture, REFoCUS (https://github.com/nbottenus/REFoCUS), and the proposed Fourier beamforming technique (called "full-waveform reconstruction" in the paper)

If you use the code/algorithm for research, please cite the above paper. 

You can reference a static version of this code by its DOI number:
INSERT DOI HERE

# Code and Sample Datasets
Each image reconstruction algorithm is implemented in both MATLAB and Python:
1) Conventional dynamic-receive beamforming ([FocTxRecon.m](FocTxRecon.m) and [FocTxRecon.py](FocTxRecon.py))
2) Virtual source synthetic aperture ([VirtualSourceRecon.m](VirtualSourceRecon.m) and [VirtualSourceRecon.py](VirtualSourceRecon.py))
3) REFoCUS ([AdjointBasedREFoCUS.m](AdjointBasedREFoCUS.m) and [AdjointBasedREFoCUS.py](AdjointBasedREFoCUS.py))
4) Full-Waveform Reconstruction in Time Domain ([TimeDomFullWaveRecon.m](TimeDomFullWaveRecon.m) and [TimeDomFullWaveRecon.py](TimeDomFullWaveRecon.py)) and Frequency Domain ([FreqDomFullWaveRecon.m](FreqDomFullWaveRecon.m) and [FreqDomFullWaveRecon.py](FreqDomFullWaveRecon.py))

The algorithm is broken into two steps. The first step is to create a stack of coherence factor images focused at various sound speeds (implemented in [MATLAB/CoherenceFactorImages.m](MATLAB/CoherenceFactorImages.m) and [Python/CoherenceFactorImages.py](Python/CoherenceFactorImages.py)). These coherence factor image stacks are saved in [SavedCoherenceFactorImages](SavedCoherenceFactorImages) as .mat files. The second step is to estimate the average and local sound speed as a function of depth from this stack of coherence images (implemented in [MATLAB/SoundSpeedEstimation.m](MATLAB/SoundSpeedEstimation.m) and [Python/CoherenceFactorImages.py](Python/SoundSpeedEstimation.py))

Please see any supporting [MultistaticDatasets](MultistaticDatasets)

**Please download the sample data (.mat files) under the [releases](https://github.com/rehmanali1994/FourierDomainBeamformer/releases) tab for this repository, and place that data in the main directory.**

# Sample results
The Fourier beamforming technique provided here is equivalent to a time-domain cross-correlation process shown here:

![](TimeDomainXCorr.gif)

Here are results for conventional dynamic-receive beamforming, virtual source synthetic aperture, REFoCUS, and the proposed Fourier beamforming technique ("full-waveform reconstruction" in the paper): 

![](Reconstructions.png)
