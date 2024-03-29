# FourierDomainBeamformer
Generalized Fourier Beamformer Based on the Cross-Correlation of Transmitted and Received Wavefields

Fourier beamforming is generally used to form medical ultrasound images using specific transmission sequences. Fourier beamforming is distinct from delay-and-sum beamforming in that the diffraction process is used to focus ultrasound signals rather than selecting signal values based on times-of-flight. The Fourier beamformer shown here is applicable to any transmit sequence as long as the element-wise transmit apodizations and delays are known along with their corresponding receive channel data.

We provide sample data and algorithms presented in

> R. Ali, “Fourier-based Synthetic-aperture Imaging for Arbitrary Transmissions by Cross-correlation of Transmitted and Received Wave-fields,” Ultrasonic Imaging, p. 016173462110263, Jul. 2021, doi: 10.1177/01617346211026350. [Online]. Available: http://dx.doi.org/10.1177/01617346211026350 *

for the reconstruction ultrasound images based on conventional dynamic-receive beamforming, virtual source synthetic aperture, REFoCUS (https://github.com/nbottenus/REFoCUS), and the proposed Fourier beamforming technique (called "full-waveform reconstruction" here, but "shot-profile migration" in the paper).

If you use the code/algorithm for research, please cite the above paper. 

You can reference a static version of this code by its DOI number:
[![DOI](https://zenodo.org/badge/346254482.svg)](https://zenodo.org/badge/latestdoi/346254482)

# Code and Sample Datasets
Each image reconstruction algorithm is implemented in both MATLAB and Python:
1) Conventional dynamic-receive beamforming ([FocTxRecon.m](FocTxRecon.m) and [FocTxRecon.py](FocTxRecon.py))
2) Virtual source synthetic aperture ([VirtualSourceRecon.m](VirtualSourceRecon.m) and [VirtualSourceRecon.py](VirtualSourceRecon.py))
3) REFoCUS ([AdjointBasedREFoCUS.m](AdjointBasedREFoCUS.m) and [AdjointBasedREFoCUS.py](AdjointBasedREFoCUS.py))
4) Full-Waveform Reconstruction in Time Domain ([TimeDomFullWaveRecon.m](TimeDomFullWaveRecon.m) and [TimeDomFullWaveRecon.py](TimeDomFullWaveRecon.py)) and Frequency Domain ([FreqDomFullWaveRecon.m](FreqDomFullWaveRecon.m) and [FreqDomFullWaveRecon.py](FreqDomFullWaveRecon.py))

Please see any supporting functions in [beamforming_functions](beamforming_functions).

**Please download the sample data (FocTxDataset.mat) under the [releases](https://github.com/rehmanali1994/FourierDomainBeamformer/releases) tab for this repository, and place that data in the main directory ([FourierDomainBeamformer](https://github.com/rehmanali1994/FourierDomainBeamformer)).**

# Sample Results
The Fourier beamforming technique provided is equivalent to the time-domain cross-correlation process shown here:

![](TimeDomainXCorr.gif)

Here are results for conventional dynamic-receive beamforming, virtual source synthetic aperture, REFoCUS, and the proposed Fourier beamforming technique ("full-waveform reconstruction" here but "shot-profile migration" in the paper): 

![](Reconstructions.png)
