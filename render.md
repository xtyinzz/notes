### Rendering

- [x] **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** [[Paper]](pdfs/nerf.pdf)


<br>

- [x] **Differentiable Direct Volume Rendering [TVCG, 2021]** S. Weiss and R. Westermann [[Paper]](pdfs/differentiable_dvr.pdf)
  - color reconstruction
    - issue with non-monotunic TF: density can get traped into local maximum
    - DiffDVR instead optimize for a RGBA field (like NeRF) insteald of intensity field

<br>

- [ ] **Real-Time Denoising of Volumetric Path Tracing for Direct Volume Rendering [TVCG, 2021]** J. A. Iglesias-Guitian, P. S. Mane and B. Moon [[Paper]](pdfs/Real-Time_Denoising_of_Volumetric_Path_Tracing_for_Direct_Volume_Rendering.pdf)
  - problems in real-time Volumetric Path Tracing for DVR:
    - flickers and noise
  - Proposed a denoising technique with image space linear predictor
    - advantage: great quality (not overly blurred like RNN, not affected much by out-of-sample data), no pre-training, and running real-time with commodity hardware

<br>

- [ ] **DNN-VolVis: Interactive Volume Visualization Supported by Deep Neural Network [Pvis, 2019]** [[Paper]](pdfs/DNN-VolVis_Interactive_Volume_Visualization_Supported_by_Deep_Neural_Network.pdf)


<br>

- [ ] **Explorable Volumetric Depth Images from Raycasting** S. Frey, F. Sadlo and T. Ertl [[Paper]](pdfs/Explorable_Volumetric_Depth_Images_from_Raycasting.pdf)

<br>
