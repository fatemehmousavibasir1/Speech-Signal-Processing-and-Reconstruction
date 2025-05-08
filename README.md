# Speech-Signal-Processing-and-Reconstruction

This project involves a complete pipeline for processing speech audio signals. The goal is to analyze the input speech, extract meaningful features, detect voiced frames, estimate pitch (fundamental frequency), and finally reconstruct the speech signal using Linear Predictive Coding (LPC) and excitation signals.

## The pipeline consists of the following stages:

**Silence Removal**
The first step removes silent parts of the signal using a simple energy-based threshold. Portions of the signal with amplitude or energy below a predefined threshold are considered silence and are removed to focus on the active speech segments.

**Framing and Windowing**
The remaining signal is divided into overlapping frames of fixed length. A Hamming window is applied to each frame to minimize discontinuities at the frame boundaries and reduce spectral leakage in subsequent analysis.

**LPC Coefficient Extraction**
For each windowed frame, LPC coefficients are extracted using the Least Squares method. These coefficients capture the spectral envelope of speech and are crucial for modeling the vocal tract filter.

**Gain Computation**
The gain (or energy) of each frame is computed. It plays an important role in signal reconstruction, representing the amplitude of the excitation signal passed through the LPC filter.

**Voiced/Unvoiced Detection**
Each frame is classified as voiced or unvoiced based on its energy. If the energy exceeds a certain threshold, the frame is considered voiced; otherwise, it is treated as unvoiced.

**Pitch Estimation (F0)**
The fundamental frequency (pitch) of each voiced frame is estimated using the autocorrelation method. Unvoiced frames are assigned an F0 of zero, while voiced frames receive a value inversely related to the time lag of the first minimum in the autocorrelation function.

**Speech Signal Reconstruction**
Using the extracted LPC coefficients, gain values, pitch, and voiced/unvoiced information, each frame is reconstructed. Voiced frames are excited by periodic pulses according to the estimated F0, while unvoiced frames are excited using white noise. These frame-level reconstructions are then overlapped and added to produce the final reconstructed speech signal.

**Dependencies:**
The project requires Python libraries such as NumPy, SciPy, Librosa, Matplotlib, and SoundFile. These libraries handle signal processing, visualization, and audio I/O.

## Execution:
The entire pipeline can be implemented either in a Jupyter notebook or as a Python script. Each step is modularized for clarity and reusability.

Results and Visualization:

Visualization of framed signals before and after windowing.

Plot of energy per frame with marked voiced frames.

Plot of estimated pitch values (F0) over time.

Playback and saving of the final reconstructed signal.
