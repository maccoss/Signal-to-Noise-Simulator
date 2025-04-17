# Signal-to-Noise Simulator

A visual educational tool for understanding signal-to-noise ratio in chromatographic peak detection.

## Overview

The Signal-to-Noise Simulator is an interactive application designed to help students and researchers understand signal-to-noise ratio (SNR) in peak detection. This tool visualizes how varying levels of noise affect peak detection and measurement, and demonstrates different signal processing techniques used in analytical chemistry.

## Features

- **Interactive Peak Visualization**: Adjust SNR values and see real-time changes in peak detectability
- **Customizable Parameters**: 
  - Control the number of peaks displayed (1-10)
  - Adjust sampling rate with "Points Across Peak" setting
  - Toggle between display modes (Normal/High-Res with Noise)
- **Educational Annotations**: Display peak heights, SNR values, and noise regions
- **Intelligent Background Detection**: Automatic detection of noise-free regions for accurate measurements
- **Multiple Analysis Modes**: Compare different signal processing techniques on the same chromatogram
- **Signal Processing Visualization**: Toggle first and second derivative displays
- **Educational Resources**: Comprehensive information about SNR in analytical chemistry

## Installation

### System Requirements

- Python 3.6 or higher
- Tkinter support for your Python installation

### Dependencies

The simulator requires the following Python libraries:
- numpy
- matplotlib
- scipy
- tkinter (usually included with Python)
- PIL with ImageTk support (for plotting)

### Installation Steps

1. Clone the repository or download the source code
```bash
git clone https://github.com/maccoss/Signal-to-Noise-Simulator.git
cd Signal-to-Noise-Simulator
```

2. Install the required dependencies:

**On Debian/Ubuntu Linux:**
```bash
sudo apt-get install python3-tk python3-pil.imagetk
pip install numpy matplotlib scipy
```

**On macOS:**
```bash
pip install numpy matplotlib scipy pillow
```

**On Windows:**
```bash
pip install numpy matplotlib scipy pillow
```

## Usage

Run the simulator with:

```bash
python3 Chromatography.py
```

### SNR Impact Visualization Tab

- Enter an **SNR Value** to see how different noise levels affect peak visibility
- Set the **Number of Peaks** to simulate simple or complex chromatograms
- Adjust **Points Across Peak** to simulate different sampling rates
- Select a **Display Mode** to visualize high-resolution signal with noise
- Toggle **Show Peak Heights** to display measurements and SNR values for each peak
- Enable **Show Noise Region** to highlight the background areas used for noise calculation
- Click **Update Visualization** to generate a new random chromatogram with current settings

### Detection Methods Comparison Tab

This tab demonstrates different signal processing techniques applied to the same chromatogram:
- Raw noisy signal
- Savitzky-Golay filtered signal (polynomial smoothing)
- Toggle **Show 1st Derivative** for peak detection visualization
- Toggle **Show 2nd Derivative** for peak inflection point detection
- Optional **Show Peak Boundaries** for visualizing peak integration regions

### Educational Resources Tab

Contains detailed information about SNR in chromatography:
- Definition and calculation methods
- Importance in analytical method validation
- Factors affecting SNR in instrumental analysis
- Techniques for improving SNR
- Practical implications for data analysis

## Interpreting the Visualization

- **Green Line**: The ideal signal without noise
- **Blue Line**: The signal with noise at current sampling rate
- **Red Line** (optional): The sampled signal points when viewing high-res mode
- **Red Dots**: Peak positions
- **Yellow Regions**: Areas used for background noise calculation
- **Orange Dashed Lines**: ±1 standard deviation of the noise
- **Black Dashed Lines**: Peak boundaries (when enabled)
- **Red Text**: Measured noise standard deviation
- **Blue Text**: Points sampled across the peak width

## Troubleshooting

If you encounter the error `ModuleNotFoundError: No module named 'tkinter'`:
- Install the tkinter package for your system as described in the installation steps

If you encounter the error `ImportError: cannot import name 'ImageTk' from 'PIL'`:
- Install the Python Imaging Library Tkinter integration with `sudo apt-get install python3-pil.imagetk` on Debian/Ubuntu
- For other operating systems, try `pip install pillow`

## Educational Use

This simulator is ideal for:
- Teaching instrumental analysis concepts
- Laboratory method development training
- Signal processing demonstrations
- Method validation education
- Understanding the importance of proper sampling rates in chromatography

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Developed for educational purposes and for evaluation of different peak detection algorithms.