# Signal-to-Noise Simulator

A visual educational tool for understanding signal-to-noise ratio in chromatographic peak detection.

## Overview

The Signal-to-Noise Simulator is an interactive application designed to help students and researchers understand signal-to-noise ratio (SNR) in peak detection. This tool visualizes how varying levels of noise affect peak detection and measurement, and demonstrates different signal processing techniques used in analytical chemistry.

## Features

- **Interactive Peak Visualization**: Adjust SNR values and see real-time changes in peak detectability
- **Customizable Parameters**: Control the number of peaks displayed (1-10)
- **Educational Annotations**: Display peak heights, SNR values, and noise regions
- **Intelligent Background Detection**: Automatic detection of noise-free regions for accurate measurements
- **Multiple Analysis Modes**: Compare different signal processing techniques on the same chromatogram
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
git clone https://github.com/maccoss/Signal-to-Noise.git
cd Signal-to-Noise
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

- Adjust the **SNR Value** slider (0.5-20) to see how different noise levels affect peak visibility
- Change the **Number of Peaks** to simulate simple or complex chromatograms
- Toggle **Show Peak Heights** to display measurements and SNR values for each peak
- Enable **Show Noise Region** to highlight the background areas used for noise calculation
- Click **Update Visualization** to generate a new random chromatogram with current settings

### Detection Methods Comparison Tab

This tab demonstrates different signal processing techniques applied to the same chromatogram:
- Raw noisy signal
- Savitzky-Golay filtered signal (polynomial smoothing)
- First derivative (useful for peak detection)
- Optional peak boundary visualization

### Educational Resources Tab

Contains detailed information about SNR in chromatography:
- Definition and calculation methods
- Importance in analytical method validation
- Factors affecting SNR in instrumental analysis
- Techniques for improving SNR
- Practical implications for data analysis

## Interpreting the Visualization

- **Green Line**: The true signal without noise
- **Blue Line**: The signal with added noise
- **Red Dots**: Peak positions
- **Yellow Regions**: Areas used for background noise calculation
- **Orange Dashed Lines**: ±1 standard deviation of the noise
- **Pink Shading**: Peak boundaries (when enabled)
- **Red Text**: Measured noise standard deviation

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

## License

This project is available for educational use.

## Acknowledgments

Developed for educational purposes in analytical chemistry and instrumental analysis courses.