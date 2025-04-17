import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
from scipy.signal import savgol_filter

def signal_to_noise_simulator():
    """Implementation of a signal-to-noise simulator for educational purposes"""
    sim_window = tk.Tk()
    sim_window.title("Signal-to-Noise Simulator")
    sim_window.geometry("1000x800")
    
    # Create notebook with tabs
    notebook = ttk.Notebook(sim_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Tab 1: SNR Impact Visualization
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="SNR Impact Visualization")
    
    # Create matplotlib figure for SNR impact
    fig1 = Figure(figsize=(9, 5), dpi=100)
    ax1 = fig1.add_subplot(111)
    canvas1 = FigureCanvasTkAgg(fig1, tab1)
    canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Create the controls frames first
    controls_frame1 = ttk.Frame(tab1)
    controls_frame1.pack(fill=tk.X, pady=5)
    
    controls_frame1_row2 = ttk.Frame(tab1)
    controls_frame1_row2.pack(fill=tk.X, pady=5)
    
    # Global variables to share data between tabs
    shared_data = {
        'x': None,
        'y_true': None,
        'y_noisy': None,
        'peaks': None,
        'noise_std': None,
        'snr': None,
        'noise_regions': []
    }
    
    # Status message label
    status_var = tk.StringVar(value="")
    status_label = ttk.Label(tab1, textvariable=status_var, foreground="red")
    status_label.pack(pady=5)
    
    # Create all variables needed by functions first
    snr_var = tk.StringVar(value="10.0")
    num_peaks_var = tk.IntVar(value=2)
    points_across_var = tk.StringVar(value="12")
    display_mode_var = tk.StringVar(value="Normal")
    show_height_var = tk.BooleanVar(value=False)
    show_noise_var = tk.BooleanVar(value=False)
    show_peak_boundaries_var = tk.BooleanVar(value=False)
    
    # Define functions before they are referenced
    # Function to update SNR visualization
    def update_snr_viz():
        # Get the desired number of points across peak
        try:
            points_across_peak = int(points_across_var.get())
            if points_across_peak < 3:
                points_across_peak = 3
        except ValueError:
            points_across_peak = 12  # Default if invalid
        
        # Store previous settings to check what changed
        previous_peaks = len(shared_data.get('peaks', [])) if shared_data.get('peaks') else 0
        previous_snr = shared_data.get('snr', None)
        
        # First generate a high-resolution x-axis for creating the theoretical peaks
        # This ensures we have an accurate representation of the true signal
        high_res_x = np.linspace(0, 10, 10000)  # Very high resolution
        
        # Get number of peaks
        num_peaks = num_peaks_var.get()
        
        # Generate peak parameters based on number of peaks - only if number changed or first run
        if previous_peaks != num_peaks or shared_data.get('peaks') is None:
            # Use current time as seed for randomness when peak positions need to change
            np.random.seed(int(time.time()))
            
            # Generate random positions with minimum spacing
            peaks = []
            min_spacing = 0.5  # Minimum spacing between peaks
            positions = []
            max_attempts = 100  # Prevent infinite loops
            
            for i in range(num_peaks):
                attempts = 0
                while attempts < max_attempts:
                    pos = np.random.uniform(0.8, 9.2)  # Buffer from edges
                    if all(abs(pos - p) >= min_spacing for p in positions):
                        positions.append(pos)
                        break
                    attempts += 1
                    
                if attempts == max_attempts:
                    pos = 1.0 + (8.0 * i / num_peaks)
                    positions.append(pos)
            
            # Sort positions for natural chromatogram appearance
            positions.sort()
            
            # Create peaks with random heights and widths
            for pos in positions:
                height = np.random.uniform(0.4, 1.0)
                width = np.random.uniform(0.1, 0.3)
                peaks.append((pos, height, width))
        else:
            # Keep existing peak positions and shapes
            peaks = shared_data.get('peaks', [])
        
        # Create high-resolution ideal signal
        high_res_y_true = np.zeros_like(high_res_x)
        for pos, height, width in peaks:
            peak = height * np.exp(-((high_res_x - pos) ** 2) / (2 * width ** 2))
            high_res_y_true += peak
            
        # Store peaks for other tabs
        shared_data['peaks'] = peaks
        
        # Calculate average peak width for sampling rate determination
        avg_width = np.mean([width for _, _, width in peaks]) if peaks else 0.2
        
        # Calculate sampling interval based on points across peak
        # Use 6*sigma as full peak width (base to base)
        full_peak_width = 6 * avg_width
        sampling_interval = full_peak_width / points_across_peak
        
        # Calculate total number of points for the sampled signal
        total_range = 10  # x-axis range
        total_points = int(total_range / sampling_interval)
        
        # Create properly sampled x-axis
        x_sampled = np.linspace(0, 10, total_points)
        
        # Interpolate the true signal from high resolution to our target resolution
        from scipy.interpolate import interp1d
        interp_func = interp1d(high_res_x, high_res_y_true, kind='linear')
        y_true_sampled = interp_func(x_sampled)
        
        # Get current SNR from entry
        try:
            snr = float(snr_var.get())
            if snr <= 0:
                snr = 0.1
        except ValueError:
            snr = 10.0
        shared_data['snr'] = snr
        
        # Calculate noise based on SNR
        signal_power = np.mean(y_true_sampled ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)
        shared_data['noise_std'] = noise_std
        
        # Generate noise for the sampled points
        np.random.seed(42)  # Same seed for consistent comparison
        noise = np.random.normal(0, noise_std, size=x_sampled.shape)
        
        # Add noise to sampled signal
        y_noisy_sampled = y_true_sampled + noise
        
        # Store data for other tabs
        shared_data['x'] = x_sampled
        shared_data['y_true'] = y_true_sampled
        shared_data['y_noisy'] = y_noisy_sampled
        shared_data['high_res_x'] = high_res_x
        shared_data['high_res_y_true'] = high_res_y_true
        
        # Find suitable regions for noise measurement
        # Adjust minimum region size based on sampling rate
        min_points_for_noise = max(points_across_peak, 5)
        
        # First, determine each peak's boundaries (±3σ from center)
        peak_regions = []
        for pos, _, width in peaks:
            # Use 3 sigma (~99.7% of peak area)
            start_idx = np.abs(x_sampled - (pos - 3 * width)).argmin()
            end_idx = np.abs(x_sampled - (pos + 3 * width)).argmin()
            peak_regions.append((start_idx, end_idx))
        
        # Flatten peak regions into a mask of used indices
        used_indices = np.zeros(len(x_sampled), dtype=bool)
        for start_idx, end_idx in peak_regions:
            used_indices[start_idx:end_idx] = True
        
        # Find potential noise regions
        noise_regions = []
        in_region = False
        start_idx = 0
        
        for i in range(len(used_indices)):
            if not used_indices[i] and not in_region:
                # Start of a potential region
                start_idx = i
                in_region = True
            elif (used_indices[i] or i == len(used_indices)-1) and in_region:
                # End of a region
                end_idx = i
                if end_idx - start_idx >= min_points_for_noise:
                    noise_regions.append((start_idx, end_idx))
                in_region = False
        
        # Store noise regions for use in other tabs
        shared_data['noise_regions'] = noise_regions
        
        # Clear previous status
        status_var.set("")
        
        # Check if we found suitable noise regions
        if not noise_regions:
            status_var.set("Warning: Not enough noise-free regions to estimate background. Too many peaks or peaks too wide.")
            estimated_noise = noise_std  # Use theoretical noise as fallback
        else:
            # Select the first noise region for measurement
            noise_start_idx, noise_end_idx = noise_regions[0]
            noise_region = y_noisy_sampled[noise_start_idx:noise_end_idx]
            estimated_noise = np.std(noise_region)
        
        # Update plot
        ax1.clear()
        
        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Plot the high-resolution ideal signal
        ax1.plot(high_res_x, high_res_y_true, 'g-', linewidth=2, alpha=0.7, label='Ideal signal')
        
        # Check display mode
        if display_mode_var.get() == "High-Res with Noise":
            # Generate high-resolution noise with the same characteristics
            high_res_noise = np.random.normal(0, noise_std, size=high_res_x.shape)
            high_res_y_noisy = high_res_y_true + high_res_noise
            
            # Plot high-resolution signal with noise in blue
            ax1.plot(high_res_x, high_res_y_noisy, 'b-', linewidth=1, label='High-res with noise')
            
            # Plot the sampled points as red dots with connecting lines
            ax1.plot(x_sampled, y_noisy_sampled, 'r-', linewidth=1, label='Sampled signal')
        else:
            # Original mode - just plot the sampled signal in blue (no markers)
            ax1.plot(x_sampled, y_noisy_sampled, 'b-', linewidth=1, label='Signal with noise')
        
        # Mark peak positions and show height measurements if requested
        for pos, height, width in peaks:
            peak_idx = np.abs(x_sampled - pos).argmin()
            measured_height = y_noisy_sampled[peak_idx]
            
            # Calculate SNR for this peak
            peak_snr = measured_height / estimated_noise
            
            # Plot peak marker
            ax1.plot(pos, measured_height, 'ro', markersize=6)
            
            # If show height is enabled, draw lines showing height measurement
            if show_height_var.get():
                # Draw vertical line from x-axis to peak
                ax1.plot([pos, pos], [0, measured_height], 'r--', linewidth=1)
                
                # Draw peak boundaries with vertical dashed black lines
                lower_bound = pos - 3 * width
                upper_bound = pos + 3 * width
                
                ax1.plot([lower_bound, lower_bound], [0, measured_height*1.1], 'k--', linewidth=1)
                ax1.plot([upper_bound, upper_bound], [0, measured_height*1.1], 'k--', linewidth=1)
                
                # Add text with height and SNR values
                ax1.text(pos + 0.1, measured_height/2, 
                         f"Height: {measured_height:.2f}\nSNR: {peak_snr:.1f}", 
                         fontsize=12, verticalalignment='center')
        
        # If show noise is enabled, highlight all noise regions
        if show_noise_var.get() and noise_regions:
            for noise_start_idx, noise_end_idx in noise_regions:
                # Highlight noise region
                noise_x_start = x_sampled[noise_start_idx]
                noise_x_end = x_sampled[noise_end_idx]
                ax1.axvspan(noise_x_start, noise_x_end, alpha=0.2, color='yellow', label='Noise region' if noise_regions.index((noise_start_idx, noise_end_idx)) == 0 else "")
            
            # Use first noise region for measurements
            noise_start_idx, noise_end_idx = noise_regions[0]
            noise_x_start = x_sampled[noise_start_idx]
            noise_x_end = x_sampled[noise_end_idx]
            
            # Add horizontal lines showing the +/- standard deviation of noise
            mean_noise = np.mean(y_noisy_sampled[noise_start_idx:noise_end_idx])
            ax1.axhline(y=mean_noise + estimated_noise, color='orange', linestyle='--', alpha=0.7)
            ax1.axhline(y=mean_noise - estimated_noise, color='orange', linestyle='--', alpha=0.7)
        
        # Add annotation for noise measurement
        ax1.text(0.02, 0.05, 
                 f"Noise SD: {estimated_noise:.3f}", 
                 fontsize=12, color='red', transform=ax1.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # Add annotation for sampling rate
        # Calculate actual points across peak width
        full_peak_width = 6 * avg_width  # 6 sigma = ~99.7% of peak
        points_per_unit = len(x_sampled) / 10.0  # 10 is our x-axis range
        actual_points = int(full_peak_width * points_per_unit)
        ax1.text(0.02, 0.12, 
                f"Points across peak: {actual_points}", 
                fontsize=12, color='blue', transform=ax1.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Set labels with increased font size
        ax1.set_xlabel('Retention Time (min)', fontsize=14)
        ax1.set_ylabel('Detector Response', fontsize=14)
        
        # Remove the title
        # ax1.set_title('Impact of Signal-to-Noise Ratio on Peak Detection', fontsize=14)
        
        # Increase tick label font size
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        ax1.legend(loc='upper right', fontsize=12)
        fig1.tight_layout()
        canvas1.draw()
        
        # Update the Detection Methods tab
        update_methods_viz()
    
    # Function to update detection methods visualization
    def update_methods_viz():
        # Check if we have data from the first tab
        if shared_data['x'] is None or shared_data['y_noisy'] is None:
            return
        
        # Get shared data from first tab
        x = shared_data['x']
        y_noisy = shared_data['y_noisy']
        peaks = shared_data['peaks']
        snr = shared_data['snr']
        
        # Calculate appropriate window size for Savitzky-Golay filter based on peak width
        # Use 1/2 the number of points across the narrowest peak
        if peaks:
            # Find narrowest peak width
            min_width = min([width for _, _, width in peaks])
            # Convert width to number of points (assuming x range and number of points)
            x_range = x[-1] - x[0]
            points_per_unit = len(x) / x_range
            # Calculate points across peak (based on actual sampling)
            narrowest_peak_points = int(6 * min_width * points_per_unit)
            # Use half of that, and make sure it's odd
            sg_window = max(narrowest_peak_points // 2, 5)
            if sg_window % 2 == 0:
                sg_window += 1  # Make window odd-sized as required by savgol_filter
        else:
            sg_window = min(int(len(x) * 0.05), 51)  # Default fallback
            
        # Clear the entire figure and recreate the axes
        fig2.clear()
        ax2 = fig2.add_subplot(111)
        
        # Only create derivative axis if needed
        show_derivatives = show_first_deriv_var.get() or show_second_deriv_var.get()
        if show_derivatives:
            ax2_deriv = ax2.twinx()
        
        # Plot raw signal and smoothed signal on primary axis
        ax2.plot(x, y_noisy, 'b-', linewidth=1, label='Signal with noise')
        
        # Method 2: Savitzky-Golay filter (better smoothing)
        y_savgol = savgol_filter(y_noisy, window_length=sg_window, polyorder=3)
        ax2.plot(x, y_savgol, 'g-', linewidth=1.5, label='Savitzky-Golay smooth')
        
        # Calculate derivatives if they'll be used
        if show_derivatives:
            # Method 3: First derivative for peak detection - on secondary axis
            y_deriv = savgol_filter(y_noisy, window_length=sg_window, polyorder=3, deriv=1)
            
            # Method 4: Second derivative for peak detection - on secondary axis
            y_deriv2 = savgol_filter(y_noisy, window_length=sg_window, polyorder=3, deriv=2)
            
            # Plot derivatives based on checkbox states
            if show_first_deriv_var.get():
                ax2_deriv.plot(x, y_deriv, 'm-', linewidth=1, label='First derivative')
                
            if show_second_deriv_var.get():
                ax2_deriv.plot(x, y_deriv2, 'c-', linewidth=1, label='Second derivative')
        
        # Add peak positions and boundaries
        for pos, height, width in peaks:
            peak_idx = np.abs(x - pos).argmin()
            measured_height = y_noisy[peak_idx]
            
            # Plot peak marker
            ax2.plot(pos, measured_height, 'ro', markersize=6)
            
            # If show peak boundaries is enabled
            if show_peak_boundaries_var.get():
                # Draw peak boundaries with vertical dashed black lines
                lower_bound = pos - 3 * width
                upper_bound = pos + 3 * width
                
                # Replace the axvspan with vertical dashed lines
                ax2.plot([lower_bound, lower_bound], [0, measured_height*1.1], 'k--', linewidth=1)
                ax2.plot([upper_bound, upper_bound], [0, measured_height*1.1], 'k--', linewidth=1)
        
        # Add info about window size used
        ax2.text(0.02, 0.05, 
                f"SG Window: {sg_window} points", 
                fontsize=10, color='green', transform=ax2.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7))
                
        # Set labels for both axes with increased font size
        ax2.set_xlabel('Retention Time (min)', fontsize=14)
        ax2.set_ylabel('Detector Response', color='b', fontsize=14)
        
        # Only set up derivative axis if derivatives are shown
        if show_derivatives:
            ax2_deriv.set_ylabel('Derivative Values', color='k', fontsize=14)  # Use black for derivatives
            
            # Add legends for both y-axes
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_deriv.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
            
            # Adjust colors of tick labels to match the line colors
            ax2.tick_params(axis='y', colors='b', labelsize=12)
            ax2_deriv.tick_params(axis='y', colors='k', labelsize=12)  # Use black for derivative axis ticks
            ax2.tick_params(axis='x', labelsize=12)
        else:
            # Just add legend for the primary axis
            ax2.legend(loc='upper right', fontsize=12)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            
        # Set title with increased font size
        ax2.set_title(f'Comparison of Detection Methods at SNR = {snr:.1f}', fontsize=14)
        
        fig2.tight_layout()
        canvas2.draw()
    
    # Now create the UI elements and reference the functions
    # SNR control with text entry
    ttk.Label(controls_frame1, text="SNR Value:").pack(side=tk.LEFT, padx=5)
    snr_entry = ttk.Entry(controls_frame1, textvariable=snr_var, width=6)
    snr_entry.pack(side=tk.LEFT, padx=5)

    # Add validation for numeric input - improved version
    def validate_snr_input(*args):
        try:
            value = float(snr_var.get())
            if value <= 0:
                snr_var.set("0.1")  # Set a minimum value
        except ValueError:
            # Only reset if the value is completely invalid
            if snr_var.get().strip() and not snr_var.get() == "-":  # Allow typing negative
                snr_var.set("10.0")  # Reset to default if not a valid number
    
    # Use after_idle to delay validation slightly, allowing text replacement to occur first
    def delayed_validation(*args):
        sim_window.after_idle(validate_snr_input)
    
    snr_var.trace("w", delayed_validation)
    
    # Number of peaks control
    ttk.Label(controls_frame1, text="Number of Peaks:").pack(side=tk.LEFT, padx=15)
    num_peaks_spinbox = ttk.Spinbox(controls_frame1, from_=1, to=10, textvariable=num_peaks_var, width=5)
    num_peaks_spinbox.pack(side=tk.LEFT, padx=5)
    
    # Points across peak control
    ttk.Label(controls_frame1, text="Points Across Peak:").pack(side=tk.LEFT, padx=15)
    points_entry = ttk.Entry(controls_frame1, textvariable=points_across_var, width=4)
    points_entry.pack(side=tk.LEFT, padx=5)
    
    # Validate points across peak input with delayed validation
    def validate_points_input(*args):
        try:
            value = int(points_across_var.get())
            if value < 3:
                points_across_var.set("3")  # Set a minimum value for reliable peak detection
        except ValueError:
            if points_across_var.get().strip():
                points_across_var.set("12")  # Default
    
    # Use after_idle to delay validation slightly, allowing text replacement to occur first
    def delayed_points_validation(*args):
        sim_window.after_idle(validate_points_input)
    
    points_across_var.trace("w", delayed_points_validation)
    
    # Add display options combo box in the second row
    ttk.Label(controls_frame1_row2, text="Display Mode:").pack(side=tk.LEFT, padx=5)
    display_mode_combo = ttk.Combobox(controls_frame1_row2, textvariable=display_mode_var, 
                                    values=["Normal", "High-Res with Noise"],
                                    width=15, state="readonly")
    display_mode_combo.pack(side=tk.LEFT, padx=5)
    
    # Visualization options in the second row
    show_height_check = ttk.Checkbutton(controls_frame1_row2, text="Show Peak Heights", variable=show_height_var)
    show_height_check.pack(side=tk.LEFT, padx=15)
    
    show_noise_check = ttk.Checkbutton(controls_frame1_row2, text="Show Noise Region", variable=show_noise_var)
    show_noise_check.pack(side=tk.LEFT, padx=15)
    
    # Update button for tab1 (in the second row)
    ttk.Button(controls_frame1_row2, text="Update Visualization", command=update_snr_viz).pack(side=tk.LEFT, padx=20)

    # Tab 2: Detection Methods
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Detection Methods Comparison")
    
    # Create matplotlib figure for detection methods
    fig2 = Figure(figsize=(9, 5), dpi=100)
    ax2 = fig2.add_subplot(111)
    canvas2 = FigureCanvasTkAgg(fig2, tab2)
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Controls frame for tab2
    controls_frame2 = ttk.Frame(tab2)
    controls_frame2.pack(fill=tk.X, pady=5)
    
    # Show peak boundaries checkbox
    show_peak_boundaries_check = ttk.Checkbutton(controls_frame2, text="Show Peak Boundaries", variable=show_peak_boundaries_var)
    show_peak_boundaries_check.pack(side=tk.LEFT, padx=15)

    # Create variables for controlling derivatives visibility
    show_first_deriv_var = tk.BooleanVar(value=True)
    show_second_deriv_var = tk.BooleanVar(value=True)
    
    # Add checkboxes for controlling derivatives
    show_first_deriv_check = ttk.Checkbutton(controls_frame2, text="Show 1st Derivative", variable=show_first_deriv_var)
    show_first_deriv_check.pack(side=tk.LEFT, padx=15)
    
    show_second_deriv_check = ttk.Checkbutton(controls_frame2, text="Show 2nd Derivative", variable=show_second_deriv_var)
    show_second_deriv_check.pack(side=tk.LEFT, padx=15)
    
    # Update button for tab2
    ttk.Button(controls_frame2, text="Update Methods", command=update_methods_viz).pack(side=tk.LEFT, padx=20)
    
    # Tab 3: Educational Information
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text="Educational Resources")
    
    # Create a text widget with educational content
    text_widget = tk.Text(tab3, wrap=tk.WORD, font=("Arial", 11))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(tab3, command=text_widget.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.config(yscrollcommand=scrollbar.set)
    
    # Educational content
    edu_content = """# Signal-to-Noise Ratio in Chromatography

## What is Signal-to-Noise Ratio (SNR)?

Signal-to-Noise Ratio (SNR) is a measure that compares the level of a desired signal to the level of background noise. In chromatography, SNR is critical for determining the limit of detection (LOD) and limit of quantitation (LOQ) of an analytical method.

SNR is typically calculated as:
   SNR = Signal Height / Noise Height

Where:
- Signal Height is the height of the peak from the baseline
- Noise Height is typically measured as the standard deviation of the baseline in a region without peaks

## Importance in Analytical Chemistry

1. **Detection Limits**: According to standard practice:
   - LOD is defined at SNR = 3
   - LOQ is defined at SNR = 10

2. **Data Quality**: Higher SNR values indicate more reliable data with better precision and accuracy.

3. **Method Validation**: SNR is a key parameter in method validation procedures.

## Factors Affecting SNR in Chromatography

1. **Instrument Factors**:
   - Detector sensitivity and type
   - Electronic noise in the detector
   - Temperature fluctuations
   - Pump pulsations

2. **Method Factors**:
   - Mobile phase composition
   - Flow rate
   - Column efficiency
   - Sample preparation techniques

3. **Sample Factors**:
   - Analyte concentration
   - Sample matrix complexity
   - Interfering compounds

## Improving SNR in Chromatographic Analysis

1. **Signal Enhancement**:
   - Increase sample concentration
   - Use more sensitive detectors
   - Optimize chromatographic conditions
   - Use derivatization to enhance detector response

2. **Noise Reduction**:
   - Electronic filtering
   - Temperature control
   - Proper grounding of instruments
   - Regular maintenance of equipment
   - Use of high-purity solvents and reagents

3. **Data Processing**:
   - Signal averaging
   - Smoothing algorithms (e.g., Savitzky-Golay)
   - Baseline correction
   - Digital filters

## Practical Implications

In real-world chromatographic analysis, the SNR directly impacts:

- **Reliability**: Can you trust that a small peak is a real analyte or just noise?
- **Quantification**: Low SNR leads to poor precision in quantitative analysis
- **Trace Analysis**: Detection of compounds at very low concentrations requires excellent SNR
- **Automated Processing**: Software peak detection algorithms struggle with low SNR data

This simulation demonstrates these challenges by allowing you to experience how different SNR values affect your ability to detect peaks, mirroring the real-world challenges faced in analytical laboratories.
"""
    
    text_widget.insert(tk.END, edu_content)
    text_widget.config(state=tk.DISABLED)  # Make it read-only

    # Initial update for first tab
    update_snr_viz()
    
    # Make tabs call appropriate update functions when selected
    def on_tab_change(event):
        selected_tab = notebook.index(notebook.select())
        if selected_tab == 1:  # Detection Methods tab
            update_methods_viz()
    
    notebook.bind("<<NotebookTabChanged>>", on_tab_change)
    
    sim_window.mainloop()

if __name__ == "__main__":
    signal_to_noise_simulator()