import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
from scipy.signal import savgol_filter

class ChromatographyGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Chromatography Peak Detection Game")
        self.root.geometry("1000x700")
        self.root.config(bg="#f0f0f0")
        
        # Game state variables
        self.score = 0
        self.round = 0
        self.max_rounds = 10
        self.current_snr = 10  # Starting SNR
        self.true_peaks = []
        self.user_detected_peaks = []
        self.game_over = False
        self.time_start = 0
        self.time_limit = 30  # seconds per round
        
        # Initialize UI
        self.create_widgets()
        self.start_new_round()
        
    def create_widgets(self):
        # Game frame
        game_frame = ttk.Frame(self.root, padding="10")
        game_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top info frame
        info_frame = ttk.Frame(game_frame)
        info_frame.pack(fill=tk.X, pady=10)
        
        # Score label
        self.score_label = ttk.Label(info_frame, text="Score: 0", font=("Arial", 14))
        self.score_label.pack(side=tk.LEFT, padx=10)
        
        # Round label
        self.round_label = ttk.Label(info_frame, text="Round: 1/10", font=("Arial", 14))
        self.round_label.pack(side=tk.LEFT, padx=10)
        
        # SNR label
        self.snr_label = ttk.Label(info_frame, text=f"Signal-to-Noise Ratio: {self.current_snr}", font=("Arial", 14))
        self.snr_label.pack(side=tk.LEFT, padx=10)
        
        # Timer label
        self.timer_label = ttk.Label(info_frame, text="Time: 30s", font=("Arial", 14))
        self.timer_label.pack(side=tk.RIGHT, padx=10)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, game_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add click event to the plot
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Bottom frame for controls
        controls_frame = ttk.Frame(game_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Instructions label
        instructions = "Click on the chromatogram to mark where you think peaks are located. The closer you are to the actual peaks, the more points you'll earn!"
        ttk.Label(controls_frame, text=instructions, wraplength=800).pack(pady=10)
        
        # Buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Submit", command=self.submit_round).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Selections", command=self.clear_selections).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Instructions", command=self.show_instructions).pack(side=tk.LEFT, padx=5)
        
    def generate_chromatogram(self, snr):
        """Generate a synthetic chromatogram with given SNR"""
        # Time points
        self.x = np.linspace(0, 10, 1000)
        
        # Generate baseline signal (zero)
        self.y_true = np.zeros_like(self.x)
        
        # Add 2-5 random peaks
        num_peaks = np.random.randint(2, 6)
        self.true_peaks = []
        
        for _ in range(num_peaks):
            # Random peak position
            peak_pos = np.random.uniform(1, 9)
            # Random peak height (larger = more visible)
            peak_height = np.random.uniform(0.5, 1.5)
            # Random peak width
            peak_width = np.random.uniform(0.1, 0.3)
            
            # Create Gaussian peak
            peak = peak_height * np.exp(-((self.x - peak_pos) ** 2) / (2 * peak_width ** 2))
            self.y_true += peak
            
            # Store peak information
            self.true_peaks.append((peak_pos, peak_height, peak_width))
        
        # Scale signal based on SNR
        signal_power = np.mean(self.y_true ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)
        
        # Add noise to the signal
        noise = np.random.normal(0, noise_std, size=self.x.shape)
        self.y_noisy = self.y_true + noise
        
        return self.x, self.y_noisy
    
    def plot_chromatogram(self):
        """Plot the current chromatogram with user selections"""
        self.ax.clear()
        self.ax.plot(self.x, self.y_noisy, 'b-', linewidth=1)
        self.ax.set_xlabel('Retention Time (min)')
        self.ax.set_ylabel('Detector Response')
        self.ax.set_title('Chromatogram - Find the Peaks!')
        
        # Plot user selections
        if self.user_detected_peaks:
            user_x = [pos for pos, _ in self.user_detected_peaks]
            user_y = [height for _, height in self.user_detected_peaks]
            self.ax.plot(user_x, user_y, 'ro', markersize=8, label='Your selections')
            
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()
    
    def on_click(self, event):
        """Handle click events on the plot"""
        if event.inaxes != self.ax or self.game_over:
            return
        
        # Get x,y coordinates of the click
        x_click = event.xdata
        y_click = event.ydata
        
        # Add to detected peaks
        self.user_detected_peaks.append((x_click, y_click))
        
        # Update plot
        self.plot_chromatogram()
    
    def clear_selections(self):
        """Clear all user peak selections"""
        self.user_detected_peaks = []
        self.plot_chromatogram()
    
    def start_new_round(self):
        """Start a new game round"""
        self.round += 1
        self.round_label.config(text=f"Round: {self.round}/{self.max_rounds}")
        
        # Adjust SNR based on round (gets harder)
        if self.round > 1:
            self.current_snr = max(2, 10 - (self.round - 1))
        self.snr_label.config(text=f"Signal-to-Noise Ratio: {self.current_snr}")
        
        # Generate new chromatogram
        self.generate_chromatogram(self.current_snr)
        
        # Reset user selections
        self.user_detected_peaks = []
        
        # Update plot
        self.plot_chromatogram()
        
        # Reset timer
        self.time_start = time.time()
        self.update_timer()
    
    def update_timer(self):
        """Update the timer display"""
        if self.game_over:
            return
            
        elapsed = time.time() - self.time_start
        remaining = max(0, self.time_limit - elapsed)
        
        self.timer_label.config(text=f"Time: {int(remaining)}s")
        
        if remaining <= 0:
            self.submit_round()
        else:
            self.root.after(1000, self.update_timer)
    
    def calculate_score(self):
        """Calculate score based on accuracy of peak detection"""
        score_for_round = 0
        
        # If no peaks detected, score is 0
        if not self.user_detected_peaks:
            return 0
            
        # For each true peak, find the closest user detection
        for true_peak in self.true_peaks:
            true_pos = true_peak[0]
            
            # Find closest user peak
            closest_dist = float('inf')
            for user_pos, _ in self.user_detected_peaks:
                dist = abs(true_pos - user_pos)
                closest_dist = min(closest_dist, dist)
            
            # Score based on distance (closer = higher score)
            if closest_dist < 0.2:  # Very accurate
                score_for_round += 100
            elif closest_dist < 0.5:  # Good detection
                score_for_round += 50
            elif closest_dist < 1.0:  # Decent detection
                score_for_round += 25
        
        # Penalty for false positives (user peaks that don't match true peaks)
        false_positives = 0
        for user_pos, _ in self.user_detected_peaks:
            is_false_positive = True
            for true_peak in self.true_peaks:
                true_pos = true_peak[0]
                if abs(user_pos - true_pos) < 1.0:  # If within reasonable distance
                    is_false_positive = False
                    break
            
            if is_false_positive:
                false_positives += 1
        
        # Penalty for false positives
        score_for_round -= false_positives * 25
        
        # Ensure score is not negative
        return max(0, score_for_round)
    
    def submit_round(self):
        """Submit current round and update score"""
        # Calculate score for this round
        round_score = self.calculate_score()
        self.score += round_score
        
        # Update score display
        self.score_label.config(text=f"Score: {self.score}")
        
        # Show round results
        messagebox.showinfo("Round Results", 
            f"Round {self.round} complete!\n"
            f"You scored {round_score} points.\n"
            f"Total score: {self.score}")
        
        # Show true peaks
        self.show_true_peaks()
        
        # Check if game is over
        if self.round >= self.max_rounds:
            self.end_game()
        else:
            # Start next round
            self.start_new_round()
    
    def show_true_peaks(self):
        """Show the true peak locations"""
        self.ax.clear()
        
        # Plot noisy chromatogram
        self.ax.plot(self.x, self.y_noisy, 'b-', linewidth=1, label='Noisy signal')
        
        # Plot true chromatogram
        self.ax.plot(self.x, self.y_true, 'g-', linewidth=1, alpha=0.7, label='True signal')
        
        # Plot true peak positions
        true_x = [pos for pos, _, _ in self.true_peaks]
        true_y = [self.y_true[np.abs(self.x - pos).argmin()] for pos, _, _ in self.true_peaks]
        self.ax.plot(true_x, true_y, 'g^', markersize=10, label='True peaks')
        
        # Plot user selections
        if self.user_detected_peaks:
            user_x = [pos for pos, _ in self.user_detected_peaks]
            user_y = [height for _, height in self.user_detected_peaks]
            self.ax.plot(user_x, user_y, 'ro', markersize=8, label='Your selections')
        
        self.ax.set_xlabel('Retention Time (min)')
        self.ax.set_ylabel('Detector Response')
        self.ax.set_title('Round Results - True Peaks vs. Your Selections')
        self.ax.legend()
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def end_game(self):
        """End the game and show final score"""
        self.game_over = True
        
        # Display final score
        messagebox.showinfo("Game Over", 
            f"Game Over!\n"
            f"Your final score is {self.score} points.\n\n"
            f"SNR Impact Analysis:\n"
            f"- High SNR (8-10): Peaks are easy to identify\n"
            f"- Medium SNR (5-7): Some peaks may be obscured by noise\n"
            f"- Low SNR (2-4): Peak detection becomes very challenging\n\n"
            f"This demonstrates how signal-to-noise ratio affects our ability to detect analytical signals!")
        
        # Ask if player wants to play again
        if messagebox.askyesno("Play Again?", "Would you like to play again?"):
            self.reset_game()
        else:
            self.root.quit()
    
    def reset_game(self):
        """Reset the game state for a new game"""
        self.score = 0
        self.round = 0
        self.game_over = False
        self.score_label.config(text="Score: 0")
        
        # Start first round
        self.start_new_round()
    
    def show_instructions(self):
        """Show game instructions"""
        instructions = (
            "Chromatography Peak Detection Game\n\n"
            "This game simulates the challenge of detecting peaks in chromatography data with varying levels of noise.\n\n"
            "How to play:\n"
            "1. Look at the chromatogram displayed on the screen\n"
            "2. Click on the positions where you think peaks are located\n"
            "3. Submit your selections before the timer runs out\n"
            "4. You'll earn points based on how accurately you identify the true peak positions\n\n"
            "Educational Value:\n"
            "- Experience how signal-to-noise ratio (SNR) affects peak detection\n"
            "- As you progress through rounds, the SNR decreases, making detection more challenging\n"
            "- This simulates real analytical chemistry challenges where instrument noise can mask important signals\n\n"
            "Tips:\n"
            "- Look for characteristic Gaussian peak shapes\n"
            "- Consider the signal pattern and try to distinguish patterns from random noise\n"
            "- In later rounds with low SNR, focus on finding the most prominent peaks first"
        )
        
        messagebox.showinfo("Game Instructions", instructions)

def advanced_mode():
    """Implementation of an advanced mode with more educational features"""
    advanced_window = tk.Toplevel()
    advanced_window.title("Advanced Signal-to-Noise Analysis")
    advanced_window.geometry("1000x800")
    
    # Create notebook with tabs
    notebook = ttk.Notebook(advanced_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Tab 1: SNR Impact Visualization
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="SNR Impact Visualization")
    
    # Create matplotlib figure for SNR impact
    fig1 = Figure(figsize=(9, 5), dpi=100)
    ax1 = fig1.add_subplot(111)
    canvas1 = FigureCanvasTkAgg(fig1, tab1)
    canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Controls frame
    controls_frame1 = ttk.Frame(tab1)
    controls_frame1.pack(fill=tk.X, pady=5)
    
    # SNR control
    ttk.Label(controls_frame1, text="SNR Value:").pack(side=tk.LEFT, padx=5)
    snr_var = tk.DoubleVar(value=10.0)
    snr_slider = ttk.Scale(controls_frame1, from_=0.5, to=20, variable=snr_var, orient=tk.HORIZONTAL, length=200)
    snr_slider.pack(side=tk.LEFT, padx=5)
    snr_label = ttk.Label(controls_frame1, text="10.0")
    snr_label.pack(side=tk.LEFT, padx=5)
    
    # Number of peaks control
    ttk.Label(controls_frame1, text="Number of Peaks:").pack(side=tk.LEFT, padx=15)
    num_peaks_var = tk.IntVar(value=2)
    num_peaks_spinbox = ttk.Spinbox(controls_frame1, from_=1, to=10, textvariable=num_peaks_var, width=5)
    num_peaks_spinbox.pack(side=tk.LEFT, padx=5)
    
    # Visualization options
    show_height_var = tk.BooleanVar(value=False)
    show_height_check = ttk.Checkbutton(controls_frame1, text="Show Peak Heights", variable=show_height_var)
    show_height_check.pack(side=tk.LEFT, padx=15)
    
    show_noise_var = tk.BooleanVar(value=False)
    show_noise_check = ttk.Checkbutton(controls_frame1, text="Show Noise Region", variable=show_noise_var)
    show_noise_check.pack(side=tk.LEFT, padx=15)
    
    # Function to update SNR visualization
    def update_snr_viz():
        # Generate time points
        x = np.linspace(0, 10, 1000)
        
        # Generate true signal with requested number of peaks
        y_true = np.zeros_like(x)
        
        # Get number of peaks
        num_peaks = num_peaks_var.get()
        
        # Use current time as seed for true randomness when updating
        np.random.seed(int(time.time()))
        
        # Generate peak parameters based on number requested
        peaks = []
        for i in range(num_peaks):
            # Space peaks somewhat evenly but with randomness
            position = 1.0 + (8.0 * i / num_peaks) + np.random.uniform(-0.3, 0.3)
            height = np.random.uniform(0.4, 1.0)
            width = np.random.uniform(0.1, 0.3)
            peaks.append((position, height, width))
        
        # Create peaks
        for pos, height, width in peaks:
            peak = height * np.exp(-((x - pos) ** 2) / (2 * width ** 2))
            y_true += peak
        
        # Get current SNR from slider
        snr = snr_var.get()
        snr_label.config(text=f"{snr:.1f}")
        
        # Calculate noise based on SNR
        signal_power = np.mean(y_true ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)
        
        # Generate noise
        np.random.seed(42)  # Use same seed for consistent comparison within the same view
        noise = np.random.normal(0, noise_std, size=x.shape)
        
        # Add noise to signal
        y_noisy = y_true + noise
        
        # Update plot
        ax1.clear()
        ax1.plot(x, y_true, 'g-', linewidth=2, alpha=0.7, label='True signal')
        ax1.plot(x, y_noisy, 'b-', linewidth=1, label='Noisy signal')
        
        # Find suitable regions for noise measurement
        # First, determine each peak's boundaries (±3σ from center)
        peak_regions = []
        for pos, _, width in peaks:
            # Use 3 sigma (~99.7% of peak area)
            start_idx = np.abs(x - (pos - 3 * width)).argmin()
            end_idx = np.abs(x - (pos + 3 * width)).argmin()
            peak_regions.append((start_idx, end_idx))
        
        # Find noise regions: sections between peaks
        noise_regions = []
        all_indices = sorted([(idx, 'start') for pos, _, width in peaks 
                             for idx in [np.abs(x - (pos - 3 * width)).argmin()]] + 
                            [(idx, 'end') for pos, _, width in peaks 
                             for idx in [np.abs(x - (pos + 3 * width)).argmin()]])
        
        # Find gaps between peaks
        last_idx = 50  # Start a bit into the chromatogram
        for idx, type_ in all_indices:
            if type_ == 'start' and idx - last_idx > 50:  # Minimum 50 points for noise region
                noise_regions.append((last_idx, idx - 10))  # Leave small buffer before peak
            if type_ == 'end':
                last_idx = idx + 10  # Leave small buffer after peak
        
        # Add final region if there's space
        if len(x) - last_idx > 50 and last_idx < len(x) - 50:
            noise_regions.append((last_idx, len(x) - 50))
        
        # If no suitable regions found, use beginning and end of chromatogram
        if not noise_regions:
            if peak_regions[0][0] > 100:  # If there's space at beginning
                noise_regions.append((50, peak_regions[0][0] - 20))
            if peak_regions[-1][1] < len(x) - 100:  # If there's space at end
                noise_regions.append((peak_regions[-1][1] + 20, len(x) - 50))
        
        # If still no noise regions, create two default regions
        if not noise_regions:
            noise_regions = [(50, 100), (900, 950)]
        
        # Select the first noise region for measurement
        noise_start_idx, noise_end_idx = noise_regions[0]
        noise_region = y_noisy[noise_start_idx:noise_end_idx]
        estimated_noise = np.std(noise_region)
        
        # Mark peak positions and show height measurements if requested
        for pos, height, width in peaks:
            peak_idx = np.abs(x - pos).argmin()
            measured_height = y_noisy[peak_idx]
            
            # Calculate SNR for this peak
            peak_snr = measured_height / estimated_noise
            
            # Plot peak marker
            ax1.plot(pos, measured_height, 'ro', markersize=6)
            
            # If show height is enabled, draw lines showing height measurement
            if show_height_var.get():
                # Draw vertical line from x-axis to peak
                ax1.plot([pos, pos], [0, measured_height], 'r--', linewidth=1)
                
                # Draw peak boundaries (±3σ from center)
                lower_bound = pos - 3 * width
                upper_bound = pos + 3 * width
                ax1.axvspan(lower_bound, upper_bound, alpha=0.1, color='pink')
                
                # Add bracket markers for peak width
                y_bracket = measured_height * 0.8
                ax1.plot([lower_bound, upper_bound], [y_bracket, y_bracket], 'r-', linewidth=1)
                ax1.plot([lower_bound, lower_bound], [y_bracket-0.02, y_bracket+0.02], 'r-', linewidth=1)
                ax1.plot([upper_bound, upper_bound], [y_bracket-0.02, y_bracket+0.02], 'r-', linewidth=1)
                
                # Add text with height and SNR values
                ax1.text(pos + 0.1, measured_height/2, 
                         f"Height: {measured_height:.2f}\nSNR: {peak_snr:.1f}", 
                         fontsize=8, verticalalignment='center')
        
        # If show noise is enabled, highlight all noise regions
        if show_noise_var.get():
            for noise_start_idx, noise_end_idx in noise_regions:
                # Highlight noise region
                noise_x_start = x[noise_start_idx]
                noise_x_end = x[noise_end_idx]
                ax1.axvspan(noise_x_start, noise_x_end, alpha=0.2, color='yellow', label='Noise region' if noise_regions.index((noise_start_idx, noise_end_idx)) == 0 else "")
            
            # Use first noise region for measurements
            noise_start_idx, noise_end_idx = noise_regions[0]
            noise_x_start = x[noise_start_idx]
            noise_x_end = x[noise_end_idx]
            
            # Add horizontal lines showing the +/- standard deviation of noise
            mean_noise = np.mean(y_noisy[noise_start_idx:noise_end_idx])
            ax1.axhline(y=mean_noise + estimated_noise, color='orange', linestyle='--', alpha=0.7)
            ax1.axhline(y=mean_noise - estimated_noise, color='orange', linestyle='--', alpha=0.7)
            
            # Add annotation for noise measurement
            ax1.text(noise_x_end + 0.1, mean_noise, 
                     f"Noise SD: {estimated_noise:.3f}", 
                     fontsize=8, verticalalignment='center')
        
        # Add summary information
        ax1.text(0.02, 0.95, f"Theoretical SNR: {snr:.1f}", transform=ax1.transAxes, fontsize=9)
        
        ax1.set_xlabel('Retention Time (min)')
        ax1.set_ylabel('Detector Response')
        ax1.set_title('Impact of Signal-to-Noise Ratio on Peak Detection')
        ax1.legend(loc='upper right')
        fig1.tight_layout()
        canvas1.draw()
    
    # Update button
    ttk.Button(controls_frame1, text="Update Visualization", command=update_snr_viz).pack(side=tk.LEFT, padx=20)
    
    # Initial update
    update_snr_viz()
    
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
    
    ttk.Label(controls_frame2, text="SNR for Comparison:").pack(side=tk.LEFT, padx=5)
    method_snr_var = tk.DoubleVar(value=5.0)
    method_snr_slider = ttk.Scale(controls_frame2, from_=1.0, to=15.0, variable=method_snr_var, orient=tk.HORIZONTAL, length=300)
    method_snr_slider.pack(side=tk.LEFT, padx=5)
    
    method_snr_label = ttk.Label(controls_frame2, text="5.0")
    method_snr_label.pack(side=tk.LEFT, padx=5)
    
    # Function to update detection methods visualization
    def update_methods_viz():
        # Generate time points
        x = np.linspace(0, 10, 1000)
        
        # Generate true signal with multiple peaks
        y_true = np.zeros_like(x)
        
        # Add several peaks with varying heights and widths
        peaks = [
            (2.0, 1.0, 0.2),   # (position, height, width)
            (3.5, 0.5, 0.15),
            (5.0, 0.3, 0.1),
            (7.0, 0.8, 0.25),
            (8.5, 0.4, 0.12)
        ]
        
        for pos, height, width in peaks:
            peak = height * np.exp(-((x - pos) ** 2) / (2 * width ** 2))
            y_true += peak
        
        # Get current SNR from slider
        snr = method_snr_var.get()
        method_snr_label.config(text=f"{snr:.1f}")
        
        # Calculate noise based on SNR
        signal_power = np.mean(y_true ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)
        
        # Generate noise
        np.random.seed(123)  # Fixed seed for consistency
        noise = np.random.normal(0, noise_std, size=x.shape)
        
        # Add noise to signal
        y_noisy = y_true + noise
        
        # Update plot
        ax2.clear()
        ax2.plot(x, y_noisy, 'b-', linewidth=1, label='Noisy signal')
        
        # Method 1: Moving average (simple smoothing)
        window_size = 25
        y_smooth = np.convolve(y_noisy, np.ones(window_size)/window_size, mode='same')
        ax2.plot(x, y_smooth, 'r-', linewidth=1, label='Moving average')
        
        # Method 2: Savitzky-Golay filter (better smoothing)
        y_savgol = savgol_filter(y_noisy, window_length=51, polyorder=3)
        ax2.plot(x, y_savgol, 'g-', linewidth=1, label='Savitzky-Golay')
        
        # Method 3: First derivative for peak detection
        from scipy.signal import savgol_filter
        y_deriv = savgol_filter(y_noisy, window_length=51, polyorder=3, deriv=1)
        # Scale derivative for visualization
        deriv_scaling = 0.2
        ax2.plot(x, deriv_scaling * y_deriv, 'm-', linewidth=1, label='First derivative')
        
        # Add true peak positions for reference
        peak_positions = [pos for pos, _, _ in peaks]
        peak_heights = [height for _, height, _ in peaks]
        ax2.plot(peak_positions, peak_heights, 'ko', markersize=6, label='True peaks')
        
        ax2.set_xlabel('Retention Time (min)')
        ax2.set_ylabel('Detector Response')
        ax2.set_title(f'Comparison of Detection Methods at SNR = {snr:.1f}')
        ax2.legend()
        fig2.tight_layout()
        canvas2.draw()
    
    # Update button for tab2
    ttk.Button(controls_frame2, text="Update Methods", command=update_methods_viz).pack(side=tk.LEFT, padx=20)
    
    # Initial update for tab2
    update_methods_viz()
    
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

    # Keep this window in focus
    advanced_window.focus_set()
    advanced_window.grab_set()

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
    
    # Controls frame
    controls_frame1 = ttk.Frame(tab1)
    controls_frame1.pack(fill=tk.X, pady=5)
    
    # SNR control
    ttk.Label(controls_frame1, text="SNR Value:").pack(side=tk.LEFT, padx=5)
    snr_var = tk.DoubleVar(value=10.0)
    snr_slider = ttk.Scale(controls_frame1, from_=0.5, to=20, variable=snr_var, orient=tk.HORIZONTAL, length=200)
    snr_slider.pack(side=tk.LEFT, padx=5)
    snr_label = ttk.Label(controls_frame1, text="10.0")
    snr_label.pack(side=tk.LEFT, padx=5)
    
    # Number of peaks control
    ttk.Label(controls_frame1, text="Number of Peaks:").pack(side=tk.LEFT, padx=15)
    num_peaks_var = tk.IntVar(value=2)
    num_peaks_spinbox = ttk.Spinbox(controls_frame1, from_=1, to=10, textvariable=num_peaks_var, width=5)
    num_peaks_spinbox.pack(side=tk.LEFT, padx=5)
    
    # Visualization options
    show_height_var = tk.BooleanVar(value=False)
    show_height_check = ttk.Checkbutton(controls_frame1, text="Show Peak Heights", variable=show_height_var)
    show_height_check.pack(side=tk.LEFT, padx=15)
    
    show_noise_var = tk.BooleanVar(value=False)
    show_noise_check = ttk.Checkbutton(controls_frame1, text="Show Noise Region", variable=show_noise_var)
    show_noise_check.pack(side=tk.LEFT, padx=15)

    # Status message label
    status_var = tk.StringVar(value="")
    status_label = ttk.Label(tab1, textvariable=status_var, foreground="red")
    status_label.pack(pady=5)
    
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
    
    # Function to update SNR visualization
    def update_snr_viz():
        # Generate time points
        x = np.linspace(0, 10, 1000)
        shared_data['x'] = x
        
        # Generate true signal with requested number of peaks
        y_true = np.zeros_like(x)
        
        # Get number of peaks
        num_peaks = num_peaks_var.get()
        
        # Use current time as seed for true randomness when updating
        np.random.seed(int(time.time()))
        
        # Generate peak parameters based on number requested
        peaks = []
        for i in range(num_peaks):
            # Space peaks somewhat evenly but with randomness
            position = 1.0 + (8.0 * i / num_peaks) + np.random.uniform(-0.3, 0.3)
            height = np.random.uniform(0.4, 1.0)
            width = np.random.uniform(0.1, 0.3)
            peaks.append((position, height, width))
        
        # Store peaks for other tabs
        shared_data['peaks'] = peaks
        
        # Create peaks
        for pos, height, width in peaks:
            peak = height * np.exp(-((x - pos) ** 2) / (2 * width ** 2))
            y_true += peak
        
        # Store true signal
        shared_data['y_true'] = y_true
        
        # Get current SNR from slider
        snr = snr_var.get()
        shared_data['snr'] = snr
        snr_label.config(text=f"{snr:.1f}")
        
        # Calculate noise based on SNR
        signal_power = np.mean(y_true ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)
        shared_data['noise_std'] = noise_std
        
        # Generate noise
        np.random.seed(42)  # Use same seed for consistent comparison within the same view
        noise = np.random.normal(0, noise_std, size=x.shape)
        
        # Add noise to signal
        y_noisy = y_true + noise
        shared_data['y_noisy'] = y_noisy
        
        # Update plot
        ax1.clear()
        ax1.plot(x, y_true, 'g-', linewidth=2, alpha=0.7, label='True signal')
        ax1.plot(x, y_noisy, 'b-', linewidth=1, label='Noisy signal')
        
        # Find suitable regions for noise measurement
        # First, determine each peak's boundaries (±3σ from center)
        peak_regions = []
        for pos, _, width in peaks:
            # Use 3 sigma (~99.7% of peak area)
            start_idx = np.abs(x - (pos - 3 * width)).argmin()
            end_idx = np.abs(x - (pos + 3 * width)).argmin()
            peak_regions.append((start_idx, end_idx))
        
        # Flatten peak regions into a mask of used indices
        used_indices = np.zeros(len(x), dtype=bool)
        for start_idx, end_idx in peak_regions:
            used_indices[start_idx:end_idx] = True
        
        # Find potential noise regions (at least 50 consecutive unused points)
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
                if end_idx - start_idx >= 50:  # Minimum 50 points for noise calculation
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
            noise_region = y_noisy[noise_start_idx:noise_end_idx]
            estimated_noise = np.std(noise_region)
        
        # Mark peak positions and show height measurements if requested
        for pos, height, width in peaks:
            peak_idx = np.abs(x - pos).argmin()
            measured_height = y_noisy[peak_idx]
            
            # Calculate SNR for this peak
            peak_snr = measured_height / estimated_noise
            
            # Plot peak marker
            ax1.plot(pos, measured_height, 'ro', markersize=6)
            
            # If show height is enabled, draw lines showing height measurement
            if show_height_var.get():
                # Draw vertical line from x-axis to peak
                ax1.plot([pos, pos], [0, measured_height], 'r--', linewidth=1)
                
                # Draw peak boundaries (±3σ from center)
                lower_bound = pos - 3 * width
                upper_bound = pos + 3 * width
                ax1.axvspan(lower_bound, upper_bound, alpha=0.1, color='pink')
                
                # Add bracket markers for peak width
                y_bracket = measured_height * 0.8
                ax1.plot([lower_bound, upper_bound], [y_bracket, y_bracket], 'r-', linewidth=1)
                ax1.plot([lower_bound, lower_bound], [y_bracket-0.02, y_bracket+0.02], 'r-', linewidth=1)
                ax1.plot([upper_bound, upper_bound], [y_bracket-0.02, y_bracket+0.02], 'r-', linewidth=1)
                
                # Add text with height and SNR values
                ax1.text(pos + 0.1, measured_height/2, 
                         f"Height: {measured_height:.2f}\nSNR: {peak_snr:.1f}", 
                         fontsize=12, verticalalignment='center')
        
        # If show noise is enabled, highlight all noise regions
        if show_noise_var.get() and noise_regions:
            for noise_start_idx, noise_end_idx in noise_regions:
                # Highlight noise region
                noise_x_start = x[noise_start_idx]
                noise_x_end = x[noise_end_idx]
                ax1.axvspan(noise_x_start, noise_x_end, alpha=0.2, color='yellow', label='Noise region' if noise_regions.index((noise_start_idx, noise_end_idx)) == 0 else "")
            
            # Use first noise region for measurements
            noise_start_idx, noise_end_idx = noise_regions[0]
            noise_x_start = x[noise_start_idx]
            noise_x_end = x[noise_end_idx]
            
            # Add horizontal lines showing the +/- standard deviation of noise
            mean_noise = np.mean(y_noisy[noise_start_idx:noise_end_idx])
            ax1.axhline(y=mean_noise + estimated_noise, color='orange', linestyle='--', alpha=0.7)
            ax1.axhline(y=mean_noise - estimated_noise, color='orange', linestyle='--', alpha=0.7)
        
        # Add summary information
        #ax1.text(0.02, 0.95, f"Theoretical SNR: {snr:.1f}", transform=ax1.transAxes, fontsize=9)
        
        # Add annotation for noise measurement - move to lower left with red text
        ax1.text(0.02, 0.05, 
                 f"Noise SD: {estimated_noise:.3f}", 
                 fontsize=12, color='red', transform=ax1.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.7))
        
        ax1.set_xlabel('Retention Time (min)')
        ax1.set_ylabel('Detector Response')
        ax1.set_title('Impact of Signal-to-Noise Ratio on Peak Detection')
        ax1.legend(loc='upper right')
        fig1.tight_layout()
        canvas1.draw()
        
        # Update the Detection Methods tab if data is available
        if hasattr(update_methods_viz, '__self__') and update_methods_viz.__self__ is not None:
            update_methods_viz()
    
    # Update button
    ttk.Button(controls_frame1, text="Update Visualization", command=update_snr_viz).pack(side=tk.LEFT, padx=20)
    
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
    show_peak_boundaries_var = tk.BooleanVar(value=False)
    show_peak_boundaries_check = ttk.Checkbutton(controls_frame2, text="Show Peak Boundaries", variable=show_peak_boundaries_var)
    show_peak_boundaries_check.pack(side=tk.LEFT, padx=15)
    
    # Function to update detection methods visualization
    def update_methods_viz():
        # Check if we have data from the first tab
        if shared_data['x'] is None or shared_data['y_noisy'] is None:
            return
        
        # Get shared data from first tab
        x = shared_data['x']
        y_true = shared_data['y_true']
        y_noisy = shared_data['y_noisy']
        peaks = shared_data['peaks']
        snr = shared_data['snr']
        
        # Update plot
        ax2.clear()
        ax2.plot(x, y_noisy, 'b-', linewidth=1, label='Noisy signal')
        
        # Method 2: Savitzky-Golay filter (better smoothing)
        y_savgol = savgol_filter(y_noisy, window_length=51, polyorder=3)
        ax2.plot(x, y_savgol, 'g-', linewidth=1, label='Savitzky-Golay')
        
        # Method 3: First derivative for peak detection
        y_deriv = savgol_filter(y_noisy, window_length=51, polyorder=3, deriv=1)
        # Scale derivative for visualization
        deriv_scaling = 0.2
        ax2.plot(x, deriv_scaling * y_deriv, 'm-', linewidth=1, label='First derivative')
        
        # Add peak positions and boundaries
        for pos, height, width in peaks:
            peak_idx = np.abs(x - pos).argmin()
            measured_height = y_noisy[peak_idx]
            
            # Plot peak marker
            ax2.plot(pos, measured_height, 'ro', markersize=6)
            
            # If show peak boundaries is enabled
            if show_peak_boundaries_var.get():
                # Draw peak boundaries (±3σ from center)
                lower_bound = pos - 3 * width
                upper_bound = pos + 3 * width
                ax2.axvspan(lower_bound, upper_bound, alpha=0.1, color='pink')
                
                # Add bracket markers for peak width
                y_bracket = measured_height * 0.8
                ax2.plot([lower_bound, upper_bound], [y_bracket, y_bracket], 'r-', linewidth=1)
                ax2.plot([lower_bound, lower_bound], [y_bracket-0.02, y_bracket+0.02], 'r-', linewidth=1)
                ax2.plot([upper_bound, upper_bound], [y_bracket-0.02, y_bracket+0.02], 'r-', linewidth=1)
        
        ax2.set_xlabel('Retention Time (min)')
        ax2.set_ylabel('Detector Response')
        ax2.set_title(f'Comparison of Detection Methods at SNR = {snr:.1f}')
        ax2.legend()
        fig2.tight_layout()
        canvas2.draw()
    
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

def main():
    root = tk.Tk()
    
    # Create main menu
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, text="Chromatography Signal-to-Noise Educational Game", font=("Arial", 16, "bold"))
    title_label.pack(pady=20)
    
    # Description
    description = (
        "This educational game helps you understand how signal-to-noise ratio affects peak detection in chromatography. "
        "Choose a mode to begin:"
    )
    desc_label = ttk.Label(main_frame, text=description, wraplength=400, justify=tk.CENTER)
    desc_label.pack(pady=20)
    
    # Buttons frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)
    
    # Game mode button
    game_button = ttk.Button(button_frame, text="Play Game Mode", width=20,
                          command=lambda: [root.withdraw(), ChromatographyGame(tk.Toplevel())])
    game_button.pack(pady=10)
    
    # Advanced mode button
    advanced_button = ttk.Button(button_frame, text="Advanced Analysis Mode", width=20,
                              command=advanced_mode)
    advanced_button.pack(pady=10)
    
    # Exit button
    exit_button = ttk.Button(button_frame, text="Exit", width=20, command=root.quit)
    exit_button.pack(pady=10)
    
    # Credits
    credits = "Created for educational purposes in analytical chemistry and instrumental analysis courses"
    credits_label = ttk.Label(main_frame, text=credits, font=("Arial", 9), foreground="gray")
    credits_label.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    signal_to_noise_simulator()