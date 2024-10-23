import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, List, Tuple
import seaborn as sns

class AestheticEEGVisualizer:
    def __init__(self, data_path: str, channels: List[str], sampling_freq: int = 128, batch_size: int = 5):
        """
        Initialize the Aesthetic EEG Visualizer.
        """
        # Set style
        plt.style.use('dark_background')
        self.colors = sns.color_palette("husl", n_colors=len(channels))
        
        # Data initialization
        self.df = pd.read_csv(data_path)
        self.selected_channels = channels
        self.sampling_freq = sampling_freq
        self.batch_size = batch_size
        self.x_data = {channel: [] for channel in channels}
        self.y_data = {channel: [] for channel in channels}
        
        # Create figure with proper spacing
        self.fig = plt.figure(figsize=(16, 10), facecolor='#1C1C1C')
        
        # Create GridSpec with proper spacing
        self.gs = GridSpec(
            len(channels) + 1,  # Add one more row for title
            1,
            figure=self.fig,
            height_ratios=[0.1] + [1] * len(channels),  # First ratio for title
            hspace=0.4  # Increase spacing between subplots
        )
        
        # Create title axis
        self.title_ax = self.fig.add_subplot(self.gs[0])
        self.title_ax.set_axis_off()
        
        # Create subplot axes
        self.axs = [self.fig.add_subplot(self.gs[i+1]) for i in range(len(channels))]
        
        # Setup plots with aesthetic elements
        self.setup_plots()

    def setup_plots(self):
        """Configure the initial plot settings with aesthetic elements."""
        # Add title to the figure
        self.title_ax.text(0.5, 0.5, 'Neural Activity Visualization',
                          fontsize=16,
                          color='#FFFFFF',
                          fontweight='bold',
                          ha='center',
                          va='center')
        
        for ax, channel, color in zip(self.axs, self.selected_channels, self.colors):
            # Set background color
            ax.set_facecolor('#2C2C2C')
            
            # Set y-axis limits with padding
            y_min, y_max = self.df[channel].min(), self.df[channel].max()
            padding = (y_max - y_min) * 0.2
            ax.set_ylim(y_min - padding, y_max + padding)
            
            # Style the grid
            ax.grid(True, alpha=0.2, linestyle='--', color='#808080')
            
            # Style the spines
            for spine in ax.spines.values():
                spine.set_color('#404040')
            
            # Style labels
            ax.set_ylabel(f'Channel {channel}', 
                         fontsize=12, 
                         color='#FFFFFF',
                         fontfamily='sans-serif',
                         fontweight='bold',
                         labelpad=10)  # Add padding to ylabel
            
            # Add channel indicator
            ax.text(0.02, 0.95, channel, 
                   transform=ax.transAxes,
                   color=color,
                   fontsize=14,
                   fontweight='bold',
                   bbox=dict(facecolor='#2C2C2C', 
                            edgecolor=color,
                            alpha=0.7,
                            pad=5))
            
            # Adjust margins
            ax.margins(x=0.02)

    def update(self, frame: int) -> List:
        """Update the animation frame with aesthetic elements."""
        start = frame * self.batch_size
        end = min((frame + 1) * self.batch_size, len(self.df))
        
        if start < len(self.df):
            time_in_seconds = [i / self.sampling_freq for i in range(start, end)]
            
            for ax, channel, color in zip(self.axs, self.selected_channels, self.colors):
                # Update data
                self.x_data[channel].extend(time_in_seconds)
                self.y_data[channel].extend(self.df[channel].iloc[start:end])
                
                # Clear and redraw
                ax.clear()
                
                # Create glow effect by plotting multiple lines with different alphas
                alphas = [0.1, 0.2, 0.3, 0.4, 1.0]
                line_widths = [4.0, 3.5, 3.0, 2.5, 2.0]
                
                for alpha, lw in zip(alphas, line_widths):
                    ax.plot(self.x_data[channel], self.y_data[channel],
                           color=color, 
                           lw=lw,
                           alpha=alpha)
                
                # Add subtle fill below the line
                ax.fill_between(self.x_data[channel], 
                              self.y_data[channel],
                              ax.get_ylim()[0],
                              alpha=0.1,
                              color=color)
                
                # Plot state markers with enhanced styling
                self._plot_state_markers(ax, time_in_seconds, start, end, channel, color)
                
                # Maintain aesthetic styling
                self._style_subplot(ax, channel, color)
                
                # Update x-axis window (show last 10 seconds)
                if len(self.x_data[channel]) > self.sampling_freq * 10:
                    ax.set_xlim(
                        self.x_data[channel][-self.sampling_freq * 10],
                        self.x_data[channel][-1]
                    )
        
        return self.axs

    def _plot_state_markers(self, ax, time_in_seconds: List[float], 
                          start: int, end: int, channel: str, color: str):
        """Plot focused and unfocused state markers with enhanced styling."""
        # Get state points
        focused_points = [
            (time_in_seconds[i], self.df[channel].iloc[start + i])
            for i in range(end - start)
            if self.df['state'].iloc[start + i] == 'focussed'
        ]
        
        unfocused_points = [
            (time_in_seconds[i], self.df[channel].iloc[start + i])
            for i in range(end - start)
            if self.df['state'].iloc[start + i] == 'unfocussed'
        ]
        
        # Plot enhanced markers
        marker_size = 100
        if focused_points:
            # Create glow effect for focused points
            for size, alpha in zip([marker_size*1.5, marker_size], [0.3, 0.6]):
                ax.scatter(*zip(*focused_points), 
                          s=size, 
                          color='#00FF00',
                          alpha=alpha,
                          marker='o',
                          edgecolor='white',
                          linewidth=1,
                          label="Focused" if alpha == 0.6 else "")
        
        if unfocused_points:
            # Create glow effect for unfocused points
            for size, alpha in zip([marker_size*1.5, marker_size], [0.3, 0.6]):
                ax.scatter(*zip(*unfocused_points), 
                          s=size, 
                          color='#FF4444',
                          alpha=alpha,
                          marker='o',
                          edgecolor='white',
                          linewidth=1,
                          label="Unfocused" if alpha == 0.6 else "")

    def _style_subplot(self, ax, channel: str, color: str):
        """Apply consistent aesthetic styling to a subplot."""
        # Background and grid
        ax.set_facecolor('#2C2C2C')
        ax.grid(True, alpha=0.2, linestyle='--', color='#808080')
        
        # Spine colors
        for spine in ax.spines.values():
            spine.set_color('#404040')
        
        # Labels
        ax.set_ylabel(f'Channel {channel}',
                     fontsize=12,
                     color='#FFFFFF',
                     fontweight='bold',
                     labelpad=10)
        
        # Channel indicator
        ax.text(0.02, 0.95, channel,
                transform=ax.transAxes,
                color=color,
                fontsize=14,
                fontweight='bold',
                bbox=dict(facecolor='#2C2C2C',
                         edgecolor=color,
                         alpha=0.7,
                         pad=5))
        
        # Legend styling
        if ax.get_legend():
            ax.legend(loc="upper right",
                     facecolor='#2C2C2C',
                     edgecolor='#404040',
                     fontsize=10,
                     framealpha=0.8)
        
        # X-axis label for bottom subplot
        if channel == self.selected_channels[-1]:
            ax.set_xlabel('Time (seconds)',
                         fontsize=12,
                         color='#FFFFFF',
                         fontweight='bold',
                         labelpad=10)
            
        # Adjust margins
        ax.margins(x=0.02)

    def animate(self, interval: int = 40):
        """Start the animation with the aesthetic settings."""
        anim = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=range(len(self.df) // self.batch_size),
            interval=interval
        )
        
        # No need for tight_layout as we're using GridSpec
        plt.show()

# Usage example:
if __name__ == "__main__":
    channels_to_plot = ['F7', 'F3', 'P8', 'P7']
    visualizer = AestheticEEGVisualizer(
        data_path='./average_df.csv',
        channels=channels_to_plot,
        sampling_freq=128,
        batch_size=5
    )
    visualizer.animate()