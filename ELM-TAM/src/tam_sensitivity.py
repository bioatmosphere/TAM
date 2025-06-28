"""
Visualizations of the sensitivity of ELM-TAM model parameters.

This script analyses the pickle file from the OLMT storing
the sensitivity analysis results based on surrogate models.

This analysis originates from producing publication-quality figures for the ELM-TAM paper.

In detail, this script:

1. visualizes the surrogate model
2. plots the main sensitivity indices
3. plots second-order sensitivity indices
4. plots total sensitivity indices
"""

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split


# Site configurations
SITES = {
    'US-MOz': {
        'file_name': '20250624_US-MOz_ICB20TRCNPRDCTCBC.pkl'
    },
    'US-Ho1': {
        'file_name': '20250604_US-Ho1_ICB20TRCNPRDCTCBC.pkl'
    },
    # Add other sites as needed
}

# Global configuration
CONFIG = {
    # External dependencies
    'paths': {
        'olmt': '/Users/6lw/Desktop/2_models/OLMT',
        'data_sites': 'data/parameter/sites/'
    },
    
    # Current site selection
    'current_site': 'US-MOz',
    
    # Plot settings
    'plotting': {
        'figure_size': (10, 6),
        'dpi': 300,
        'output_format': 'pdf'
    }
}

# Dynamic site configuration accessor
def get_site_config(site_name=None):
    """Get configuration for specified site or current site."""
    site = site_name or CONFIG['current_site']
    if site not in SITES:
        raise ValueError(f"Site '{site}' not found in SITES configuration")
    
    return {
        'site_name': site,
        'file_name': SITES[site]['file_name'],
        'data_directory': CONFIG['paths']['data_sites'],
        **CONFIG['plotting']
    }

# Add OLMT path
sys.path.append(CONFIG['paths']['olmt'])
import model_ELM



def plot_surrogate_comparison(yval, ypredict_val, qoi, v, output_dir):
    """
    Plot comparison between original and surrogate model predictions.
    
    At its core, this function performs an evaluation of the surrogate model
    by comparing its predictions against the original model outputs.

    Essential data are pulled from pickle files.

    Args:
        yval: Original model values
        ypredict_val: Surrogate model predictions
        qoi: Quantity of interest index
        v: Variable name
        output_dir: Directory to save plots
    """
    # Set publication-quality plotting parameters
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.linewidth': 1.2,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': False  # Set to True if LaTeX is available
    })

    # Calculate additional metrics
    yval_col = yval.astype(float)[:, qoi]
    ypred_col = ypredict_val.astype(float)[:, qoi]
    
    rsq = np.corrcoef(yval_col, ypred_col)[0, 1]**2
    rmse = np.sqrt(np.mean((yval_col - ypred_col)**2))
    mae = np.mean(np.abs(yval_col - ypred_col))
    bias = np.mean(ypred_col - yval_col)
    
    # Create figure with publication-quality styling
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate density for color mapping
    from scipy.stats import gaussian_kde
    if len(yval_col) > 10:  # Only use density coloring if enough points
        xy = np.vstack([yval_col, ypred_col])
        density = gaussian_kde(xy)(xy)
        scatter = ax.scatter(yval_col, ypred_col,
                           c=density, cmap='viridis',
                           s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Point Density', fontsize=12)
    else:
        ax.scatter(yval_col, ypred_col,
                  c='#2E86AB', s=60, alpha=0.7, 
                  edgecolors='black', linewidth=0.5)
    
    # Calculate axis limits with padding
    all_vals = np.concatenate([yval_col, ypred_col])
    val_range = np.max(all_vals) - np.min(all_vals)
    padding = val_range * 0.05
    lims = [np.min(all_vals) - padding, np.max(all_vals) + padding]
    
    # Add perfect fit line
    ax.plot(lims, lims, 'k--', alpha=0.8, linewidth=2, label='1:1 line')
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(yval_col, ypred_col)
    reg_line = slope * np.array(lims) + intercept
    ax.plot(lims, reg_line, 'r-', alpha=0.8, linewidth=2, 
            label=f'Regression (slope={slope:.3f})')
    
    # Set axis limits
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    
    # Customize plot appearance
    ax.set_xlabel('ELM-TAM Values', fontweight='bold')
    ax.set_ylabel('Surrogate Values', fontweight='bold')
    ax.set_title(f'{v} - Surrogate Model Validation (QOI {qoi+1})', 
                fontweight='bold', pad=20)
    
    # Add statistics text box
    stats_text = f'R² = {rsq:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nBias = {bias:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
            verticalalignment='top', fontfamily='monospace', fontsize=11)
    
    # Customize grid
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Customize legend
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', direction='in', 
                   length=6, width=1.2, pad=8)
    ax.tick_params(axis='both', which='minor', direction='in', 
                   length=3, width=1)
    
    # Enable minor ticks
    ax.minorticks_on()
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot with high quality
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, f'surrogate_{v}_qoi{qoi+1}.pdf')
    # plt.savefig(output_file, 
    #             bbox_inches='tight',
    #             dpi=300,
    #             facecolor='white',
    #             edgecolor='none',
    #             format='pdf')
    
    # Also save as high-resolution PNG for presentations
    png_file = os.path.join(output_dir, f'surrogate_{v}_qoi{qoi+1}.png')
    plt.savefig(png_file, 
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none',
                format='png')
    
    print(f"QOI {qoi+1}: R² = {rsq:.3f}, RMSE = {rmse:.3f}")
    plt.close(fig)

def get_plot_styles(n_parameters):
    """
    Generate distinct colors and patterns for parameter visualization.
    
    Args:
        n_parameters: Number of parameters to generate styles for
    
    Returns:
        tuple: (colors list, patterns list)
    """
    # Create color palette
    if n_parameters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        # For more parameters, use a different colormap
        colors = plt.cm.viridis(np.linspace(0, 1, n_parameters))
    
    # Define hatch patterns
    hatches = [
        '/', '\\', '|', '-',    # Simple lines
        '+', 'x', '*',          # Crosses
        'o', 'O', '.',          # Dots and circles
        '//', '\\\\', '||', '--' # Double patterns
    ]
    
    # Ensure we have enough patterns
    while len(hatches) < n_parameters:
        hatches.extend(hatches)
    
    return colors[:n_parameters], hatches[:n_parameters]


def plot_stacked_sensitivity_bars(data, variable_name, output_dir, plot_type='main'):
    """
    Plot stacked bar chart for sensitivity indices.
    
    Args:
        data: Loaded data object with sensitivity information
        variable_name: Name of the variable to plot
        output_dir: Directory to save the plot
        plot_type: Type of plot ('main' or 'total')
    """
    # Create subdirectory for this plot type
    subdir = os.path.join(output_dir, plot_type)
    os.makedirs(subdir, exist_ok=True)
    
    # Select the appropriate sensitivity data
    if plot_type == 'main':
        sens_data = data.sens_main[variable_name]
        title = f'Main Sensitivity Indices for {variable_name}'
        filename = f'sens_main_{variable_name}.{CONFIG["plotting"]["output_format"]}'
    else:  # total
        sens_data = data.sens_tot[variable_name]
        title = f'Total Sensitivity Indices for {variable_name}'
        filename = f'sens_total_{variable_name}.{CONFIG["plotting"]["output_format"]}'
    
    fig, ax = plt.subplots(figsize=CONFIG['plotting']['figure_size'])
    
    nvar = sens_data.shape[1]
    x_pos = np.arange(nvar)
    colors, _ = get_plot_styles(data.nparms_ensemble)
    
    # Plot the stacked bars
    bottom = np.zeros(nvar)
    patches = []
    
    for p in range(data.nparms_ensemble):
        color = colors[p % len(colors)]
        bar = ax.bar(
            x_pos, 
            sens_data[p, :], 
            bottom=bottom, 
            color=color, 
            edgecolor='none'
        )
        bottom += sens_data[p, :]
        
        # Create a legend entry
        patches.append(
            mpatches.Patch(
                facecolor=color,
                edgecolor='none',
                label=data.ensemble_parms[p] + str(data.ensemble_pfts[p])
            )
        )
    
    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Year {i+1}' for i in range(nvar)], rotation=45)
    ax.set_ylabel('Sensitivity Index')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Place legend outside the plot
    ax.legend(
        handles=patches, 
        loc='upper left', 
        bbox_to_anchor=(1, 1), 
        title='Parameters'
    )
    
    # Save the plot
    plt.savefig(os.path.join(subdir, filename), 
                bbox_inches='tight',
                dpi=CONFIG['plotting']['dpi'])
    plt.close()


def plot_second_order_sensitivity(data, variable_name, output_dir):
    """
    Plot second-order sensitivity indices as heatmaps.
    
    Args:
        data: Loaded data object with sensitivity information
        variable_name: Name of the variable to plot
        output_dir: Directory to save the plot
    """
    # Create subdirectory for second-order plots
    subdir = os.path.join(output_dir, 'second_order')
    os.makedirs(subdir, exist_ok=True)
    nvar = data.sens_main[variable_name].shape[1]
    fig, axes = plt.subplots(1, nvar, figsize=(5*nvar, 4))
    if nvar == 1:
        axes = [axes]  # Ensure axes is always a list
    
    fig.suptitle(f'Second-order Sensitivity Indices for {variable_name}', 
                 fontsize=14, y=1.05)
    
    param_names = data.ensemble_parms
    
    for n in range(nvar):
        im = axes[n].imshow(
            data.sens_2nd[variable_name][:, :, n],
            cmap='viridis'
        )
        
        # Add tick labels
        axes[n].set_xticks(np.arange(len(param_names)))
        axes[n].set_yticks(np.arange(len(param_names)))
        axes[n].set_xticklabels(param_names, rotation=45, ha='right')
        axes[n].set_yticklabels(param_names)
        axes[n].tick_params(axis='both', which='major', labelsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[n])
        
        # Customize subplot
        axes[n].set_title(f'Year {n+1}')
        axes[n].set_xlabel('Parameter Index')
        axes[n].set_ylabel('Parameter Index')
        axes[n].grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    filename = f'sens_2nd_order_{variable_name}.{CONFIG["plotting"]["output_format"]}'
    plt.savefig(os.path.join(subdir, filename), 
                bbox_inches='tight',
                dpi=CONFIG['plotting']['dpi'])
    plt.close()


def plot_surrogate_model_validation(data, variable_name, output_dir):
    """
    Plot surrogate model validation comparing predictions vs actual values.
    
    Args:
        data: Loaded data object with surrogate model information
        variable_name: Name of the variable to plot
        output_dir: Directory to save the plot
    """
    nqoi = data.output[variable_name].shape[0]
    
    # Extract outputs and samples 
    y = data.output[variable_name].transpose()
    p = data.samples.transpose()
    
    # Split data into training and validation sets
    ptrain, pval, ytrain, yval = train_test_split(p, y, test_size=0.2, random_state=42)
    
    emulator = data.surrogate[variable_name]
    yscaler = data.yscaler[variable_name]
    pscaler = data.pscaler[variable_name]
    pval_norm = pscaler.transform(pval)
    ypredict_val = yscaler.inverse_transform(emulator.predict(pval_norm))
    
    # Create surrogate validation plots
    surrogate_output_dir = os.path.join(output_dir, 'surrogate')
    for qoi in range(nqoi):
        plot_surrogate_comparison(yval, ypredict_val, qoi, variable_name, surrogate_output_dir)


def load_sensitivity_data(site_name, file_name, data_directory):
    """
    Load sensitivity analysis data from pickle file.
    
    Args:
        site_name: Name of the site
        file_name: Name of the pickle file
        data_directory: Directory containing the data
        
    Returns:
        Loaded data object
    """
    file_path = os.path.join(data_directory, site_name, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def process_sensitivity_analysis(data, output_dir):
    """
    Process all sensitivity analysis visualizations for loaded data.
    
    Args:
        data: Loaded sensitivity analysis data
        output_dir: Base directory for output files
    """
    myvars = data.postproc_vars
    
    for v in myvars:
        if v != 'taxis':
            print(f"Processing variable: {v}")
            
            # Plot main sensitivity indices
            plot_stacked_sensitivity_bars(data, v, output_dir, plot_type='main')
            
            # Plot second-order sensitivity indices
            plot_second_order_sensitivity(data, v, output_dir)
            
            # Plot total sensitivity indices
            plot_stacked_sensitivity_bars(data, v, output_dir, plot_type='total')
            
            # Plot surrogate model validation
            plot_surrogate_model_validation(data, v, output_dir)



def main():
    """
    Main function to execute sensitivity analysis visualization.
    """
    try:
        # Load sensitivity analysis data
        site_config = get_site_config()
        print(f"Loading data for site: {site_config['site_name']}")
        loaded_data = load_sensitivity_data(
            site_config['site_name'], 
            site_config['file_name'], 
            site_config['data_directory']
        )
        
        # Set up output directory
        output_dir = os.path.join(site_config['data_directory'], site_config['site_name'])
        
        # Process all sensitivity analysis visualizations
        print("Processing sensitivity analysis visualizations...")
        process_sensitivity_analysis(loaded_data, output_dir)
        
        print("Analysis complete! Check output directory for plots.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the file path and ensure the data file exists.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
