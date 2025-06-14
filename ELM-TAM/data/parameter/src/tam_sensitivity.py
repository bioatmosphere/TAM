"""
Visualizations of the sensitivity of ELM-TAM model parameters.

This script analyses the pickle file from the OLMT storing
the sensitivity analysis results based on surrogate models.

This analysis originates from producing publication-quality figures for the ELM-TAM paper.

In detail, this script:

1. visualizes the surrogate model

2. plots the main sensitivity indices

"""

path2OLMT = '/Users/6lw/Desktop/2_models/OLMT'
import sys
sys.path.append(path2OLMT)
import model_ELM

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split



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

    # Calculate R-squared
    rsq = np.corrcoef(yval.astype(float)[:,qoi], 
                        ypredict_val.astype(float)[:,qoi])[0,1]**2
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data points
    ax.scatter(yval.astype(float)[:,qoi],
                ypredict_val.astype(float)[:,qoi],
                c='red',
                alpha=0.6,
                s=50)
    
    # Add perfect fit line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
    
    # Customize plot
    ax.set_title(f'Model Comparison (R² = {rsq:.3f})', pad=20, fontsize=14)
    ax.set_xlabel('Original Model Output', fontsize=12)
    ax.set_ylabel('Surrogate Model Prediction', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'surrogate_{v}_qoi{qoi}.pdf')
    plt.savefig(output_file, 
                bbox_inches='tight',
                dpi=300,
                facecolor='white')
    
    print(f"QOI {qoi}: R² = {rsq:.3f}")
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



#[US-Ho1, 20250604_US-Ho1_ICB20TRCNPRDCTCBC.pkl]
#[US-MOz, 20250611_US-MOz_ICB20TRCNPRDCTCBC.pkl]
# load the pickle file
site_name = 'US-MOz' # US-Ho1
file_name = '20250612_US-MOz_ICB20TRCNPRDCTCBC.pkl'

directory = '../data/sites/'
file_path = os.path.join(directory, site_name, file_name)
with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

#variables to plot
myvars = loaded_data.postproc_vars
for v in myvars:
    if v != 'taxis':
        # --------------------------
        # Plot main(1st order) sensitivity indices
        # --------------------------
        fig, ax = plt.subplots(figsize=(10, 6))  # Larger figure for better visualization
        
        # sens_main: dictionary with sensitivity indices for each variable
        # each key is a variable, and the value is a 2D array with shape (nparms_ensemble, nyears)
        nvar = loaded_data.sens_main[v].shape[1]
        x_pos = np.arange(nvar)
        
        # Define distinct colors and patterns
        # colors = plt.cm.tab20.colors  # Use a colormap for distinct colors
        # hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']  # Patterns
        # Use in your plotting code
        colors, hatches = get_plot_styles(loaded_data.nparms_ensemble)

        # Plot the stacked bars
        bottom = np.zeros(nvar)
        patches = []  # Store legend handles
        
        for p in range(loaded_data.nparms_ensemble):
            color = colors[p % len(colors)]
            #hatch = hatches[p % len(hatches)]
            bar = ax.bar(
                x_pos, 
                loaded_data.sens_main[v][p, :], 
                bottom=bottom, 
                color=color, 
                #hatch=hatch, 
                edgecolor='black'
            )
            bottom += loaded_data.sens_main[v][p, :]
            
            # Create a legend entry
            patches.append(
                mpatches.Patch(
                    facecolor=color,
                    #hatch=hatch,
                    edgecolor='black',
                    label=loaded_data.ensemble_parms[p] + str(loaded_data.ensemble_pfts[p])
                )
            )
            
        # Adjust the axis and labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Year {i+1}' for i in range(nvar)], rotation=45)
        ax.set_ylabel('Sensitivity Index')
        ax.set_title(f'Main Sensitivity Indices for {v}')
        
        # Set y-axis limits from 0 to 1
        ax.set_ylim(0, 1)

        # Add horizontal gridlines for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Place legend outside the plot
        ax.legend(
            handles=patches, 
            loc='upper left', 
            bbox_to_anchor=(1, 1), 
            title='Parameters'
        )
        
        # Save the plot first
        output_dir = os.path.join(directory, site_name)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/sens_main_{v}.pdf', 
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

        # # --------------------------
        # # Plot 2nd sensitivity indices
        # # --------------------------
        # # Plot total sensitivity indices
        # fig, ax = plt.subplots(figsize=(10, 6))  # Larger figure for better visualization
        
        # # sens_main: dictionary with sensitivity indices for each variable
        # # each key is a variable, and the value is a 2D array with shape (nparms_ensemble, nyears)
        # nvar = loaded_data.sens_main[v].shape[1]
        # x_pos = np.arange(nvar)
        
        # # Define distinct colors and patterns
        # # colors = plt.cm.tab20.colors  # Use a colormap for distinct colors
        # # hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']  # Patterns
        # # Use in your plotting code
        # colors, hatches = get_plot_styles(loaded_data.nparms_ensemble)

        # # Plot the stacked bars
        # bottom = np.zeros(nvar)
        # patches = []  # Store legend handles
        
        # for p in range(loaded_data.nparms_ensemble):
        #     color = colors[p % len(colors)]
        #     #hatch = hatches[p % len(hatches)]
        #     bar = ax.bar(
        #         x_pos, 
        #         loaded_data.sens_2nd[v][p, 0, :], 
        #         bottom=bottom, 
        #         color=color, 
        #         #hatch=hatch, 
        #         edgecolor='black'
        #     )
        #     bottom += loaded_data.sens_2nd[v][p, 0, :]
            
        #     # Create a legend entry
        #     patches.append(
        #         mpatches.Patch(
        #             facecolor=color,
        #             #hatch=hatch,
        #             edgecolor='black',
        #             label=loaded_data.ensemble_parms[p] + str(loaded_data.ensemble_pfts[p])
        #         )
        #     )
            
        # # Adjust the axis and labels
        # ax.set_xticks(x_pos)
        # ax.set_xticklabels([f'Year {i+1}' for i in range(nvar)], rotation=45)
        # ax.set_ylabel('Sensitivity Index')
        # ax.set_title(f'2nd Order Sensitivity Indices for {v}')
        
        # # Set y-axis limits from 0 to 1
        # ax.set_ylim(0, 1)

        # # Add horizontal gridlines for better readability
        # ax.grid(axis='y', linestyle='--', alpha=0.3)

        # # Place legend outside the plot
        # ax.legend(
        #     handles=patches, 
        #     loc='upper left', 
        #     bbox_to_anchor=(1, 1), 
        #     title='Parameters'
        # )
        
        # # Save the plot first
        # output_dir = os.path.join(directory, site_name)
        # os.makedirs(output_dir, exist_ok=True)
        # plt.savefig(f'{output_dir}/sens_2nd_{v}.pdf', 
        #             bbox_inches='tight',
        #             dpi=300)
       
        # plt.close()


        # --------------------------
        # Plot TOTAL sensitivity indices
        # --------------------------
        # Plot total sensitivity indices
        fig, ax = plt.subplots(figsize=(10, 6))  # Larger figure for better visualization
        
        # sens_main: dictionary with sensitivity indices for each variable
        # each key is a variable, and the value is a 2D array with shape (nparms_ensemble, nyears)
        nvar = loaded_data.sens_main[v].shape[1]
        x_pos = np.arange(nvar)
        
        # Define distinct colors and patterns
        # colors = plt.cm.tab20.colors  # Use a colormap for distinct colors
        # hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']  # Patterns
        # Use in your plotting code
        colors, hatches = get_plot_styles(loaded_data.nparms_ensemble)

        # Plot the stacked bars
        bottom = np.zeros(nvar)
        patches = []  # Store legend handles
        
        for p in range(loaded_data.nparms_ensemble):
            color = colors[p % len(colors)]
            #hatch = hatches[p % len(hatches)]
            bar = ax.bar(
                x_pos, 
                loaded_data.sens_tot[v][p, :], 
                bottom=bottom, 
                color=color, 
                #hatch=hatch, 
                edgecolor='black'
            )
            bottom += loaded_data.sens_tot[v][p, :]
            
            # Create a legend entry
            patches.append(
                mpatches.Patch(
                    facecolor=color,
                    #hatch=hatch,
                    edgecolor='black',
                    label=loaded_data.ensemble_parms[p] + str(loaded_data.ensemble_pfts[p])
                )
            )
            
        # Adjust the axis and labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Year {i+1}' for i in range(nvar)], rotation=45)
        ax.set_ylabel('Sensitivity Index')
        ax.set_title(f'Total Sensitivity Indices for {v}')
        
        # Set y-axis limits from 0 to 1
        ax.set_ylim(0, 1)

        # Add horizontal gridlines for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Place legend outside the plot
        ax.legend(
            handles=patches, 
            loc='upper left', 
            bbox_to_anchor=(1, 1), 
            title='Parameters'
        )
        
        # Save the plot first
        output_dir = os.path.join(directory, site_name)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/sens_total_{v}.pdf', 
                    bbox_inches='tight',
                    dpi=300)
       
        plt.close()

    # --------------------------
    # Plot the surrogate model
    # --------------------------
    nqoi = loaded_data.output[v].shape[0]

    # Extract outputs and samples 
    y = loaded_data.output[v].transpose()
    p = loaded_data.samples.transpose()
    # Split data into training and validation sets
    ptrain, pval, ytrain, yval = train_test_split(p, y, test_size=0.2, random_state=42)

    emulator = loaded_data.surrogate[v]
    yscaler = loaded_data.yscaler[v]
    pscaler = loaded_data.pscaler[v]
    pval_norm   = pscaler.transform(pval)
    ypredict_val   = yscaler.inverse_transform(emulator.predict(pval_norm))

    
    # Use the function in the main loop
    output_dir = os.path.join(directory, site_name, 'surrogate')
    for qoi in range(nqoi):
        plot_surrogate_comparison(yval, ypredict_val, qoi, v, output_dir)
