"""
Cross-site sensitivity analysis comparison for ELM-TAM.

This script compares parameter sensitivities across multiple sites,
focusing on the combined GPP+ER carbon flux sensitivity.

Key visualizations:
1. Heatmap: Parameter × Site for main and total sensitivity
2. Grouped bar chart: Parameter ranking across sites
3. Site comparison panels: Direct comparison of parameter importance
"""

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Site configurations (ordered by PFT number)
SITES = {
    'US-Ho1': {
        'file_name': '20251014_US-Ho1_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'US-Ho1 (Hardwood)',
        'pft_number': 1,
        'pft_label': 'PFT 1',
        'color': '#A23B72'
    },
    'FI-Hyy': {
        'file_name': '20251007_FI-Hyy_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'FI-Hyy (Boreal Pine)',
        'pft_number': 2,
        'pft_label': 'PFT 2',
        'color': '#27AE60'
    },
    'RU-SkP': {
        'file_name': '20251019_RU-SkP_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'RU-SkP (Boreal Steppe)',
        'pft_number': 3,
        'pft_label': 'PFT 3',
        'color': '#E74C3C'
    },
    'BR-Sa1': {
        'file_name': '20251014_BR-Sa1_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'BR-Sa1 (Amazon Tropical)',
        'pft_number': 4,
        'pft_label': 'PFT 4',
        'color': '#16A085'
    },
    'PA-SPn': {
        'file_name': '20251017_PA-SPn_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'PA-SPn (Tropical Lowland)',
        'pft_number': 6,
        'pft_label': 'PFT 6',
        'color': '#F39C12'
    },
    'US-MOz': {
        'file_name': '20250910_US-MOz_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'US-MOz (Oak-Hickory)',
        'pft_number': 7,
        'pft_label': 'PFT 7',
        'color': '#2E86AB'
    },
    'CA-Oas': {
        'file_name': '20251015_CA-Oas_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'CA-Oas (Boreal Aspen)',
        'pft_number': 8,
        'pft_label': 'PFT 8',
        'color': '#9B59B6'
    },
    'ES-LJu': {
        'file_name': '20251022_ES-LJu_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'ES-LJu (Mediterranean Shrubland)',
        'pft_number': 9,
        'pft_label': 'PFT 9',
        'color': '#D35400'
    },
    'US-SRC': {
        'file_name': '20251022_US-SRC_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'US-SRC (Willow Bioenergy)',
        'pft_number': 10,
        'pft_label': 'PFT 10',
        'color': '#34495E'
    },
    'RU-Cok': {
        'file_name': '20251021_RU-Cok_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'RU-Cok (Arctic Tundra)',
        'pft_number': 11,
        'pft_label': 'PFT 11',
        'color': '#1ABC9C'
    },
    'US-Atq': {
        'file_name': '20251020_US-Atq_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'US-Atq (Alaskan Tundra)',
        'pft_number': 12,
        'pft_label': 'PFT 12',
        'color': '#E67E22'
    },
    'US-Var': {
        'file_name': '20250929_US-Var_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'US-Var (Mediterranean Grassland)',
        'pft_number': 13,
        'pft_label': 'PFT 13',
        'color': '#C0392B'
    },
    'AU-DaP': {
        'file_name': '20251020_AU-DaP_ICB20TRCNPRDCTCBC.pkl',
        'full_name': 'AU-DaP (Eucalyptus Savanna)',
        'pft_number': 14,
        'pft_label': 'PFT 14',
        'color': '#8E44AD'
    },
}

# Global configuration
CONFIG = {
    'paths': {
        'olmt': '/Users/6lw/Desktop/2_models/OLMT',
        'data_sites': '/Users/6lw/Desktop/3_vegetation/root/Root-Complexity/ELM-TAM/data/parameter/sites/'
    },
    'output_dir': '/Users/6lw/Desktop/3_vegetation/root/Root-Complexity/ELM-TAM/data/parameter/cross_site',
    'plotting': {
        'figure_size': (16, 8),
        'dpi': 300,
        'output_format': 'png'
    }
}

# Add OLMT path
sys.path.append(CONFIG['paths']['olmt'])


class RobustUnpickler(pickle.Unpickler):
    """Robust unpickler for handling compatibility issues."""

    def find_class(self, module, name):
        # Handle numpy random state compatibility
        if 'numpy.random' in module and 'MT19937' in name:
            try:
                from numpy.random import MT19937
                return MT19937
            except (ImportError, AttributeError):
                # Create a more compatible dummy class
                class DummyMT19937:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __setstate__(self, state):
                        pass
                    def __reduce__(self):
                        return (self.__class__, ())
                return DummyMT19937

        # Handle missing model_ELM module
        if module.startswith('model_ELM'):
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)
            return DummyClass

        # Handle sklearn version compatibility
        if module.startswith('sklearn') and 'MLPRegressor' in name:
            try:
                from sklearn.neural_network import MLPRegressor
                return MLPRegressor
            except ImportError:
                class DummyMLPRegressor:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __setstate__(self, state):
                        if isinstance(state, dict):
                            self.__dict__.update(state)
                    def predict(self, X):
                        return np.zeros((X.shape[0], 1))
                return DummyMLPRegressor

        # Handle other missing modules gracefully
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            class GenericObject:
                def __init__(self, *args, **kwargs):
                    pass
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)
            return GenericObject


def load_site_data(site_name, file_name, data_directory):
    """
    Load sensitivity analysis data for a site.

    Args:
        site_name: Name of the site
        file_name: Name of the pickle file
        data_directory: Directory containing the data

    Returns:
        Loaded data object or None if loading fails
    """
    file_path = os.path.join(data_directory, site_name, file_name)

    if not os.path.exists(file_path):
        print(f"  WARNING: Data file not found: {file_path}")
        return None

    try:
        # Try to create necessary dummy modules
        import types
        dummy_modules = ['model_ELM', 'model_ELM.main']
        for mod_name in dummy_modules:
            if mod_name not in sys.modules:
                module = types.ModuleType(mod_name)
                class DummyELMcase:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __setstate__(self, state):
                        if isinstance(state, dict):
                            self.__dict__.update(state)
                module.ELMcase = DummyELMcase
                sys.modules[mod_name] = module

        # Try standard pickle first with warnings suppressed
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

        print(f"  Successfully loaded: {site_name}")
        return data

    except Exception as e:
        # If standard pickle fails, try robust unpickler
        try:
            with open(file_path, 'rb') as f:
                unpickler = RobustUnpickler(f)
                data = unpickler.load()
            print(f"  Successfully loaded: {site_name} (with RobustUnpickler)")
            return data
        except Exception as e2:
            print(f"  ERROR loading {site_name}: {e}")
            print(f"    Secondary error: {e2}")
            return None


def aggregate_gpp_er_sensitivity(data):
    """
    Aggregate GPP and ER sensitivities.

    Args:
        data: Loaded sensitivity data object

    Returns:
        dict with 'main', 'total', and '2nd_order' aggregated sensitivities (mean across time)
    """
    # Get sensitivity data for GPP and ER
    sens_main_gpp = data.sens_main['GPP']
    sens_total_gpp = data.sens_tot['GPP']
    sens_main_er = data.sens_main['ER']
    sens_total_er = data.sens_tot['ER']

    # Get 2nd-order sensitivity data
    sens_2nd_gpp = data.sens_2nd['GPP']  # [n_params × n_params × n_years]
    sens_2nd_er = data.sens_2nd['ER']

    # Aggregate by averaging GPP and ER
    sens_main_combined = (sens_main_gpp + sens_main_er) / 2
    sens_total_combined = (sens_total_gpp + sens_total_er) / 2
    sens_2nd_combined = (sens_2nd_gpp + sens_2nd_er) / 2

    # Calculate mean across time
    mean_main = sens_main_combined.mean(axis=1)
    mean_total = sens_total_combined.mean(axis=1)
    mean_2nd = sens_2nd_combined.mean(axis=2)  # Average over time dimension

    return {
        'main': mean_main,
        'total': mean_total,
        '2nd_order': mean_2nd,
        'param_names': data.ensemble_parms
    }


def normalize_sensitivity_matrix(matrix, method='relative'):
    """
    Normalize sensitivity matrix for better cross-site comparison.

    Args:
        matrix: numpy array [n_sites × n_params]
        method: Normalization method
            - 'relative': Divide by sum within each site (recommended)
            - 'max': Divide by max within each site
            - 'zscore': Z-score standardization within each site
            - 'percentile': Convert to percentile ranks within each site
            - 'none': No normalization (raw values)

    Returns:
        Normalized matrix [n_sites × n_params]
    """
    normalized = np.zeros_like(matrix)
    n_sites = matrix.shape[0]

    for i in range(n_sites):
        if method == 'relative':
            # Divide by sum (relative importance)
            row_sum = matrix[i, :].sum()
            if row_sum > 0:
                normalized[i, :] = matrix[i, :] / row_sum
            else:
                normalized[i, :] = matrix[i, :]

        elif method == 'max':
            # Divide by maximum
            row_max = matrix[i, :].max()
            if row_max > 0:
                normalized[i, :] = matrix[i, :] / row_max
            else:
                normalized[i, :] = matrix[i, :]

        elif method == 'zscore':
            # Z-score standardization
            row_mean = matrix[i, :].mean()
            row_std = matrix[i, :].std()
            if row_std > 0:
                normalized[i, :] = (matrix[i, :] - row_mean) / row_std
            else:
                normalized[i, :] = matrix[i, :] - row_mean

        elif method == 'percentile':
            # Percentile ranking (0-1 scale)
            from scipy.stats import rankdata
            normalized[i, :] = rankdata(matrix[i, :], method='average') / len(matrix[i, :])

        elif method == 'none':
            # No normalization
            normalized[i, :] = matrix[i, :]

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def plot_cross_site_heatmap(site_data_dict, output_dir, normalize='none'):
    """
    Create heatmap showing parameter sensitivity across sites.

    Args:
        site_data_dict: Dictionary of {site_name: aggregated_sensitivity_data}
        output_dir: Directory to save the plot
        normalize: Normalization method ('none', 'relative', 'max', 'zscore', 'percentile')
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data and sort by PFT number
    site_names = sorted(site_data_dict.keys(), key=lambda s: SITES[s]['pft_number'])
    n_sites = len(site_names)

    # Get parameter names from first site
    param_names = site_data_dict[site_names[0]]['param_names']
    n_params = len(param_names)

    # Build matrix: rows = sites, columns = parameters
    main_matrix = np.zeros((n_sites, n_params))

    for i, site in enumerate(site_names):
        main_matrix[i, :] = site_data_dict[site]['main']

    # Apply normalization
    normalized_matrix = normalize_sensitivity_matrix(main_matrix, method=normalize)

    # Create figure with single heatmap (adjust height based on number of sites)
    fig_height = max(6, n_sites * 0.6)  # At least 6 inches, scale with number of sites
    fig, ax = plt.subplots(1, 1, figsize=(18, fig_height))

    # Determine colorbar label and value range based on normalization
    if normalize == 'relative':
        cbar_label = 'Relative Importance (fraction of total sensitivity)'
        vmin, vmax = 0, normalized_matrix.max()
        annotation_threshold = 0.05  # 5% of total
        title_suffix = ' (Normalized: Relative Importance)'
    elif normalize == 'max':
        cbar_label = 'Normalized Sensitivity (relative to site maximum)'
        vmin, vmax = 0, 1.0
        annotation_threshold = 0.05
        title_suffix = ' (Normalized: Relative to Max)'
    elif normalize == 'zscore':
        cbar_label = 'Standardized Sensitivity (Z-score)'
        vmin, vmax = normalized_matrix.min(), normalized_matrix.max()
        annotation_threshold = 0.5  # 0.5 std above mean
        title_suffix = ' (Normalized: Z-score)'
    elif normalize == 'percentile':
        cbar_label = 'Percentile Rank (0-1)'
        vmin, vmax = 0, 1.0
        annotation_threshold = 0.5  # 50th percentile
        title_suffix = ' (Normalized: Percentile)'
    else:  # 'none'
        cbar_label = 'Main Sensitivity Index'
        vmin, vmax = 0, normalized_matrix.max()
        annotation_threshold = 0.05
        title_suffix = ''

    # Main sensitivity heatmap
    im = ax.imshow(normalized_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                   vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(n_sites))
    # Adjust font size based on number of sites
    yaxis_fontsize = max(8, 12 - n_sites // 5)  # Smaller font for more sites
    ax.set_yticklabels([SITES[s]['pft_label'] for s in site_names], fontsize=yaxis_fontsize)
    ax.set_xticks(np.arange(n_params))
    ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel('Parameter', fontsize=14, fontweight='bold')
    ax.set_ylabel('PFT', fontsize=14, fontweight='bold')
    ax.set_title(f'Combined GPP+ER - Main Sensitivity Index Across PFTs{title_suffix}',
                 fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=13)

    # Add values on heatmap
    for i in range(n_sites):
        for j in range(n_params):
            if normalized_matrix[i, j] > annotation_threshold:
                # Format based on normalization method
                if normalize == 'zscore':
                    value_text = f'{normalized_matrix[i, j]:.2f}'
                else:
                    value_text = f'{normalized_matrix[i, j]:.2f}'

                # Determine text color based on cell brightness
                if normalize == 'zscore':
                    # For z-score, use middle value for color threshold
                    threshold = (vmin + vmax) / 2
                else:
                    threshold = 0.5 * vmax

                text = ax.text(j, i, value_text,
                              ha="center", va="center",
                              color="white" if normalized_matrix[i, j] > threshold else "black",
                              fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save with normalization method in filename
    if normalize != 'none':
        filename = f'cross_site_heatmap_GPP_ER_{normalize}.png'
    else:
        filename = 'cross_site_heatmap_GPP_ER.png'

    plt.savefig(os.path.join(output_dir, filename),
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_cross_site_interactions_heatmap(site_data_dict, output_dir, n_top=15, normalize='none'):
    """
    Create heatmap showing top parameter interactions across sites.

    Args:
        site_data_dict: Dictionary of {site_name: aggregated_sensitivity_data}
        output_dir: Directory to save the plot
        n_top: Number of top interactions to show (default: 15)
        normalize: Normalization method ('none', 'relative', 'max', 'percentile')
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data and sort by PFT number
    site_names = sorted(site_data_dict.keys(), key=lambda s: SITES[s]['pft_number'])
    n_sites = len(site_names)
    param_names = site_data_dict[site_names[0]]['param_names']
    n_params = len(param_names)

    # Find top N interactions across all sites
    # Build list of all unique parameter pairs and their maximum sensitivity
    interaction_scores = {}

    for i in range(n_params):
        for j in range(i+1, n_params):  # Only upper triangle (avoid duplicates)
            interaction_name = f'{param_names[i]}×{param_names[j]}'

            # Find maximum sensitivity for this interaction across all sites
            max_sens = 0
            for site in site_names:
                sens_2nd = site_data_dict[site]['2nd_order']
                max_sens = max(max_sens, sens_2nd[i, j])

            interaction_scores[interaction_name] = {
                'i': i,
                'j': j,
                'max_sens': max_sens
            }

    # Sort interactions by maximum sensitivity
    sorted_interactions = sorted(interaction_scores.items(),
                                key=lambda x: x[1]['max_sens'],
                                reverse=True)

    # Take top N
    top_interactions = sorted_interactions[:n_top]
    top_interaction_names = [inter[0] for inter in top_interactions]

    # Build matrix: rows = sites, columns = top interactions
    interactions_matrix = np.zeros((n_sites, n_top))

    for site_idx, site in enumerate(site_names):
        sens_2nd = site_data_dict[site]['2nd_order']
        for inter_idx, (inter_name, inter_info) in enumerate(top_interactions):
            i, j = inter_info['i'], inter_info['j']
            interactions_matrix[site_idx, inter_idx] = sens_2nd[i, j]

    # Apply normalization
    normalized_matrix = normalize_sensitivity_matrix(interactions_matrix, method=normalize)

    # Create figure (adjust height based on number of sites)
    fig_height = max(6, n_sites * 0.6)  # At least 6 inches, scale with number of sites
    fig, ax = plt.subplots(1, 1, figsize=(20, fig_height))

    # Determine colorbar label and value range based on normalization
    if normalize == 'relative':
        cbar_label = 'Relative Importance (fraction of total)'
        vmin, vmax = 0, normalized_matrix.max()
        annotation_threshold = 0.03
        title_suffix = ' (Normalized: Relative Importance)'
    elif normalize == 'max':
        cbar_label = 'Normalized Sensitivity (relative to site maximum)'
        vmin, vmax = 0, 1.0
        annotation_threshold = 0.05
        title_suffix = ' (Normalized: Relative to Max)'
    elif normalize == 'percentile':
        cbar_label = 'Percentile Rank (0-1)'
        vmin, vmax = 0, 1.0
        annotation_threshold = 0.5
        title_suffix = ' (Normalized: Percentile)'
    else:  # 'none'
        cbar_label = '2nd-Order Sensitivity Index'
        vmin, vmax = 0, normalized_matrix.max()
        annotation_threshold = 0.01
        title_suffix = ''

    # Create heatmap
    im = ax.imshow(normalized_matrix, aspect='auto', cmap='RdPu', interpolation='nearest',
                   vmin=vmin, vmax=vmax)

    ax.set_yticks(np.arange(n_sites))
    # Adjust font size based on number of sites
    yaxis_fontsize = max(8, 12 - n_sites // 5)  # Smaller font for more sites
    ax.set_yticklabels([SITES[s]['pft_label'] for s in site_names], fontsize=yaxis_fontsize)
    ax.set_xticks(np.arange(n_top))
    ax.set_xticklabels(top_interaction_names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Parameter Interaction', fontsize=14, fontweight='bold')
    ax.set_ylabel('PFT', fontsize=14, fontweight='bold')
    ax.set_title(f'Combined GPP+ER - Top {n_top} Parameter Interactions Across PFTs{title_suffix}',
                 fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=13)

    # Add values on heatmap
    for i in range(n_sites):
        for j in range(n_top):
            if normalized_matrix[i, j] > annotation_threshold:
                value_text = f'{normalized_matrix[i, j]:.2f}'

                # Determine text color
                threshold = 0.5 * vmax
                text = ax.text(j, i, value_text,
                              ha="center", va="center",
                              color="white" if normalized_matrix[i, j] > threshold else "black",
                              fontsize=8, fontweight='bold')

    plt.tight_layout()

    # Save with normalization method in filename
    if normalize != 'none':
        filename = f'cross_site_interactions_heatmap_GPP_ER_{normalize}_top{n_top}.png'
    else:
        filename = f'cross_site_interactions_heatmap_GPP_ER_top{n_top}.png'

    plt.savefig(os.path.join(output_dir, filename),
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_cross_site_grouped_bars(site_data_dict, output_dir):
    """
    Create grouped bar chart comparing parameter importance across sites.

    Args:
        site_data_dict: Dictionary of {site_name: aggregated_sensitivity_data}
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    site_names = list(site_data_dict.keys())
    n_sites = len(site_names)
    param_names = site_data_dict[site_names[0]]['param_names']
    n_params = len(param_names)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Set up bar positions
    x = np.arange(n_params)
    width = 0.8 / n_sites

    # Plot 1: Main sensitivity
    for i, site in enumerate(site_names):
        offset = (i - n_sites/2 + 0.5) * width
        values = site_data_dict[site]['main']
        ax1.bar(x + offset, values, width, label=SITES[site]['full_name'],
                color=SITES[site]['color'], alpha=0.8, edgecolor='black', linewidth=0.8)

    ax1.set_xlabel('Parameter', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Main Sensitivity Index', fontsize=13, fontweight='bold')
    ax1.set_title('Combined GPP+ER - Main Sensitivity Comparison',
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    ax1.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, None)

    # Plot 2: Total sensitivity
    for i, site in enumerate(site_names):
        offset = (i - n_sites/2 + 0.5) * width
        values = site_data_dict[site]['total']
        ax2.bar(x + offset, values, width, label=SITES[site]['full_name'],
                color=SITES[site]['color'], alpha=0.8, edgecolor='black', linewidth=0.8)

    ax2.set_xlabel('Parameter', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Total Sensitivity Index', fontsize=13, fontweight='bold')
    ax2.set_title('Combined GPP+ER - Total Sensitivity Comparison',
                 fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
    ax2.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_site_grouped_bars_GPP_ER.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✓ Saved: cross_site_grouped_bars_GPP_ER.png")


def plot_cross_site_ranking(site_data_dict, output_dir):
    """
    Create parameter ranking plot showing top parameters across sites.

    Args:
        site_data_dict: Dictionary of {site_name: aggregated_sensitivity_data}
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    site_names = list(site_data_dict.keys())
    n_sites = len(site_names)
    param_names = site_data_dict[site_names[0]]['param_names']

    # Create figure
    fig, axes = plt.subplots(1, n_sites, figsize=(7*n_sites, 8))
    if n_sites == 1:
        axes = [axes]

    for idx, site in enumerate(site_names):
        ax = axes[idx]

        # Get total sensitivity and sort
        total_sens = site_data_dict[site]['total']
        sorted_indices = np.argsort(total_sens)[::-1]

        # Plot horizontal bars
        y_pos = np.arange(len(param_names))
        ax.barh(y_pos, total_sens[sorted_indices],
                color=SITES[site]['color'], alpha=0.7,
                edgecolor='black', linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([param_names[i] for i in sorted_indices], fontsize=10)
        ax.set_xlabel('Total Sensitivity Index', fontsize=12, fontweight='bold')
        ax.set_title(SITES[site]['full_name'],
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()

        # Add value labels for top 5 parameters
        for i in range(min(5, len(param_names))):
            val = total_sens[sorted_indices[i]]
            if val > 0.01:
                ax.text(val, i, f' {val:.3f}',
                       va='center', fontsize=9, fontweight='bold')

    fig.suptitle('Combined GPP+ER - Parameter Ranking by Site',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_site_ranking_GPP_ER.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✓ Saved: cross_site_ranking_GPP_ER.png")


def plot_cross_site_radar(site_data_dict, output_dir, n_top=10):
    """
    Create radar/spider plot for top parameters across sites.

    Args:
        site_data_dict: Dictionary of {site_name: aggregated_sensitivity_data}
        output_dir: Directory to save the plot
        n_top: Number of top parameters to show (default: 10)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort by PFT number
    site_names = sorted(site_data_dict.keys(), key=lambda s: SITES[s]['pft_number'])
    param_names = site_data_dict[site_names[0]]['param_names']

    # Find top N parameters (by maximum total sensitivity across all sites)
    max_sens = np.zeros(len(param_names))
    for site in site_names:
        max_sens = np.maximum(max_sens, site_data_dict[site]['total'])

    top_indices = np.argsort(max_sens)[::-1][:n_top]
    top_param_names = [param_names[i] for i in top_indices]

    # Number of parameters
    num_vars = len(top_param_names)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))

    # Plot for each site
    for site in site_names:
        total_sens = site_data_dict[site]['total']
        values = [total_sens[i] for i in top_indices]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2.5,
                label=SITES[site]['pft_label'],
                color=SITES[site]['color'], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=SITES[site]['color'])

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_param_names, fontsize=10 if n_top > 10 else 11)
    ax.set_ylim(0, None)
    ax.set_ylabel('Total Sensitivity Index', fontsize=12, fontweight='bold')
    ax.set_title(f'Combined GPP+ER - Top {n_top} Parameters Across PFTs',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)

    plt.tight_layout()

    # Save with different filename based on n_top
    filename = f'cross_site_radar_GPP_ER_top{n_top}.png'
    plt.savefig(os.path.join(output_dir, filename),
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_cross_site_radar_all(site_data_dict, output_dir):
    """
    Create radar/spider plot for ALL parameters across sites.

    Args:
        site_data_dict: Dictionary of {site_name: aggregated_sensitivity_data}
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort by PFT number
    site_names = sorted(site_data_dict.keys(), key=lambda s: SITES[s]['pft_number'])
    param_names = site_data_dict[site_names[0]]['param_names']
    n_params = len(param_names)

    # Use all parameters (sorted by maximum total sensitivity)
    max_sens = np.zeros(n_params)
    for site in site_names:
        max_sens = np.maximum(max_sens, site_data_dict[site]['total'])

    # Sort all parameters by importance
    sorted_indices = np.argsort(max_sens)[::-1]
    sorted_param_names = [param_names[i] for i in sorted_indices]

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create figure with larger size for more parameters
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(projection='polar'))

    # Plot for each site
    for site in site_names:
        total_sens = site_data_dict[site]['total']
        values = [total_sens[i] for i in sorted_indices]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2.5,
                label=SITES[site]['pft_label'],
                color=SITES[site]['color'], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=SITES[site]['color'])

    # Set labels with smaller font for readability with 17 parameters
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sorted_param_names, fontsize=9)
    ax.set_ylim(0, None)
    ax.set_ylabel('Total Sensitivity Index', fontsize=12, fontweight='bold')
    ax.set_title(f'Combined GPP+ER - All {n_params} Parameters Across PFTs',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_site_radar_GPP_ER_all17.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✓ Saved: cross_site_radar_GPP_ER_all17.png")


def plot_publication_figure(site_data_dict, output_dir, normalize='relative', n_top_interactions=10):
    """
    Create comprehensive publication-ready figure with main effects, interactions, and radar plot.

    Args:
        site_data_dict: Dictionary of {site_name: aggregated_sensitivity_data}
        output_dir: Directory to save the plot
        normalize: Normalization method for heatmaps ('relative' recommended)
        n_top_interactions: Number of top interactions to show (default: 10)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data and sort by PFT number
    site_names = sorted(site_data_dict.keys(), key=lambda s: SITES[s]['pft_number'])
    n_sites = len(site_names)
    param_names = site_data_dict[site_names[0]]['param_names']
    n_params = len(param_names)

    # Create figure with 2x2 grid layout (adjust height for number of sites)
    # Left column: stacked heatmaps (main effects + interactions)
    # Right column: radar plot (spanning both rows)
    fig_height = max(12, n_sites * 1.2)  # At least 12 inches, scale with number of sites
    fig = plt.figure(figsize=(20, fig_height))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.3, 1], height_ratios=[1, 1],
                  hspace=0.35, wspace=0.3)

    # ============================================
    # Panel A: Main Effects Heatmap (Top Left)
    # ============================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Build and normalize main effects matrix
    main_matrix = np.zeros((n_sites, n_params))
    for i, site in enumerate(site_names):
        main_matrix[i, :] = site_data_dict[site]['main']

    normalized_main = normalize_sensitivity_matrix(main_matrix, method=normalize)

    # Plot heatmap
    im1 = ax1.imshow(normalized_main, aspect='auto', cmap='YlOrRd',
                     interpolation='nearest', vmin=0, vmax=normalized_main.max())

    ax1.set_yticks(np.arange(n_sites))
    # Adjust font size based on number of sites
    yaxis_fontsize = max(8, 11 - n_sites // 5)  # Smaller font for more sites
    ax1.set_yticklabels([SITES[s]['pft_label'] for s in site_names], fontsize=yaxis_fontsize)
    ax1.set_xticks(np.arange(n_params))
    ax1.set_xticklabels(param_names, rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel('Parameter', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PFT', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Main Effects', fontsize=14, fontweight='bold', pad=15, loc='left')

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    if normalize == 'relative':
        cbar1.set_label('Relative Importance', fontsize=10)
        annotation_threshold = 0.05
    else:
        cbar1.set_label('Sensitivity Index', fontsize=10)
        annotation_threshold = 0.05

    # Annotate cells
    for i in range(n_sites):
        for j in range(n_params):
            if normalized_main[i, j] > annotation_threshold:
                ax1.text(j, i, f'{normalized_main[i, j]:.2f}',
                        ha="center", va="center",
                        color="white" if normalized_main[i, j] > 0.5*normalized_main.max() else "black",
                        fontsize=7, fontweight='bold')

    # ============================================
    # Panel B: Parameter Interactions Heatmap (Bottom Left)
    # ============================================
    ax2 = fig.add_subplot(gs[1, 0])

    # Find top N interactions
    interaction_scores = {}
    for i in range(n_params):
        for j in range(i+1, n_params):
            interaction_name = f'{param_names[i]}×{param_names[j]}'
            max_sens = 0
            for site in site_names:
                sens_2nd = site_data_dict[site]['2nd_order']
                max_sens = max(max_sens, sens_2nd[i, j])
            interaction_scores[interaction_name] = {'i': i, 'j': j, 'max_sens': max_sens}

    sorted_interactions = sorted(interaction_scores.items(),
                                key=lambda x: x[1]['max_sens'], reverse=True)
    top_interactions = sorted_interactions[:n_top_interactions]
    top_interaction_names = [inter[0] for inter in top_interactions]

    # Build interactions matrix
    interactions_matrix = np.zeros((n_sites, n_top_interactions))
    for site_idx, site in enumerate(site_names):
        sens_2nd = site_data_dict[site]['2nd_order']
        for inter_idx, (inter_name, inter_info) in enumerate(top_interactions):
            i, j = inter_info['i'], inter_info['j']
            interactions_matrix[site_idx, inter_idx] = sens_2nd[i, j]

    normalized_inter = normalize_sensitivity_matrix(interactions_matrix, method=normalize)

    # Plot heatmap
    im2 = ax2.imshow(normalized_inter, aspect='auto', cmap='RdPu',
                     interpolation='nearest', vmin=0, vmax=normalized_inter.max())

    ax2.set_yticks(np.arange(n_sites))
    ax2.set_yticklabels([SITES[s]['pft_label'] for s in site_names], fontsize=yaxis_fontsize)
    ax2.set_xticks(np.arange(n_top_interactions))
    ax2.set_xticklabels(top_interaction_names, rotation=45, ha='right', fontsize=8)
    ax2.set_xlabel('Parameter Interaction', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PFT', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Top Parameter Interactions', fontsize=14, fontweight='bold', pad=15, loc='left')

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    if normalize == 'relative':
        cbar2.set_label('Relative Importance', fontsize=10)
    else:
        cbar2.set_label('2nd-Order Index', fontsize=10)

    # Annotate cells
    for i in range(n_sites):
        for j in range(n_top_interactions):
            if normalized_inter[i, j] > annotation_threshold:
                ax2.text(j, i, f'{normalized_inter[i, j]:.2f}',
                        ha="center", va="center",
                        color="white" if normalized_inter[i, j] > 0.5*normalized_inter.max() else "black",
                        fontsize=7, fontweight='bold')

    # ============================================
    # Panel C: Radar Plot (Holistic View - Right Side)
    # ============================================
    ax3 = fig.add_subplot(gs[:, 1], projection='polar')

    # Find top parameters for radar plot (top 8 for cleaner visualization)
    n_radar = 8
    max_sens = np.zeros(len(param_names))
    for site in site_names:
        max_sens = np.maximum(max_sens, site_data_dict[site]['total'])

    top_indices = np.argsort(max_sens)[::-1][:n_radar]
    top_param_names = [param_names[i] for i in top_indices]

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_radar, endpoint=False).tolist()
    angles += angles[:1]

    # Plot for each site
    for site in site_names:
        total_sens = site_data_dict[site]['total']
        values = [total_sens[i] for i in top_indices]
        values += values[:1]

        ax3.plot(angles, values, 'o-', linewidth=2.5,
                label=SITES[site]['pft_label'],
                color=SITES[site]['color'], markersize=7)
        ax3.fill(angles, values, alpha=0.15, color=SITES[site]['color'])

    # Set labels
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(top_param_names, fontsize=10)
    ax3.set_ylim(0, None)
    ax3.set_title('(C) Holistic View\n(Top 8 Parameters)',
                 fontsize=15, fontweight='bold', pad=25)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11,
              frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linewidth=1.0)

    # Overall figure title
    if normalize == 'relative':
        fig.suptitle('Cross-PFT Parameter Sensitivity Analysis (Relative Importance Normalization)',
                    fontsize=17, fontweight='bold', y=0.99)
    else:
        fig.suptitle('Cross-PFT Parameter Sensitivity Analysis',
                    fontsize=17, fontweight='bold', y=0.99)

    # Save figure
    if normalize != 'none':
        filename = f'publication_figure_GPP_ER_{normalize}.png'
    else:
        filename = 'publication_figure_GPP_ER.png'

    plt.savefig(os.path.join(output_dir, filename),
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def main():
    """
    Main function to execute cross-site sensitivity comparison.
    """
    print("="*70)
    print("Cross-Site Sensitivity Analysis Comparison")
    print("="*70)
    print()

    # Load data for all sites
    print("Loading site data...")
    site_data = {}

    for site_name, site_config in SITES.items():
        print(f"  Loading {site_name}...")
        data = load_site_data(
            site_name,
            site_config['file_name'],
            CONFIG['paths']['data_sites']
        )

        if data is not None:
            # Aggregate GPP+ER sensitivity
            agg_data = aggregate_gpp_er_sensitivity(data)
            site_data[site_name] = agg_data

    if len(site_data) == 0:
        print("\nERROR: No site data could be loaded!")
        return

    print(f"\nSuccessfully loaded {len(site_data)} sites: {list(site_data.keys())}")
    print()

    # Create output directory
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Generate cross-site comparison plots
    print("Generating cross-site comparison plots...")
    print()

    print("1. Creating heatmaps (Site × Parameter)...")
    print("   a) Raw values (no normalization)")
    plot_cross_site_heatmap(site_data, output_dir, normalize='none')

    print("   b) Relative importance normalization (RECOMMENDED)")
    plot_cross_site_heatmap(site_data, output_dir, normalize='relative')

    print("   c) Max normalization")
    plot_cross_site_heatmap(site_data, output_dir, normalize='max')

    print("   d) Percentile ranking")
    plot_cross_site_heatmap(site_data, output_dir, normalize='percentile')

    print("\n2. Creating interaction heatmaps (Top 15 parameter interactions)...")
    print("   a) Raw values (no normalization)")
    plot_cross_site_interactions_heatmap(site_data, output_dir, n_top=15, normalize='none')

    print("   b) Relative importance normalization (RECOMMENDED)")
    plot_cross_site_interactions_heatmap(site_data, output_dir, n_top=15, normalize='relative')

    print("   c) Max normalization")
    plot_cross_site_interactions_heatmap(site_data, output_dir, n_top=15, normalize='max')

    print("\n3. Creating radar/spider plot (Top 10 parameters)...")
    plot_cross_site_radar(site_data, output_dir, n_top=10)

    print("4. Creating radar/spider plot (All 17 parameters)...")
    plot_cross_site_radar_all(site_data, output_dir)

    print("\n5. Creating publication-ready comprehensive figure...")
    print("   a) With relative importance normalization (RECOMMENDED)")
    plot_publication_figure(site_data, output_dir, normalize='relative', n_top_interactions=10)

    print("   b) With raw values (no normalization)")
    plot_publication_figure(site_data, output_dir, normalize='none', n_top_interactions=10)

    print()
    print("="*70)
    print("Cross-site analysis complete!")
    print(f"All plots saved in: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
