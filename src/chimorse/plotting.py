"""
plotting.py
-----------
Visualization routines for energy surfaces, convergence diagnostics,
parity plots, and pruning analysis of the chiral Morse model.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

from .config import INTERACTION_CMAPS, PHI_COLORMAP
from .analysis import (extract_energy_minimums, infer_screw_direction, expand_by_screw_periodicity,
                      expand_chi_psi_by_screw_periodicity, circular_distance_deg,
                      extract_energy_comparison, build_energy_table, build_full_energy_table,
                      get_symm_chi)
from .fourier import compute_harmonic_rmse_grid, create_matrix_lsqt_2d
from .fitting import fit_alpha_values, prune_by_magnitude, print_modeling_information
from .dataio import load_data

# ======================================================================
# Axis / style helpers
# ======================================================================

def extract_ER_range(df, col):
    """Return [min, max] of df[col] padded by 1/6 of the range on each side, for plot axis limits."""
    min_var = df[col].min()
    max_var = df[col].max()
    window = (max_var - min_var)/6

    return [min_var - window, max_var + window]

# ----------------------------------------------------------------------

def _angle_label(x_axis):
    """Return the LaTeX axis label for 'chi', 'phi1', or 'psi'."""
    if x_axis == 'chi':
        return r'$\chi$ (°)'
    if x_axis == 'phi1':
        return r'$\varphi_1$ (°)'
    return r'$\psi$ (°)'

# ----------------------------------------------------------------------

def _select_sample_curve(df, row):
    """Select the E(r) curve from df matching row's chi/psi (and z if present);
       return the subset and a legend label."""
    group_cols = [c for c in ('chi', 'psi', 'z') if c in df.columns]

    mask = df[group_cols[0]] == row[group_cols[0]]
    for col in group_cols[1:]:
        mask &= df[col] == row[col]

    labels = {
        'chi': fr'$\chi$={row["chi"]:.0f}',
        'psi': fr'$\psi$={row["psi"]:.0f}',
    }
    if 'z' in group_cols:
        labels['z'] = fr'$\zeta$={row["z"]}'
    label = ', '.join(labels[col] for col in group_cols)

    return df.loc[mask], label

# ----------------------------------------------------------------------

def apply_style(ax, spine=False, grid=True, hide_top_right=True):
    """Apply common tick/grid/spine styling to an axis."""
    ax.tick_params(axis="both", direction="in")
    if grid:
        ax.grid(which='major', linestyle="--", alpha=0.5, zorder=0)
        ax.grid(which='minor', linestyle=':', linewidth=0.4, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
    if hide_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if spine:
        for side in ['left', 'right', 'top', 'bottom']:
            ax.spines[side].set_linewidth(0.5)
            ax.spines[side].set_color('gray')

# ----------------------------------------------------------------------

def apply_compact_style(ax, linewidth=0.6, tick_width=0.5):
    """apply_style with visible spines and a thin tick width."""
    apply_style(ax, spine=True, grid=False, hide_top_right=False)
    ax.tick_params(width=tick_width, length=2)

# ======================================================================
# Surface helpers
# ======================================================================

def make_surface(data, aggfunc):
    """Pivot data into a chi x psi grid via aggfunc, reindexed to a common grid
       with missing cells filled by nearest-neighbor interpolation."""
    grid = np.sort(data['chi'].unique())
    E = data.pivot_table(index='psi', columns='chi', values='e',
                        aggfunc=aggfunc).reindex(index=grid, columns=grid)
    return E.interpolate(axis=0,
                            method='nearest',
                        limit_direction='both')

# ----------------------------------------------------------------------

def draw_chi_psi_heatmap(ax, E, cmap, left_label=True, xlabel_pad=0):
    """Draw the E(chi,psi) heatmap on ax, with chi on x and psi on y."""
    im = ax.imshow(E.values, origin='lower', extent=[0, 360, 0, 360],
                   aspect='equal', cmap=cmap, alpha=0.8)

    ax.set_xlabel(r'$\chi$ (°)', labelpad=xlabel_pad)
    if left_label:
        ax.set_ylabel(r'$\psi$ (°)')

    return im

# ----------------------------------------------------------------------

def plot_line_cuts(ax, data, x_axis, other_axis, cut_values, cmap, orientation, tol=0.26):
    """Plot E vs x_axis as line cuts at fixed other_axis values,
       colored by cut value (horizontal or vertical); 
       return the active cuts and color norm."""
    norm = mpl.colors.Normalize(vmin=cut_values.min(), vmax=cut_values.max())

    active_cuts = []
    for val in cut_values:
        sel = data[circular_distance_deg(data[other_axis].values, val) <= tol]
        if not sel.empty:
            active_cuts.append(val)
            sel = sel.groupby(x_axis, as_index=False).agg({'e': 'mean', 'r': 'mean'}).sort_values(x_axis)

            if orientation == 'horizontal':
                ax.plot(sel[x_axis], sel['e'], color=cmap(norm(val)), lw=1.0)
            else:
                ax.plot(sel['e'], sel[x_axis], color=cmap(norm(val)), lw=1.0)

    return active_cuts, norm

# ======================================================================
# E(r) plots
# ======================================================================

def plot_ER_model_compare(Er_raw, potential_models, phi_vals=(0, 0), 
                          r_range=None, e_range=None, save_path=None):
    """Plot E(r) for the raw data and each fitted potential model at a fixed (phi1, phi2)."""
    names = ['Original Data'] + [m['name'] for m in potential_models]
    Er_raw = Er_raw.loc[(Er_raw['phi1'] == phi_vals[0]) & (Er_raw['phi2'] == phi_vals[1])]
    data_dfs = [Er_raw] + [m['df'] for m in potential_models]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    colors = ['black', 'orange', 'royalblue', 'red', 'purple']
    styles = ['-', '--', '-.', ':']

    for i, (Er, name) in enumerate(zip(data_dfs, names)):
        ax.plot(
            Er['r'], Er['e'], label=name,
            linestyle=styles[i % len(styles)], linewidth=1.5,
            color=colors[i % len(colors)]
        )
    if r_range:
        xlim = r_range
        ylim = e_range
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    ax.set_xlabel('Pair Distance (Å)')
    ax.set_ylabel('Binding Energy (eV)')

    phi1, phi2 = phi_vals
    ax.legend(fontsize=8)

    apply_style(ax, spine=True, grid=False)

    if save_path is not None:
        save_dir = Path(save_path).parent.parent.parent / "compare"
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "ER_model_comparison.pdf", bbox_inches='tight')

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------

def plot_ER(df, show_zoom_box=False, show_sample_curves=True, alpha_fit=False, save_path=None, vertical_r=None):
    """Plot E(r) scatter for all data (left) and for the per-orientation minima (right),
       with optional zoom box and sample curves."""
    alpha_dot = 0.002 if 'z' in df.columns else 0.02

    E_min_df = extract_energy_minimums(df, r_max=12)

    zoom_E_range = extract_ER_range(E_min_df, 'e')
    zoom_r_range = extract_ER_range(E_min_df, 'r')

    rect = Rectangle(
        (zoom_r_range[0], zoom_E_range[0]),
        zoom_r_range[1] - zoom_r_range[0],
        zoom_E_range[1] - zoom_E_range[0],
        linewidth=0.8, edgecolor='0.3',
        facecolor='0.9', alpha=0.4, linestyle='--'
    )

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    def scatter_min(ax, data):
        ax.scatter(
            data['r'], data['e'],
            c='#00000000', s=2, alpha=alpha_dot,
            zorder=1, edgecolors='none', rasterized=True
        )

    # ---------- Left ----------
    scatter_min(axes[0], df)
    if show_zoom_box:
        axes[0].add_patch(rect)
    if vertical_r is not None:
        axes[0].axvline(vertical_r, color='green', lw=0.8, ls='--', zorder=2)
        axes[0].text(
            0.95, 0.95, fr'$r={vertical_r}$',
            transform=axes[0].transAxes, ha='right', va='top',
            fontsize=9, color='green',
            bbox=dict(facecolor='white', alpha=1, edgecolor='green')
        )

    axes[0].set_xlabel('Pair Distance (Å)')
    axes[0].set_ylabel('Binding Energy (eV)')
    axes[0].set_xlim(extract_ER_range(df, 'r'))
    axes[0].set_ylim(-3, 25)
    for location in ['left', 'right', 'top', 'bottom']:
        axes[0].spines[location].set_linewidth(0.4)

    apply_style(axes[0], spine=True, grid=False, hide_top_right=False)
    axes[0].tick_params(width=0.4)

    # ---------- Right ----------
    scatter_min(axes[1], E_min_df)

    if show_sample_curves:
        E_min, E_max = df['e'].min(), E_min_df['e'].max()
        row_min = E_min_df.loc[E_min_df['e'] == E_min].iloc[0]
        row_max = E_min_df.loc[E_min_df['e'] == E_max].iloc[0]

        line_low, label_low = _select_sample_curve(df, row_min)
        line_high, label_high = _select_sample_curve(df, row_max)

        axes[1].plot(line_low['r'], line_low['e'], lw=1.0, color="#301592FF", label=label_low)
        axes[1].plot(line_high['r'], line_high['e'], lw=1.0, color="#D48C51FF", label=label_high)

        axes[1].legend(loc='lower right')

    axes[1].set_xlabel('Pair Distance (Å)')

    apply_style(axes[1], spine=True, grid=False, hide_top_right=False)
    axes[1].tick_params(width=0.4)

    if show_zoom_box:
        axes[1].set_xlim(*zoom_r_range)
        axes[1].set_ylim(*zoom_E_range)

    plt.tight_layout()
    if save_path is not None:
        plot_name = "ER_profiles_alpha.pdf" if alpha_fit else "ER_profiles.pdf"
        plt.savefig(save_path / plot_name, bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------

def plot_ER_orientations(df, orientations, color_by='chi', save_path=None):
    """Scatter the minimum-energy E(r) points colored by chi/phi1/phi2, 
       with optional highlighted orientation curves."""
    fig, ax = plt.subplots(figsize=(3.8, 2.8))

    E_min_df = extract_energy_minimums(df, r_max=12)

    cmap = plt.get_cmap(PHI_COLORMAP)
    norm = plt.Normalize(E_min_df[color_by].min(), E_min_df[color_by].max())

    if color_by == 'chi':
        title_cbar = r'$\chi$ (°)'
    elif color_by == 'phi1':
        title_cbar = r'$\varphi_1$ (°)'
    else:
        title_cbar = r'$\varphi_2$ (°)'

    alpha = 0.1 if color_by == 'phi2' else 0.05

    scatter = ax.scatter(
        E_min_df['r'], E_min_df['e'],
        c=E_min_df[color_by], cmap=cmap, s=10, alpha=alpha,
        norm=norm, edgecolors='none', rasterized=True
    )
    scatter_dummy = ax.scatter(
        E_min_df['r'], E_min_df['e'],
        c=E_min_df[color_by], cmap=cmap, s=0, alpha=1, norm=norm
    )

    cbar = plt.colorbar(scatter_dummy, ax=ax, pad=0.02)
    cbar.set_label(title_cbar)
    cbar.ax.tick_params(direction='in')

    if orientations:
        for chi_target, psi_target in orientations:
            line = df[(df['chi']==chi_target) & (df['psi']==psi_target)]
            label = fr'$\chi$={chi_target:.0f}, $\psi$={psi_target:.0f}'
            ax.plot(line['r'], line['e'], lw=1., label=label, color=cmap(norm(chi_target)))

        ax.legend(loc='lower right')

    ax.set_xlabel('Pair Distance (Å)')
    ax.set_ylabel('Binding Energy (eV)')

    ylim = extract_ER_range(E_min_df, 'e')
    xlim = extract_ER_range(E_min_df, 'r')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    apply_style(ax, spine=True, grid=False, hide_top_right=False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path / "ER_test.pdf", bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------

def plot_ER_raw_all(molecule, zero_zeta,
                    colors=INTERACTION_CMAPS,
                    choice_color=.58,
                    save_path=None):
    """Plot the min/max E(r) envelope of minimum-energy curves for each interaction type."""
    fig, ax = plt.subplots(figsize=(3.2, 2.8))

    styles = {'EP': '-', 'EA': '--', 'OP': '-.', 'OA': ':'}
    interaction_sort = {'EP': 2, 'EA': 3, 'OP': 1, 'OA': 0}

    for interaction in colors:
        color = plt.get_cmap(colors[interaction])(choice_color)
        df = load_data(molecule, interaction, zero_zeta)
        cols = [col for col in df.columns if col not in ['e', 'r']]
        df_min = df.loc[df.groupby(cols)['e'].idxmin()].reset_index(drop=True)
        bounds = df_min.groupby('r')['e'].agg(['min', 'max']).reset_index()
        bounds['min']  = gaussian_filter1d(bounds['min'], sigma=1)
        bounds['max'] = gaussian_filter1d(bounds['max'], sigma=1)
        current_z = interaction_sort[interaction]
        # center = df_min.groupby('r')['e'].median().reset_index()
        # ax.plot(center['r'], center['e'], lw=1.8)
        ax.plot(bounds['r'], bounds['min'], c=color,
                lw=1.8, alpha=.9, zorder=current_z)
        ax.plot(bounds['r'], bounds['max'], c=color, label=interaction,
                lw=1.8, alpha=.9, zorder=current_z)
        ax.fill_between(
            bounds['r'], bounds['min'], bounds['max'],
            alpha=.08, color=color,
            rasterized=True,
            zorder=current_z
        )
    ax.set_xlabel(r'$r_e$ (Å)')
    ax.set_ylabel('D (eV)')

    df_OA = load_data(molecule, 'OA', zero_zeta)
    E_min_OA = extract_energy_minimums(df_OA, r_max=12)

    xlim = (extract_ER_range(E_min_OA, 'r'))
    ylim = (extract_ER_range(E_min_OA, 'e'))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_yticks(np.linspace(ylim[0], ylim[1], num=4))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    ax.legend(loc='lower right', fontsize=10)

    apply_style(ax, spine=True, grid=False, hide_top_right=False)
    ax.tick_params(width=0.4)

    plt.tight_layout()
    if save_path is not None:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "R_opt_all.pdf", bbox_inches='tight')
    plt.show()

# ======================================================================
# Energy surface plots
# ======================================================================

def plot_energy_surfaces(df, screw_step,
                         mode="chi", cmap="plasma", save_path=None, fix_r=9):
    """Plot energy surfaces at fixed r and at the minimum energy, in the chi or phi1 frame."""
    tol = 1e-6
    df_fix = df[(df['r'] > fix_r - tol) &
                  (df['r'] < fix_r + tol)]
    columns='chi' if mode=='chi' else 'phi1'
    piv_fix = df_fix.pivot_table(index='phi2', columns=columns, values='e')
    piv_min = df.pivot_table(index='phi2', columns=columns, values='e', aggfunc='min')

    E_fix_chi, E_fix_phi2 = build_energy_table(df, piv_fix, screw_step)
    E_min_chi, E_min_phi2 = build_energy_table(df, piv_min, screw_step)

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))

    def plot_panel(ax, E, title, mode):
        im = ax.imshow(E, origin='lower', aspect='equal', cmap=cmap,
                    extent=[0, 360, 0, 360])
        ax.set_xlabel(r'$\varphi_1$ (°)' if mode!='chi' else r'$\chi$ (°)')
        ax.set_ylabel(r'$\varphi_2$ (°)' if mode!='chi' else r'$\varphi$ (°)')
        ax.set_title(title)
        apply_style(ax, spine=True, grid=False, hide_top_right=False)
        return im

    im1 = plot_panel(axes[0], E_fix_chi if mode=='chi' else E_fix_phi2, f'Energy surface (r = {fix_r})', mode)
    im2 = plot_panel(axes[1], E_min_chi if mode=='chi' else E_min_phi2, r'Energy surface ($r_{eq}$)', mode)

    for ax, im in zip(axes, (im1, im2)):
        cbar = fig.colorbar(im, ax=ax, label='Binding Energy (eV)', shrink=0.76, pad=0.04)
        cbar.ax.tick_params(direction='in')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path / f"energy_surface_{mode}.pdf", bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------

def plot_energy_distance_vs_orientation(
    df, x_axis='chi', energy_mode='min', phi2_mode='all', fix_r=9, fix_e=-0.8, tol=0.01,
    step=2, save_path=None):
    """Plot E (top) and r (bottom) vs the chosen angle for a range of phi2, colored by phi2; 
       energy_mode selects min-E, fixed-r, or fixed-E curves."""
    cols = [x_axis, 'phi2', 'r', 'e']

    # ---------- Top: Energy ----------
    if energy_mode == 'fixed_r':
        df_top = df[df['r'] == fix_r]
        top_title = f'Energy vs Orientation (r = {fix_r})'
    else:
        idx_left = df[cols].groupby([x_axis, 'phi2'])['e'].idxmin()
        df_top = df.loc[idx_left]
        top_title = r'$E_{min}$ vs Orientation'

    # ---------- Down: Distance ----------
    if energy_mode == 'fixed_e':
        def pick_fix_e(g):
            g = g.iloc[(g['e'] - fix_e).abs().argsort()]
            return g.iloc[0] if abs(g['e'].iloc[0] - fix_e) <= tol else None

        df_down = (
            df[cols].groupby([x_axis, 'phi2'])
            .apply(pick_fix_e).dropna().reset_index(drop=True)
        )
        down_title = rf'$R$ at $E={fix_e}$'
    else:
        idx_top = df[cols].groupby([x_axis, 'phi2'])['e'].idxmin()
        df_down = df.loc[idx_top]
        down_title = r'$R_{eq}$ vs Orientation'

    fig, axes = plt.subplots(2, 1, figsize=(4.0, 2.8), sharex=True, constrained_layout=True)

    def format_axis(ax, title):
        ax.set_title(title)
        ax.set_xticks(np.linspace(0, 360, 7))
        ax.set_xticks(np.linspace(0, 360, 37), minor=True)
        ax.tick_params(axis="both", which='minor', direction="in")
        apply_style(ax, spine=True, grid=False, hide_top_right=False)

    phi2_min, phi2_max = df['phi2'].min(), df['phi2'].max()
    phi2_vals = np.arange(phi2_min, phi2_max + 0.5 * step, step) if phi2_mode=='all' else np.array([phi2_min])

    cmap = plt.get_cmap(PHI_COLORMAP)
    norm = mpl.colors.Normalize(vmin=phi2_min, vmax=phi2_max)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='right', pad=0.02, fraction=0.08)
    cbar.set_label(r'$\phi_2$ (°)')
    cbar.ax.tick_params(direction='in')

    for phi2 in phi2_vals:
        color = cmap(norm(phi2))

        left = df_top[df_top['phi2'] == phi2].sort_values(x_axis)
        right = df_down[df_down['phi2'] == phi2].sort_values(x_axis)

        axes[0].plot(left[x_axis], left['e'], color=color, lw=1.0)
        axes[1].plot(right[x_axis], right['r'], color=color, lw=1.0)

    color = cmap(norm(phi2_min))

    format_axis(axes[0], top_title)
    format_axis(axes[1], down_title)

    axes[0].set_ylabel('Binding Energy (eV)')
    axes[1].set_ylabel('Pair Distance (Å)')
    axes[1].set_xlabel(_angle_label(x_axis))
    if save_path is not None:
        plt.savefig(save_path / "energy_distance_orientation.pdf", bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------

def plot_energy_distance_vs_chi_psi(
    df_raw, screw_step, x_axis='chi', cut_values='auto', n_cuts=6, tol=0.25,
    cmap_name='viridis', save_path=None):
    """Plot D (top) and r_e (bottom) vs chi or psi as line cuts at fixed values of the other angle,
       colored by that value."""
    df_min_e = extract_energy_minimums(df_raw, r_max=12)
    screw_dir = infer_screw_direction(df_raw)

    df = expand_by_screw_periodicity(df_min_e, screw_step=screw_step, screw_dir=screw_dir)

    if x_axis not in ('chi', 'psi'):
        raise ValueError("x_axis must be 'chi' or 'psi'")

    other_axis = 'psi' if x_axis == 'chi' else 'chi'
    data = df[['phi1', 'phi2', x_axis, other_axis, 'e', 'r']].copy()
    data = data.groupby([x_axis, other_axis], as_index=False).agg({'e': 'mean', 'r': 'mean'})

    # Choose cut values
    unique_vals = np.sort(data[other_axis].unique())
    if cut_values == 'auto':
        cut_values = unique_vals if len(unique_vals) <= n_cuts else unique_vals[
            np.linspace(0, len(unique_vals) - 1, n_cuts).astype(int)]
    else:
        cut_values = np.asarray(cut_values)

    fig, axes = plt.subplots(2, 1, figsize=(4.0, 2.8), sharex=True, constrained_layout=True)

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=np.min(cut_values), vmax=np.max(cut_values))

    for val in cut_values:
        color = cmap(norm(val))
        sel = data[circular_distance_deg(data[other_axis].to_numpy(), val) <= tol].copy()
        if not sel.empty:
            sel = sel.groupby(x_axis, as_index=False).agg({'e': 'mean', 'r': 'mean'}).sort_values(x_axis)
            axes[0].plot(sel[x_axis], sel['e'], color=color, lw=1.0)
            axes[1].plot(sel[x_axis], sel['r'], color=color, lw=1.0)

    x_label = r'$\chi$ (°)' if x_axis == 'chi' else r'$\psi$ (°)'
    other_label = r'$\psi$ (°)' if other_axis == 'psi' else r'$\chi$ (°)'

    axes[0].set_title(f'Energy line cuts vs {x_axis}')
    axes[1].set_title(f'Distance line cuts vs {x_axis}')
    axes[0].set_ylabel('D (eV)')
    axes[1].set_ylabel(r'$r_e$ (Å)')
    axes[1].set_xlabel(x_label)

    for ax in axes:
        ax.set_xticks(np.linspace(0, 360, 7))
        ax.set_xticks(np.linspace(0, 360, 37), minor=True)
        ax.tick_params(axis="both", which='minor', direction="in")
        apply_style(ax, spine=True, grid=False, hide_top_right=False)


    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='right', pad=0.02, fraction=0.08)
    cbar.set_label(other_label)
    cbar.ax.tick_params(direction='in')

    if save_path is not None:
        plt.savefig(save_path / f'linecuts_vs_{x_axis}.pdf', bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------

def plot_energy_vs_chi_psi(
    df_raw, screw_step, x_axis='chi', cut_values='auto', n_cuts=6, tol=0.25,
    cmap_name='viridis', save_path=None):
    """Compact single-panel version of plot_energy_distance_vs_chi_psi: 
       plot D vs chi or psi as line cuts at fixed values of the other angle."""
    df_min_e = extract_energy_minimums(df_raw, r_max=12)
    screw_dir = infer_screw_direction(df_raw)

    df = expand_by_screw_periodicity(df_min_e, screw_step, screw_dir=screw_dir)

    if x_axis not in ('chi', 'psi'):
        raise ValueError("x_axis must be 'chi' or 'psi'")

    other_axis = 'psi' if x_axis == 'chi' else 'chi'
    data = df[['phi1', 'phi2', x_axis, other_axis, 'e', 'r']].copy()
    data = data.groupby([x_axis, other_axis], as_index=False).agg({'e': 'mean', 'r': 'mean'})

    # Choose cut values
    unique_vals = np.sort(data[other_axis].unique())
    if cut_values == 'auto':
        cut_values = unique_vals if len(unique_vals) <= n_cuts else unique_vals[
            np.linspace(0, len(unique_vals) - 1, n_cuts).astype(int)]
    else:
        cut_values = np.asarray(cut_values)

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 1.4), constrained_layout=True)

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=np.min(cut_values), vmax=np.max(cut_values))

    for val in cut_values:
        color = cmap(norm(val))
        sel = data[circular_distance_deg(data[other_axis].to_numpy(), val) <= tol].copy()
        if not sel.empty:
            sel = sel.groupby(x_axis, as_index=False).agg({'e': 'mean', 'r': 'mean'}).sort_values(x_axis)
            ax.plot(sel[x_axis], sel['e'], color=color, lw=1.0)

    x_label = r'$\chi$ (°)' if x_axis == 'chi' else r'$\psi$ (°)'
    other_label = r'$\psi$ (°)' if other_axis == 'psi' else r'$\chi$ (°)'

    # ax.set_title(f'Energy line cuts vs {x_axis}')
    ax.set_ylabel('D (eV)')
    ax.set_xlabel(x_label)
    ax.set_ylim(-1.5, -.2)

    ax.set_xticks(np.linspace(0, 360, 7))
    # ax.set_xticks(np.linspace(0, 360, 37), minor=True)
    # ax.tick_params(axis="both", which='minor', direction="in")
    apply_style(ax, spine=True, grid=False, hide_top_right=True)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, location='right', pad=0.02, fraction=0.1)
    cbar.set_label(other_label)
    cbar.ax.tick_params(direction='in', length=.2)

    if save_path is not None:
        plt.savefig(save_path / f'linecuts_vs_{x_axis}.pdf', bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------

def plot_energy_surfaces_difference(df, screw_step, MODEL, harmonics, mode="chi", 
                                    fix_r=None, large_error_order=False, save_path=None):
    """Plot the energy-surface difference (data minus model) at fixed r 
       or at the minimum energy, in the chi or phi1 frame."""
    columns='chi' if mode=='chi' else 'phi1'
    def get_piv_fix(df):
        tol = 1e-6
        df_fix = df[(df['r'] > fix_r - tol) &
                    (df['r'] < fix_r + tol)]
        piv_fix = df_fix.pivot_table(index='phi2', columns=columns, values='e')
        return piv_fix
    def get_piv_min(df):
        return df.pivot_table(index='phi2', columns=columns, values='e', aggfunc='min')

    piv_min_orig = get_piv_min(df) if not fix_r else get_piv_fix(df)
    piv_min_model = get_piv_min(MODEL) if not fix_r else get_piv_fix(MODEL)

    E_min_chi_orig, E_min_phi2_orig = build_energy_table(df, piv_min_orig, screw_step)
    E_min_chi_model, E_min_phi2_model = build_energy_table(df, piv_min_model, screw_step)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.8))

    def plot_panel(ax, E, title):
        if large_error_order:
            im = ax.imshow(E, origin='lower', aspect='equal', vmin=-.02, vmax=.02, cmap='twilight_shifted')
        else:
            im = ax.imshow(E, origin='lower', aspect='equal', cmap='twilight_shifted')
        im = ax.imshow(E, origin='lower', aspect='equal', cmap='twilight_shifted')
        ax.set_xlabel(r'$\varphi_1$ (°)')
        ax.set_ylabel(r'$\varphi_2$ (°)')
        ax.set_title(title)
        apply_style(ax, spine=True, grid=False, hide_top_right=False)
        return im

    diff_data = E_min_chi_orig - E_min_chi_model if mode == 'chi' else E_min_phi2_orig - E_min_phi2_model
    label_title = (rf'$\Delta E_{{min}}$ ($h_{{\chi}}$={harmonics[0]}, $h_{{\psi}}$={harmonics[1]})')
    if large_error_order:
        label_title += ('\n'
        rf'Error Order~{diff_data.max():.3f} eV')

    if fix_r:
        label_title = (rf'$\Delta E (r={fix_r})$ ($h_{{\chi}}$={harmonics[0]}, $h_{{\psi}}$={harmonics[1]})')
        if large_error_order:
            label_title += ('\n'
            rf'Error Order~{diff_data.max():.3f} eV')

    im = plot_panel(ax, diff_data, label_title)

    cbar = fig.colorbar(im, ax=ax, label='Binding Energy (eV)', shrink=0.76, pad=0.04)
    cbar.ax.tick_params(direction='in')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path / f"energy_surface_{mode}.pdf", bbox_inches='tight')
    plt.show()

# ======================================================================
# Convergence plots
# ======================================================================

def plot_rmse_heatmap(D_errors, re_errors, h_chi_max, h_psi_max, error_text_size, save_path=None):
    """Plot RMSE(D) and RMSE(r_e) heatmaps over the h_chi/h_psi grid, with value annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, errors, title in zip(axes, [D_errors, re_errors],
                                ['D Convergence (RMSE)', r'$r_e$ Convergence (RMSE)']):
        im = ax.imshow(errors, origin='lower', cmap='RdBu_r', aspect='auto',
                    vmin=min(D_errors.min(), re_errors.min()), vmax=max(D_errors.max(), re_errors.max())
                    )
        ax.set_xlabel(r'$h_{\chi}$')
        ax.set_ylabel(r'$h_{\psi}$')
        ax.set_title(title)
        ax.set_xticks(range(0, h_chi_max, 1), labels=range(1, h_chi_max+1, 1))
        ax.set_yticks(range(0, h_psi_max, 1), labels=range(1, h_psi_max+1, 1))
        ax.tick_params(axis='both', direction='in', length=0)
        cbar = plt.colorbar(im, ax=ax, label='RMSE')
        cbar.ax.tick_params(direction='in')
        for n in range(h_chi_max):
            for m in range(h_psi_max):
                text = ax.text(n, m, f'{errors[m, n]:.3f}', size=error_text_size, 
                               ha="center", va="center", color="w")

    plt.suptitle('Fourier Model Convergence vs Harmonic Numbers')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path/ 'fourier_convergence_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nLow-error points (h_chi, h_psi, RMSE_D, RMSE_re):")
    print(f"  ({h_chi_max}, {h_chi_max}): {D_errors[-1,-1]:.4f}, {re_errors[-1,-1]:.4f}")

# ----------------------------------------------------------------------

def plot_harmonic_convergence(df_org, interaction, screw_step, harmonics, error_text_size=7):
    """Plot RMSE(D) and RMSE(r_e) heatmaps over h_chi/h_psi, 
       plus line cuts of the best RMSE vs each harmonic order."""
    E_min_df = extract_energy_minimums(df_org, r_max=12)
    symm_chi = get_symm_chi(interaction)

    D = -E_min_df['e']
    re = E_min_df['r']

    chi_rad = np.deg2rad(E_min_df['chi'])
    psi_rad = np.deg2rad(E_min_df['psi'])

    D_rmse = compute_harmonic_rmse_grid(chi_rad, psi_rad, D, harmonics[0], harmonics[1], 
                                        symm_chi=symm_chi, screw_step=screw_step)
    re_rmse = compute_harmonic_rmse_grid(chi_rad, psi_rad, re, harmonics[0], harmonics[1], 
                                         symm_chi=symm_chi, screw_step=screw_step)

    k0 = int(round(360 / (2 * screw_step)))

    h_chi_vals = np.arange(1, D_rmse.shape[0] + 1)
    h_psi_vals = np.arange(1, D_rmse.shape[1] + 1)
    psi_labels = [str(k0 * j) for j in h_psi_vals]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    # ---------- Heatmaps ----------
    datasets = [
        (D_rmse, r'RMSE for $D$'),
        (re_rmse, r'RMSE for $r_e$')
    ]

    for i, (data, title) in enumerate(datasets):
        ax = axes[i, 0]
        im = ax.imshow(data, origin='lower', aspect='auto', cmap='RdBu_r')

        ax.set_title(title)
        ax.set_xlabel(r'$\psi$ harmonic (actual $n$)')
        ax.set_ylabel(r'$\chi$ harmonic')

        ax.set_xticks(np.arange(len(h_psi_vals)))
        ax.set_xticklabels(psi_labels)
        ax.set_yticks(np.arange(len(h_chi_vals)))
        ax.set_yticklabels(h_chi_vals)

        fig.colorbar(im, ax=ax, label='RMSE')
        for m in range(data.shape[0]):
            for n in range(data.shape[1]):
                text = ax.text(n, m, f'{data[m, n]:.4f}', size=error_text_size, ha="center", va="center", color="w")

    # ---------- Line cuts ----------
    line_data = [
        (h_chi_vals, D_rmse.min(axis=1), re_rmse.min(axis=1),
         r'Best RMSE vs $\chi$', r'$\chi$ harmonic', r'best RMSE over $\psi$'),

        (h_psi_vals * k0, D_rmse.min(axis=0), re_rmse.min(axis=0),
         r'Best RMSE vs $\psi$', r'$\psi$ harmonic (actual $n$)', r'best RMSE over $\chi$')
    ]

    for i, (x, y_D, y_re, title, xlabel, ylabel) in enumerate(line_data):
        ax = axes[i, 1]

        ax.plot(x, y_D, marker='o', label=r'$D$', color='green')
        ax.plot(x, y_re, marker='s', label=r'$r_e$', color='orange')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------

def plot_3D_surface_wireframe(df, fix_r=None, mode='chi', save_path=None,
                                       elev=25, azim=-60, smooth=True):
    """Plot a 3D wireframe surface of E(phi1,phi2) (or chi/phi) at fixed r or at the minimum energy, 
       optionally Gaussian-smoothed."""
    columns = 'chi' if mode == 'chi' else 'phi1'
    if fix_r is not None:
        tol = 1e-6
        df_slice = df[(df['r'] > fix_r - tol) & (df['r'] < fix_r + tol)]
        piv = df_slice.pivot_table(index='phi2', columns=columns, values='e')
        title = rf'$E(\varphi_1, \varphi_2)$ at r = {fix_r} Å'
    else:
        piv = df.pivot_table(index='phi2', columns=columns, values='e', aggfunc='min')
        title = r'$E_{min}(\varphi_1, \varphi_2)$'

    E_full = build_full_energy_table(df, piv, mode=mode)

    if smooth:
        E_full = gaussian_filter(E_full, sigma=0.8)

    n_x = E_full.shape[1]
    n_y = E_full.shape[0]
    x_vals = np.linspace(0, 360, n_x)
    y_vals = np.linspace(0, 360, n_y)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = E_full

    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(Z.min(), Z.max())
    colors = plt.cm.RdYlBu_r(norm(Z))

    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=colors,
        shade=False,
        edgecolor='black',
        linewidth=0.4,
        rcount=70, ccount=70,
        alpha=0.9
    )

    xlabel = r'$\chi$ (deg)' if mode == 'chi' else r'$\varphi_2$ (deg)'
    ylabel = r'$\varphi$ (deg)' if mode == 'chi' else r'$\varphi_1$ (deg)'
    zlabel = 'Energy (a.u.)'

    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=-7)

    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_linewidth(1.5)
    ax.yaxis.pane.set_linewidth(1.5)
    ax.zaxis.pane.set_linewidth(1.5)
    ax.invert_yaxis()

    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=12, pad=0.06)
    cbar.set_label('Energy (a.u.)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()

    if save_path is not None:
        suffix = f"_r{fix_r}" if fix_r is not None else "_Emin"
        fname = f"3D_wireframe_detailed_{mode}{suffix}.pdf"
        plt.savefig(save_path / fname, bbox_inches='tight', dpi=300)
    plt.show()

# ======================================================================
# Chi-psi panel plots
# ======================================================================

def plot_energy_vs_chi_psi_compact(df_raw, interaction, screw_step, x_axis='chi',
                         cut_values=[0, 5, 10, 15], colors=INTERACTION_CMAPS,
                         tol=0.25, save_path=None):
    """Plot E vs chi or psi as line cuts at fixed values of the other angle,
       with inline colored labels instead of a legend or colorbar."""
    cut_values = np.asarray(cut_values)

    fig, ax = plt.subplots(figsize=(2.3, 1.2), constrained_layout=True)

    df_min_e = extract_energy_minimums(df_raw, r_max=12)
    screw_dir = infer_screw_direction(df_raw)
    df_cuts = expand_by_screw_periodicity(df_min_e, screw_step, screw_dir=screw_dir)
    data = df_cuts[['phi1', 'phi2', 'chi', 'psi', 'e', 'r']].groupby(
        ['chi', 'psi'], as_index=False).agg({'e': 'mean', 'r': 'mean'})

    cmap = colors[interaction]
    base_cmap = plt.get_cmap(cmap)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'truncated_cmap', base_cmap(np.linspace(0.0, .4, 256)))

    x_txt, y_txt= .15, .66

    cut_vals_chi = np.asarray(cut_values)
    if x_axis=='chi':
        active_cuts, norm = plot_line_cuts(ax, data, 'chi', 'psi', cut_vals_chi, cmap, orientation='horizontal')
    else:
        active_cuts, norm = plot_line_cuts(ax, data, 'psi', 'chi', cut_vals_chi, cmap, orientation='horizontal')

    ax.set_xticklabels([])
    ax.set_ylabel('E (eV)', labelpad=1)
    ax.set_yticks([-1, -0.5])
    ax.set_xlim(0, 360)


    for idx, val in enumerate(active_cuts):
        ax.text(x_txt, y_txt - idx * 0.15, f"$\\psi$={val}°",
                    color=cmap(norm(val)), fontsize=7, transform=ax.transAxes,
                    va='top', ha='left')
    apply_compact_style(ax, tick_width=0)

    if save_path is not None:
        plt.savefig(save_path / f'linecuts_{x_axis}.pdf', bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------

def plot_energy_surfaces_chi_psi(df, interaction, screw_step,
                                 colors=INTERACTION_CMAPS,
                                 fix_r=None, model_df=None,
                                 plot_type='reference', left_label=True, save_path=None):
    """Plot the E(chi,psi) heatmap for the data, model, or their difference (plot_type), 
       at fixed r or at the minimum energy."""
    colormap = colors[interaction]

    tol = 1e-6
    df = expand_chi_psi_by_screw_periodicity(df, screw_step)

    if fix_r:
        df_plot = df[np.abs(df['r'] - fix_r) < tol]
        E = make_surface(df_plot, 'mean')
    else:
        E = make_surface(df, 'min')

    if plot_type == 'difference':
        model_df = expand_chi_psi_by_screw_periodicity(model_df, screw_step)
        if fix_r:
            model_df_plot = model_df[np.abs(model_df['r'] - fix_r) < tol]
            E_model = make_surface(model_df_plot, 'mean')
        else:
            E_model = make_surface(model_df, 'min')
        E = E - E_model
        colormap = 'twilight_shifted'

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.8))

    im = draw_chi_psi_heatmap(ax, E, colormap, left_label=left_label)

    ax.set_xticks([0, 90, 180, 270, 360])
    if left_label:
        ax.set_yticks([0, 90, 180, 270, 360])
    else:
        ax.set_yticklabels('')

    apply_style(ax, spine=True, grid=False, hide_top_right=False)
    ax.tick_params(width=0.4)

    shrink_val = .704 if left_label else .638
    cbar_label = 'ΔE (eV)' if plot_type == 'difference' else 'E (eV)'
    cbar = fig.colorbar(im, ax=ax, shrink=shrink_val, pad=0.0008,
                       orientation='horizontal', location='top')
    cbar.ax.tick_params(direction='in', length=6.4, width=0.2)
    cbar.set_label(cbar_label, labelpad=7)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_color('grey')

    plt.tight_layout()
    if save_path is not None:
        plot_name = f"energy_surface_chi_psi_{plot_type}.pdf"
        plt.savefig(save_path / plot_name, bbox_inches='tight')

    plt.show()

# ----------------------------------------------------------------------

def plot_chi_psi_panel(df_org, interaction, screw_step,
                       left_label=True, colors=INTERACTION_CMAPS,
                       psi_selection=[60, 150, 210, 300], save_path=None):
    """Combined panel: E(chi,psi) heatmap with E vs chi line cuts (top, at fixed psi) 
       and E vs psi line cuts (right, at fixed chi), each with inline colored labels."""
    df = expand_chi_psi_by_screw_periodicity(df_org, screw_step)
    cmap = colors[interaction]

    fig = plt.figure(figsize=(3.1, 4))
    gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[4.8, 2.2], height_ratios=[.9, 1.95, .8],
                           wspace=0.0, hspace=0.0, figure=fig)
    text_pos = {'EP':(.15, .66, .16, .93),
                'EA':(.18, .68, .18, .93),
                'OP':(.40, .68, .37, .9),
                'OA':(.18, .68, .34, .9)}

    chi_selection = {'EP':[0, 50, 120, 180],
                    'EA':[0, 100, 180, 270],
                    'OP':[0, 20, 180, 230],
                    'OA':[0, 50, 180, 270]}

    # ---------- Heatmap ----------
    ax_heatmap = fig.add_subplot(gs[1:, 0])
    E = make_surface(df, 'min')
    im = draw_chi_psi_heatmap(ax_heatmap, E, cmap, left_label=left_label, xlabel_pad=-2)

    ax_heatmap.set_xticks([0, 120, 240])
    ax_heatmap.set_yticks([0, 120, 240])
    ax_heatmap.tick_params(labelbottom=True, labelleft=True)
    # ax_heatmap.tick_params(axis='y', labelrotation=90)
    apply_compact_style(ax_heatmap)

    cbar = fig.colorbar(
        im, ax=ax_heatmap,
        # label='E (eV)',
        ticks=[-1, -.5],
        shrink=1, pad=0.14,
        orientation='horizontal',location='bottom'
    )
    cbar.ax.tick_params(direction='in', length=5.5, width=.2)
    cbar.outline.set_linewidth(.6)

    # ---------- Top: chi line cuts ----------
    ax_chi = fig.add_subplot(gs[0, 0])

    df_min_e = extract_energy_minimums(df_org, r_max=12)
    screw_dir = infer_screw_direction(df_org)
    df_cuts = expand_by_screw_periodicity(df_min_e, screw_step, screw_dir=screw_dir)
    data = df_cuts[['phi1', 'phi2', 'chi', 'psi', 'e', 'r']].groupby(
        ['chi', 'psi'], as_index=False).agg({'e': 'mean', 'r': 'mean'})

    base_cmap = plt.get_cmap(cmap)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'truncated_cmap', base_cmap(np.linspace(0.0, .4, 256)))

    x_chi, y_chi, x_psi, y_psi = text_pos[interaction]

    cut_vals_chi = np.asarray(psi_selection)
    active_cuts_chi, norm_chi = plot_line_cuts(ax_chi, data, 'chi', 'psi', cut_vals_chi, cmap, 
                                               orientation='horizontal')

    ax_chi.set_xticklabels([])
    if left_label:
        ax_chi.set_ylabel('E (eV)', labelpad=1)
    ax_chi.set_yticks([-1, -0.5])
    ax_chi.set_xlim(0, 360)
    # ax_chi.tick_params(axis='y', labelrotation=90)


    for idx, val in enumerate(active_cuts_chi):
        ax_chi.text(x_chi, y_chi - idx * 0.15, f"$\\psi$={val}°",
                    color=cmap(norm_chi(val)), fontsize=7, transform=ax_chi.transAxes,
                    va='top', ha='left')
    apply_compact_style(ax_chi, tick_width=0)

    # ---------- Right: psi line cuts ----------
    ax_psi = fig.add_subplot(gs[1, 1])

    cut_vals_psi = np.asarray(chi_selection[interaction])
    active_cuts_psi, norm_psi = plot_line_cuts(ax_psi, data, 'psi', 'chi', cut_vals_psi, cmap, 
                                               orientation='vertical')

    ax_psi.set_ylabel('')
    ax_psi.set_yticklabels([])
    ax_psi.set_xlabel('E (eV)', labelpad=10)
    ax_psi.set_xticks([-1, -0.5])
    ax_psi.set_ylim(0, 360)

    for idx, val in enumerate(active_cuts_psi):
        ax_psi.text(x_psi, y_psi - idx * 0.11, f"$\\chi$={val}°",
                    color=cmap(norm_psi(val)), fontsize=7, transform=ax_psi.transAxes,
                    va='top', ha='left')

    apply_compact_style(ax_psi, tick_width=0)
    ax_psi.set_yticks([0, 90, 180, 270])

    if save_path is not None:
        plt.savefig(save_path / 'psi_chi_panel.pdf', bbox_inches='tight')
    plt.show()

# ======================================================================
# Convergence panel
# ======================================================================

def plot_convergence_panel(df, interaction, screw_step, harmonics=((0, 9), 9),
                           left_label=True, colors=INTERACTION_CMAPS,
                           data_type='D', error_text_size=5, save_path=None):
    """Convergence panel: RMSE(D, r_e, or alpha) heatmap over h_chi x h_psi, 
       with line cuts of the best RMSE vs each harmonic order."""
    cmap_name = colors[interaction]
    cmap_name = cmap_name.reversed()

    light_color = plt.get_cmap(colors[interaction])(1.0)

    E_min_df = extract_energy_minimums(df, r_max=12)

    D = -E_min_df['e']
    re = E_min_df['r']
    chi_rad = np.deg2rad(E_min_df['chi'])
    psi_rad = np.deg2rad(E_min_df['psi'])

    h_chi_range, h_psi_max = harmonics
    symm_chi = (interaction[1] == 'P')
    D_rmse = compute_harmonic_rmse_grid(chi_rad, psi_rad, D, h_chi_range, h_psi_max,
                                        symm_chi, screw_step=screw_step)
    re_rmse = compute_harmonic_rmse_grid(chi_rad, psi_rad, re, h_chi_range, h_psi_max,
                                         symm_chi, screw_step=screw_step)
    if data_type == 'D':
        data = D_rmse
    elif data_type == 're':
        data = re_rmse
    else:
        alpha_vals = fit_alpha_values(df, interaction)
        data = compute_harmonic_rmse_grid(chi_rad, psi_rad, alpha_vals, h_chi_range, h_psi_max,
                                         symm_chi, screw_step=screw_step)
    data *= 1000
    k0 = int(round(360 / (2 * screw_step)))

    h_chi_vals = np.arange(h_chi_range[0], h_chi_range[1] + 1)
    h_psi_vals = np.arange(1, D_rmse.shape[1] + 1)

    fig = plt.figure(figsize=(2.4, 3.2))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1.3, .7],
                           height_ratios=[0.6, 1.9], wspace=0.0, hspace=0.0, figure=fig)

    base_cmap = plt.get_cmap(cmap_name)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'truncated_cmap', base_cmap(np.linspace(0.5, 1.0, 256)))

    # ---------- Heatmap ----------
    ax_heatmap = fig.add_subplot(gs[1, 0])
    im = ax_heatmap.imshow(data, origin='lower', aspect='auto',
                           cmap=cmap_name,
                           alpha=0.8)

    ax_heatmap.set_xlabel(r'$N_{\psi}$')
    if left_label:
        ax_heatmap.set_ylabel(r'$N_{\chi}$')
    ax_heatmap.set_xticks(np.arange(len(h_psi_vals)))
    ax_heatmap.set_xticklabels([str(k0 * j) for j in h_psi_vals])
    ax_heatmap.set_yticks(np.arange(len(h_chi_vals)))
    ax_heatmap.set_yticklabels(h_chi_vals)

    down_row = 1 if interaction[0]=='E' else 0
    for m in range(data.shape[0]):
        for n in range(data.shape[1]):
            ax_heatmap.text(n, m, f'{data[m, n]:.1f}', size=error_text_size,
                          ha="center", va="center", color="black" if m > down_row else light_color)
    apply_compact_style(ax_heatmap)

    # ---------- Top: psi line ----------
    ax_psi_line = fig.add_subplot(gs[0, 0])
    ax_psi_line.plot(h_psi_vals * k0, data.min(axis=0), marker='o',
                     color=cmap(0.9), linewidth=1.2, markersize=3)
    ax_psi_line.set_xticklabels([])
    # if left_label:
    #     ax_psi_line.set_ylabel('best RMSE\n(meV)', size=6, labelpad=-1)

    formatter = ax_psi_line.yaxis.get_major_formatter()
    formatter.set_useOffset(True)
    formatter.set_scientific(True)

    if interaction=='EA':
        label_decimal='%.2f'
    elif interaction=='EP':
        label_decimal='%.1f'
    else:
        label_decimal='%.3f'

    ax_psi_line.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax_psi_line.yaxis.set_major_formatter(mticker.FormatStrFormatter(label_decimal))

    ax_psi_line.tick_params(labelbottom=False)
    apply_compact_style(ax_psi_line, tick_width=0)
    # ---------- Right: chi line ----------
    ax_chi_line = fig.add_subplot(gs[1, 1])
    ax_chi_line.plot(data.min(axis=1), h_chi_vals, marker='o',
                     color=cmap(0.9), linewidth=1.2, markersize=3)
    # ax_chi_line.set_xlabel('best RMSE\n(meV)', size=6)
    ax_chi_line.set_xlabel('(meV)', size=8)
    ax_chi_line.set_yticklabels([])
    ax_chi_line.tick_params(labelleft=False)
    if interaction[0]=='O':
        ax_chi_line.set_xlim(2, 20)
    apply_compact_style(ax_chi_line, tick_width=0)

    if save_path is not None:
        plt.savefig(save_path / 'convergence_panel.pdf', bbox_inches='tight')

    plt.show()

# ======================================================================
# Parity plots
# ======================================================================

def plot_parity(df_data, df_model, interaction, alpha_fit,
                colors=INTERACTION_CMAPS,
                compare_target='Emin', save_path=None):
    """Parity scatter (model vs reference) for D, r_e, or full E (compare_target), with a 1:1 reference line."""
    point_color = plt.get_cmap(colors[interaction])(0.2)

    E_data, E_model, D_model, D_data, re_model, re_data = extract_energy_comparison(df_data, df_model)
    if compare_target=='Emin':
        ref_vals, model_vals = D_data, D_model
    elif compare_target=='re':
        ref_vals, model_vals = re_data, re_model
    elif compare_target=='E':
        ref_vals, model_vals = E_data, E_model
    else:
        raise ValueError(f'Entered compare_target *{compare_target}* is not valid! must be among: E, Emin, re')

    fig, ax = plt.subplots(1, 1, figsize=(2.6, 2.5))

    ax.scatter(
        ref_vals, model_vals,
        s=.2, alpha=.8,
        color=point_color,
        linewidths=0,
        rasterized=True
    )

    lim_min = np.min([ref_vals.min(), model_vals.min()])
    lim_max = np.max([ref_vals.max(), model_vals.max()]) if compare_target!='E' else 5

    lims = [lim_min, lim_max]

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # 1:1 line
    ax.plot(lims, lims, '--', color='pink', linewidth=1, zorder=1)

    ticks = np.linspace(lims[0] + 0.05, lims[1] - 0.05, num=5)

    if compare_target == 'E':
        ax.plot(lims, [0,0], linewidth=.5, color='gray', alpha=.3, linestyle='--')
        ax.plot([0,0], lims, linewidth=.5, color='gray', alpha=.3, linestyle='--')
        ticks = np.linspace(-1, 5, num=7)

    label_decimal = '{x:.0f}' if compare_target=='E' else '{x:.1f}'
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(label_decimal)
    ax.yaxis.set_major_formatter(label_decimal)

    ax.set_xlabel('Reference (eV)')
    ax.set_ylabel('Model (eV)')

    apply_style(ax, spine=True, grid=False, hide_top_right=False)
    ax.tick_params(width=0.4)

    plt.tight_layout()

    if save_path is not None:
        plot_name = f"parity_plot_{compare_target}.pdf"
        if alpha_fit:
            plot_name = f"parity_plot_{compare_target}_alpha.pdf"
        plt.savefig(save_path / plot_name, bbox_inches='tight')

    plt.show()

# ----------------------------------------------------------------------

def plot_parity_angle_color(df_data, df_model, color_by, zoom_in=None, save_path=None):
    """Parity scatter (model vs reference) colored by chi, with a 1:1 reference line."""

    cols_no_e = [c for c in df_data.columns if c not in ('e', 'psi', 'chi')]
    model_subset = df_model.merge(
        df_data[cols_no_e].drop_duplicates(),
        on=cols_no_e,
        how='inner'
    ).copy()

    sort_cols = cols_no_e + [c for c in ('psi', 'chi') if c in df_data.columns]
    E_model = model_subset.sort_values(by=sort_cols).reset_index(drop=True)
    E_ref  = df_data.sort_values(by=sort_cols).reset_index(drop=True)

    cmap = plt.get_cmap('twilight')
    norm = plt.Normalize(E_model[color_by].min(), E_model[color_by].max())

    fig, ax = plt.subplots(1, 1, figsize=(2.6, 2.5))

    ax.scatter(
        E_ref['e'], E_model['e'],
        s=1, alpha=.3,
        c=E_model[color_by],
        cmap=cmap, norm=norm,
        linewidths=0, rasterized=True, edgecolors='none'
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=.7)
    cbar.ax.tick_params(direction='in', width=.2)
    cbar.outline.set_color('grey')
    cbar.outline.set_linewidth(.5)

    lim_min = np.min([E_ref['e'].min(), E_model['e'].min()])
    lim_max = np.max([E_ref['e'].max(), E_model['e'].max()])
    if zoom_in!=None:
        lim_max = zoom_in
    lims = [lim_min, lim_max]

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    ax.plot(lims, lims, '--', color='pink', linewidth=1, zorder=1)

    ticks = np.linspace(lims[0] + 0.05, lims[1] - 0.05, num=5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter('{x:.1f}')
    ax.yaxis.set_major_formatter('{x:.1f}')

    ax.set_xlabel('Reference (eV)')
    ax.set_ylabel('Model (eV)')

    apply_style(ax, spine=True, grid=False, hide_top_right=False)
    ax.tick_params(width=0.4)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path / "parity_plot.pdf", bbox_inches='tight')

    plt.show()

# ======================================================================
# Pruning / coefficient plots
# ======================================================================

def plot_rmse_pruning_coeff_threshold(interaction,
                                       rmse_list, n_coeff_list,
                                       relative_thresholds, coeff_type,
                                       colors=INTERACTION_CMAPS, save_path=None):
    """Plot RMSE vs number of retained coefficients (left) and vs relative pruning threshold (right, log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    rmse_meV = np.array(rmse_list) * 1000
    h_line = 5 if coeff_type=='D' else 10
    color = plt.get_cmap(colors[interaction])(0.2)

    n_min = 0 if coeff_type=='D' else 5
    n_max = max(n_coeff_list)
    # ---------- Left: RMSE vs number of coefficients ----------
    axes[0].plot(n_coeff_list, rmse_meV, 'o-', markersize=5,
                 color=color, linewidth=1.5, markerfacecolor='white',
                 markeredgewidth=1.5, markeredgecolor=color)
    axes[0].set_xlabel('# Coefficients')
    axes[0].set_ylabel('RMSE (meV)')
    axes[0].set_xlim(n_min, n_max+ n_max//10)

    # ---------- Right: RMSE vs threshold (log scale) ----------
    axes[1].semilogx(relative_thresholds, rmse_meV, 's-', markersize=4,
                     color=color, linewidth=1.5, markerfacecolor='white',
                     markeredgewidth=1.5, markeredgecolor=color)
    axes[1].set_xlabel('Relative Threshold')
    axes[1].sharey(axes[0])
    axes[1].tick_params(axis='both', which='minor', width=0.4, length=2,
                direction='in')

    for ax in axes:
        ax.axhline(y=h_line, color='gray', lw=0.8, alpha=0.5, linestyle='--', zorder=0)
        ax.set_ylim(0, 35)

        apply_style(ax, spine=True, grid=False, hide_top_right=False)
        ax.tick_params(width=0.5, labelsize=9)
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, axis='y')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path / f'rmse_pruning_{coeff_type}.pdf', bbox_inches='tight', dpi=300)

    plt.show()

# ----------------------------------------------------------------------

def plot_rmse_pruning(interaction, A, D, re, relative_th, alpha=None,
                      hide_label=True, hide_left_label=True,
                      colors=INTERACTION_CMAPS, save_path=None):
    """Plot RMSE vs number of retained coefficients for D, r_e, and (if given) alpha, 
       from magnitude-based pruning over relative_th."""
    rmse_list_D ,n_coeff_list_D = prune_by_magnitude(A, D, relative_th)
    rmse_list_re ,n_coeff_list_re = prune_by_magnitude(A, re, relative_th)
    rmse_meV_D = np.array(rmse_list_D) * 1000
    rmse_meA_re = np.array(rmse_list_re) * 1000

    if alpha is not None:
        rmse_list_alpha ,n_coeff_list_alpha = prune_by_magnitude(A, alpha, relative_th)
        rmse_me_alpha = np.array(rmse_list_alpha) * 1000

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.6))

    color_D = plt.get_cmap(colors[interaction])(0.15)
    color_re = plt.get_cmap(colors[interaction])(0.4)
    color_alpha = plt.get_cmap(colors[interaction])(0.7)

    n_min = 0
    n_max = max(n_coeff_list_D)

    ax.plot(n_coeff_list_D, rmse_meV_D, 'o-', markersize=5,
                 color=color_D, linewidth=1.5, markerfacecolor='white',
                 markeredgewidth=1.5, markeredgecolor=color_D, label='D')
    ax.plot(n_coeff_list_re, rmse_meA_re, 's-', markersize=5,
                 color=color_re, linewidth=1.5, markerfacecolor='white',
                 markeredgewidth=1.5, markeredgecolor=color_re, label=r'$r_e$')
    if alpha is not None:
        ax.plot(n_coeff_list_alpha, rmse_me_alpha, '*-', markersize=5,
                color=color_alpha, linewidth=1.5, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor=color_alpha, label=r'$\alpha$')

    ax.set_xlabel('# Coefficients')
    if not hide_left_label:
        ax.set_ylabel('RMSE (meV)')
    else:
        ax.set_yticklabels('')
    # ax.set_xlim(n_min, n_max+ n_max//10)

    ax.axhline(y=5, color='green', lw=0.8, alpha=0.5, linestyle='--', zorder=0)
    ax.axhline(y=10, color='gray', lw=0.8, alpha=0.5, linestyle='--', zorder=0)
    ax.set_ylim(0, 50)

    apply_style(ax, spine=True, grid=False, hide_top_right=False)
    ax.tick_params(width=0.5, labelsize=9)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, axis='y')

    plt.tight_layout()
    if not hide_label:
        plt.legend()
    if save_path is not None:
        alpha_tag = 'w_alpha' if alpha is not None else 'wo_alpha'
        plt.savefig(save_path / f'rmse_pruning_{alpha_tag}.pdf', bbox_inches='tight', dpi=300)

    plt.show()

# ----------------------------------------------------------------------

def plot_prune_coefficients(df, molecule, interaction,
                             harmonic_ceils, relative_th, alpha_fit,
                             print_errors=False, colors=INTERACTION_CMAPS, save_path=None):
    """Compute D, r_e (and alpha if alpha_fit), build the Fourier design matrix, and call plot_rmse_pruning."""
    print_modeling_information(molecule, interaction, harmonic_ceils)

    E_min_df = extract_energy_minimums(df, r_max=12)
    D, re = -E_min_df['e'], E_min_df['r']
    chi_rad, psi_rad = np.deg2rad(E_min_df['chi']), np.deg2rad(E_min_df['psi'])
    h_chi, h_psi = harmonic_ceils[interaction]
    symm_chi = get_symm_chi(interaction)

    A, labels = create_matrix_lsqt_2d(h_chi, h_psi, chi_rad, psi_rad, symm_chi, molecule.screw_step)

    alpha_vals = fit_alpha_values(df, interaction) if alpha_fit else None
    plot_rmse_pruning(interaction, A, D, re, relative_th, alpha=alpha_vals,
                        hide_label=False, hide_left_label=False,
                        colors=colors, save_path=save_path)
