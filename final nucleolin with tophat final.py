# Nucleolin Aggregate Analysis in C9 vs Isogenic Organoids

import nd2
import numpy as np
import pandas as pd
from skimage import filters, morphology, measure, segmentation, exposure
from skimage.filters import unsharp_mask
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob
import warnings
import traceback
from matplotlib.patches import Circle
import pingouin as pg
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# Parameters
PIXEL_SIZE_UM = 0.11
MAX_AGGREGATE_AREA_UM2 = 10.0
AGGREGATE_INTENSITY_THRESHOLD = 0.2
NUCLEOLIN_POSITIVITY_THRESHOLD = 0.01

# Nucleus quality filters
MAX_NUCLEUS_AREA_UM2 = 100.0
MAX_DAPI_INTENSITY_PERCENTILE = 80

# QC Parameters
QC_MODE = True  # Set to False to skip QC plots
QC_SAMPLE_FRACTION = 0.3  # Plot QC for this fraction of images (to avoid too many plots)
SAVE_QC_FIGURES = True  # Save QC figures to disk

MAX_AGGREGATE_AREA_PIXELS = MAX_AGGREGATE_AREA_UM2 / (PIXEL_SIZE_UM ** 2)
MAX_NUCLEUS_AREA_PIXELS = MAX_NUCLEUS_AREA_UM2 / (PIXEL_SIZE_UM ** 2)

print(f"Max aggregate area: {MAX_AGGREGATE_AREA_PIXELS:.1f} pixels")
print(f"Max nucleus area: {MAX_NUCLEUS_AREA_PIXELS:.1f} pixels")
print(f"QC Mode: {QC_MODE}, Sample Fraction: {QC_SAMPLE_FRACTION}")

# File setup
nd2_files = sorted(glob.glob("*.nd2"))
conditions = []
for f in nd2_files:
    if "N6" in f or "n6" in f:
        conditions.append("C9")
    elif "I6" in f or "i6" in f or "Isogenic" in f:
        conditions.append("Isogenic")
    else:
        raise ValueError(f"Cannot determine condition for: {f}")

print(f"Found {len(nd2_files)} files: {conditions}")

# Create QC directory if needed
if QC_MODE and SAVE_QC_FIGURES and not os.path.exists("QC_Figures"):
    os.makedirs("QC_Figures")


# Helper function for QC plots
import numpy as np


def cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size for two independent samples.

    Parameters
    ----------
    x, y : array-like
        Two independent samples.

    Returns
    -------
    delta : float
        Cliff's delta: ranges from -1 to 1.
        0 = no effect, 1 = all x > y, -1 = all x < y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Handle empty inputs
    if x.size == 0 or y.size == 0:
        return np.nan

    # Broadcast comparison: (len(x), len(y))
    diff = x[:, None] - y[None, :]

    wins = np.sum(diff > 0)  # x_i > y_j
    ties = np.sum(diff == 0)  # x_i == y_j
    losses = np.sum(diff < 0)  # x_i < y_j

    # Standard Cliff's delta ignores ties (or treats them as 0)
    # Formula: (wins - losses) / (n_x * n_y)
    return (wins - losses) / (x.size * y.size)
def plot_qc_figure(fig, filename, condition, image_name, step_description):
    """Helper to save QC figures with consistent formatting"""
    if SAVE_QC_FIGURES:
        safe_name = os.path.splitext(image_name)[0].replace(" ", "_").replace(".", "_")
        qc_filename = f"QC_Figures/{safe_name}_{step_description}.png"
        fig.savefig(qc_filename, dpi=150, bbox_inches='tight')
        print(f"    Saved QC figure: {qc_filename}")


def analyze_nd2_file(filepath, qc_plot=False):

    image_name = os.path.basename(filepath)
    print(f"\nProcessing: {image_name}")

    with nd2.ND2File(filepath) as ndfile:
        # Load full array before file closes
        img_4d = np.asarray(ndfile.to_xarray().values)

    # Ensure 4D ZCYX
    if img_4d.ndim != 4:
        raise ValueError(f"Expected 4D array (Z, C, Y, X), got shape {img_4d.shape}")

    if img_4d.shape[1] < 2:
        raise ValueError(f"File {filepath} has only {img_4d.shape[1]} channels. Expected at least 2.")

    print(f"  Original shape: {img_4d.shape}")

    # Channel extraction (Z,C,Y,X):
    dapi_3d       = img_4d[:, 0, :, :]
    nucleolin_3d  = img_4d[:, 1, :, :]

    # Projections:
    dapi       = dapi_3d.max(axis=0)
    nucleolin  = nucleolin_3d.max(axis=0)

    # Middle Z slice
    middle_z = img_4d.shape[0] // 2
    dapi_middle      = img_4d[middle_z, 0, :, :]
    nucleolin_middle = img_4d[middle_z, 1, :, :]


    image_name = os.path.basename(filepath)
    print(f"\nProcessing: {image_name}")
    print(f"  Original shape: {img_4d.shape}")
    print(f"  DAPI range: [{dapi.min():.1f}, {dapi.max():.1f}]")
    print(f"  Nucleolin range: [{nucleolin.min():.1f}, {nucleolin.max():.1f}]")

    # === QC PLOT 1: RAW IMAGES ===
    if qc_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'QC: Raw Images - {image_name}', fontsize=16)

        # DAPI projections
        axes[0, 0].imshow(dapi, cmap='gray', vmax=np.percentile(dapi, 99))
        axes[0, 0].set_title('DAPI Max Projection')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(dapi_middle, cmap='gray', vmax=np.percentile(dapi_middle, 99))
        axes[0, 1].set_title(f'DAPI Z={middle_z}')
        axes[0, 1].axis('off')

        axes[0, 2].hist(dapi.ravel(), bins=100, alpha=0.7, color='blue')
        axes[0, 2].axvline(np.percentile(dapi, 50), color='red', linestyle='--', label='Median')
        axes[0, 2].axvline(np.percentile(dapi, 99), color='green', linestyle='--', label='99th %')
        axes[0, 2].set_title('DAPI Intensity Distribution')
        axes[0, 2].legend()
        axes[0, 2].set_xlabel('Intensity')
        axes[0, 2].set_ylabel('Frequency')

        # Nucleolin projections
        axes[1, 0].imshow(nucleolin, cmap='gray', vmax=np.percentile(nucleolin, 99))
        axes[1, 0].set_title('Nucleolin Max Projection')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(nucleolin_middle, cmap='gray', vmax=np.percentile(nucleolin_middle, 99))
        axes[1, 1].set_title(f'Nucleolin Z={middle_z}')
        axes[1, 1].axis('off')

        axes[1, 2].hist(nucleolin.ravel(), bins=100, alpha=0.7, color='orange')
        axes[1, 2].axvline(np.percentile(nucleolin, 50), color='red', linestyle='--', label='Median')
        axes[1, 2].axvline(np.percentile(nucleolin, 99), color='green', linestyle='--', label='99th %')
        axes[1, 2].set_title('Nucleolin Intensity Distribution')
        axes[1, 2].legend()
        axes[1, 2].set_xlabel('Intensity')
        axes[1, 2].set_ylabel('Frequency')

        plt.tight_layout()
        plot_qc_figure(fig, filepath, conditions[nd2_files.index(filepath)], image_name, "01_raw_images")
        plt.close(fig)

    # Normalize nucleolin to [0, 1]
    nucleolin_norm = exposure.rescale_intensity(nucleolin.astype(np.float32), out_range=(0, 1))

    # DAPI signal preparation processes
    # Step 1: Segment nuclei
    dapi_for_seg = exposure.equalize_adapthist(
        exposure.rescale_intensity(dapi.astype(np.float32), out_range=(0, 1)),
        clip_limit=0.03
    )
    dapi_sharp = unsharp_mask(dapi_for_seg, radius=2, amount=1.0)

    dapi_thresh = filters.threshold_otsu(dapi_sharp)
    dapi_binary = dapi_sharp > dapi_thresh
    dapi_binary = morphology.remove_small_objects(dapi_binary, min_size=50)
    dapi_binary = morphology.binary_opening(dapi_binary, morphology.disk(1))
    dapi_binary = morphology.binary_closing(dapi_binary, morphology.disk(2))

    # === QC PLOT 2: DAPI PROCESSING ===
    if qc_plot:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'QC: DAPI Processing - {image_name}', fontsize=16)

        axes[0, 0].imshow(dapi_for_seg, cmap='gray')
        axes[0, 0].set_title('Adaptive Hist Eq')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(dapi_sharp, cmap='gray')
        axes[0, 1].set_title('Sharpened')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(dapi_binary, cmap='gray')
        axes[0, 2].set_title(f'Binary (Otsu: {dapi_thresh:.3f})')
        axes[0, 2].axis('off')

        axes[0, 3].hist(dapi_sharp.ravel(), bins=100, alpha=0.7, color='blue')
        axes[0, 3].axvline(dapi_thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold: {dapi_thresh:.3f}')
        axes[0, 3].set_title('Sharpened DAPI Histogram')
        axes[0, 3].legend()
        axes[0, 3].set_xlabel('Intensity')
        axes[0, 3].set_ylabel('Frequency')

        # Watershed visualization
        distance = ndi.distance_transform_edt(dapi_binary)
        axes[1, 0].imshow(distance, cmap='hot')
        axes[1, 0].set_title('Distance Transform')
        axes[1, 0].axis('off')

        from scipy.ndimage import maximum_filter
        footprint = morphology.disk(5)
        local_max = maximum_filter(distance, footprint=footprint) == distance
        local_max = local_max & dapi_binary

        axes[1, 1].imshow(dapi, cmap='gray', vmax=np.percentile(dapi, 99))
        axes[1, 1].imshow(local_max, cmap='Reds', alpha=0.5)
        axes[1, 1].set_title(f'Local Maxima: {np.sum(local_max)}')
        axes[1, 1].axis('off')

        if not np.any(local_max):
            nuclei_labels = measure.label(dapi_binary)
            axes[1, 2].imshow(nuclei_labels, cmap='tab20c')
            axes[1, 2].set_title('Simple Labeling')
        else:
            markers = measure.label(local_max)
            nuclei_labels = segmentation.watershed(-distance, markers, mask=dapi_binary)
            axes[1, 2].imshow(markers, cmap='tab20c')
            axes[1, 2].set_title(f'Watershed Markers: {len(np.unique(markers)) - 1}')

        axes[1, 2].axis('off')

        axes[1, 3].imshow(nuclei_labels, cmap='tab20c')
        axes[1, 3].set_title(f'Final Labels: {len(np.unique(nuclei_labels)) - 1}')
        axes[1, 3].axis('off')

        plt.tight_layout()
        plot_qc_figure(fig, filepath, conditions[nd2_files.index(filepath)], image_name, "02_dapi_processing")
        plt.close(fig)
    else:
        # Process without QC plots
        distance = ndi.distance_transform_edt(dapi_binary)
        from scipy.ndimage import maximum_filter
        footprint = morphology.disk(5)
        local_max = maximum_filter(distance, footprint=footprint) == distance
        local_max = local_max & dapi_binary

        if not np.any(local_max):
            nuclei_labels = measure.label(dapi_binary)
        else:
            markers = measure.label(local_max)
            nuclei_labels = segmentation.watershed(-distance, markers, mask=dapi_binary)

    # Step 2: Filter live nuclei
    nucleus_props = measure.regionprops_table(
        nuclei_labels,
        intensity_image=dapi,
        properties=('label', 'area', 'mean_intensity', 'bbox')
    )
    nucleus_df = pd.DataFrame(nucleus_props)

    if nucleus_df.empty:
        print(f"  No nuclei detected in {image_name}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  Initial nuclei detected: {len(nucleus_df)}")

    # 2 step filter for nuclei
    dapi_90th = np.percentile(nucleus_df['mean_intensity'], MAX_DAPI_INTENSITY_PERCENTILE)
    nucleus_df_filtered = nucleus_df[
        (nucleus_df['area'] <= MAX_NUCLEUS_AREA_PIXELS) &
        (nucleus_df['mean_intensity'] <= dapi_90th)
        ].copy()

    print(f"  Nuclei after filtering: {len(nucleus_df_filtered)}")
    print(f"    Removed {len(nucleus_df) - len(nucleus_df_filtered)} nuclei:")
    print(f"    - Area > {MAX_NUCLEUS_AREA_PIXELS:.0f} px: {sum(nucleus_df['area'] > MAX_NUCLEUS_AREA_PIXELS)}")
    print(f"    - Intensity > {dapi_90th:.1f}: {sum(nucleus_df['mean_intensity'] > dapi_90th)}")

    if nucleus_df_filtered.empty:
        print(f"  No valid nuclei after filtering in {image_name}")
        return pd.DataFrame(), pd.DataFrame()

    good_labels = set(nucleus_df_filtered['label'])
    nuclei_labels_filtered = np.where(np.isin(nuclei_labels, list(good_labels)), nuclei_labels, 0)
    orig_labels = nucleus_df_filtered['label'].tolist()
    num_nuclei = len(orig_labels)
    print(f"    Kept {num_nuclei} high-quality nuclei (live cells)")

    # === QC PLOT 3: NUCLEI FILTERING ===
    if qc_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'QC: Nuclei Filtering - {image_name}', fontsize=16)

        # Show all detected nuclei
        axes[0, 0].imshow(dapi, cmap='gray', vmax=np.percentile(dapi, 99))
        axes[0, 0].set_title(f'All Detected Nuclei: {len(nucleus_df)}')
        axes[0, 0].axis('off')

        # Show filtered nuclei
        axes[0, 1].imshow(dapi, cmap='gray', vmax=np.percentile(dapi, 99))
        axes[0, 1].imshow(nuclei_labels_filtered > 0, cmap='Reds', alpha=0.3)
        axes[0, 1].set_title(f'Filtered Nuclei: {num_nuclei}')
        axes[0, 1].axis('off')

        # Area vs Intensity scatter
        axes[0, 2].scatter(nucleus_df['area'], nucleus_df['mean_intensity'], alpha=0.6, label='All')
        axes[0, 2].scatter(nucleus_df_filtered['area'], nucleus_df_filtered['mean_intensity'],
                           alpha=0.8, color='red', label='Kept')
        axes[0, 2].axhline(dapi_90th, color='green', linestyle='--', label=f'90th %: {dapi_90th:.1f}')
        axes[0, 2].axvline(MAX_NUCLEUS_AREA_PIXELS, color='orange', linestyle='--',
                           label=f'Max area: {MAX_NUCLEUS_AREA_PIXELS:.0f}')
        axes[0, 2].set_xlabel('Area (pixels)')
        axes[0, 2].set_ylabel('Mean Intensity')
        axes[0, 2].set_title('Nuclei Filtering Criteria')
        axes[0, 2].legend()

        # Show rejected nuclei
        rejected_mask = np.isin(nuclei_labels, nucleus_df[~nucleus_df['label'].isin(good_labels)]['label'])
        axes[1, 0].imshow(dapi, cmap='gray', vmax=np.percentile(dapi, 99))
        axes[1, 0].imshow(rejected_mask, cmap='Blues', alpha=0.3)
        axes[1, 0].set_title(f'Rejected Nuclei: {len(nucleus_df) - num_nuclei}')
        axes[1, 0].axis('off')

        # Labeled nuclei
        axes[1, 1].imshow(nuclei_labels_filtered, cmap='tab20c')
        axes[1, 1].set_title(f'Filtered Labels: {len(np.unique(nuclei_labels_filtered)) - 1}')
        axes[1, 1].axis('off')

        # Area distribution
        axes[1, 2].hist([nucleus_df['area'], nucleus_df_filtered['area']],
                        bins=30, alpha=0.7, label=['All', 'Kept'], color=['gray', 'red'])
        axes[1, 2].axvline(MAX_NUCLEUS_AREA_PIXELS, color='orange', linestyle='--')
        axes[1, 2].set_xlabel('Area (pixels)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Nucleus Area Distribution')
        axes[1, 2].legend()

        plt.tight_layout()
        plot_qc_figure(fig, filepath, conditions[nd2_files.index(filepath)], image_name, "03_nuclei_filtering")
        plt.close(fig)

    # Step 3: Detect nucleolin aggregates
    selem = morphology.disk(3)
    nucleolin_tophat = morphology.white_tophat(nucleolin_norm, selem)
    agg_binary = nucleolin_tophat > AGGREGATE_INTENSITY_THRESHOLD
    agg_binary = morphology.remove_small_objects(agg_binary, min_size=5)
    agg_labels = measure.label(agg_binary)
    agg_regions = measure.regionprops(agg_labels, intensity_image=nucleolin_norm)

    agg_list = []
    for reg in agg_regions:
        if reg.area <= MAX_AGGREGATE_AREA_PIXELS:
            agg_list.append({
                'label': reg.label,
                'area': reg.area,
                'mean_intensity': reg.mean_intensity,
                'centroid_y': reg.centroid[0],
                'centroid_x': reg.centroid[1],
                'bbox': reg.bbox
            })
    agg_df = pd.DataFrame(agg_list) if agg_list else pd.DataFrame(
        columns=['label', 'area', 'mean_intensity', 'centroid_y', 'centroid_x', 'bbox'])

    print(f"    Found {len(agg_df)} aggregates (size & intensity filtered)")

    # === QC PLOT 4: AGGREGATE DETECTION ===
    if qc_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'QC: Aggregate Detection - {image_name}', fontsize=16)

        # Nucleolin normalized
        axes[0, 0].imshow(nucleolin_norm, cmap='gray', vmax=np.percentile(nucleolin_norm, 99))
        axes[0, 0].set_title('Normalized Nucleolin')
        axes[0, 0].axis('off')

        # Top-hat filtered
        axes[0, 1].imshow(nucleolin_tophat, cmap='gray', vmax=np.percentile(nucleolin_tophat, 99))
        axes[0, 1].set_title(f'Top-hat Filtered')
        axes[0, 1].axis('off')

        # Aggregate binary
        axes[0, 2].imshow(agg_binary, cmap='gray')
        axes[0, 2].set_title(f'Aggregate Binary (thresh={AGGREGATE_INTENSITY_THRESHOLD})')
        axes[0, 2].axis('off')

        # Overlay aggregates on nucleolin
        axes[1, 0].imshow(nucleolin_norm, cmap='gray', vmax=np.percentile(nucleolin_norm, 99))
        axes[1, 0].imshow(agg_binary, cmap='Reds', alpha=0.3)
        axes[1, 0].set_title(f'Detected Aggregates: {len(agg_df)}')
        axes[1, 0].axis('off')

        # Aggregate intensity distribution
        if not agg_df.empty:
            axes[1, 1].hist(agg_df['mean_intensity'], bins=30, alpha=0.7, color='red')
            axes[1, 1].axvline(AGGREGATE_INTENSITY_THRESHOLD, color='black', linestyle='--',
                               label=f'Threshold: {AGGREGATE_INTENSITY_THRESHOLD}')
            axes[1, 1].set_xlabel('Aggregate Intensity')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Aggregate Intensity Distribution')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No Aggregates', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')

        # Aggregate size distribution
        if not agg_df.empty:
            axes[1, 2].hist(agg_df['area'], bins=30, alpha=0.7, color='red')
            axes[1, 2].axvline(MAX_AGGREGATE_AREA_PIXELS, color='black', linestyle='--',
                               label=f'Max: {MAX_AGGREGATE_AREA_PIXELS:.1f} px')
            axes[1, 2].set_xlabel('Aggregate Area (pixels)')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].set_title('Aggregate Size Distribution')
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, 'No Aggregates', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].axis('off')

        plt.tight_layout()
        plot_qc_figure(fig, filepath, conditions[nd2_files.index(filepath)], image_name, "04_aggregate_detection")
        plt.close(fig)

    # Step 4: Assign aggregates to nuclei
    label_to_idx = {lab: (i + 1) for i, lab in enumerate(orig_labels)}
    nucleus_has_aggregate = np.zeros(num_nuclei + 1, dtype=bool)
    aggregate_count_per_nucleus = np.zeros(num_nuclei + 1, dtype=int)
    total_agg_intensity_per_nucleus = np.zeros(num_nuclei + 1, dtype=float)
    total_agg_area_per_nucleus = np.zeros(num_nuclei + 1, dtype=float)

    # Dictionary to store which aggregates belong to which nucleus for QC
    nucleus_aggregate_map = {i: [] for i in range(1, num_nuclei + 1)}

    for _, agg in agg_df.iterrows():
        try:
            cy_raw = agg['centroid_y']
            cx_raw = agg['centroid_x']
            if not (np.isfinite(cy_raw) and np.isfinite(cx_raw)):
                continue
            cy = int(round(float(cy_raw)))
            cx = int(round(float(cx_raw)))
        except (ValueError, TypeError, OverflowError):
            continue

        if cy < 0 or cy >= nuclei_labels_filtered.shape[0] or cx < 0 or cx >= nuclei_labels_filtered.shape[1]:
            continue

        original_label = nuclei_labels_filtered[cy, cx]
        if original_label != 0 and original_label in label_to_idx:
            idx = label_to_idx[original_label]
            nucleus_has_aggregate[idx] = True
            aggregate_count_per_nucleus[idx] += 1
            total_agg_intensity_per_nucleus[idx] += agg['mean_intensity']
            total_agg_area_per_nucleus[idx] += agg['area']
            nucleus_aggregate_map[idx].append({
                'centroid': (cy, cx),
                'area': agg['area'],
                'intensity': agg['mean_intensity']
            })

    # Step 5: Build dataframe for nucleus
    nucleus_final = pd.DataFrame({
        'orig_label': orig_labels,
        'nuclear_area_pixels': [nucleus_df_filtered[nucleus_df_filtered['label'] == lab]['area'].iloc[0] for lab in
                                orig_labels],
        'has_aggregate': [nucleus_has_aggregate[i] for i in range(1, num_nuclei + 1)],
        'aggregate_count': [aggregate_count_per_nucleus[i] for i in range(1, num_nuclei + 1)]
    })

    # Map nucleolin intensity
    nucleolin_props = measure.regionprops_table(
        nuclei_labels_filtered,
        intensity_image=nucleolin_norm,
        properties=('label', 'mean_intensity')
    )
    nucleolin_df = pd.DataFrame(nucleolin_props)
    intensity_map = dict(zip(nucleolin_df['label'], nucleolin_df['mean_intensity']))
    nucleus_final['mean_intensity'] = [intensity_map.get(lab, 0.0) for lab in orig_labels]

    # Total aggregate intensity
    nucleus_final['total_aggregate_intensity'] = [
        total_agg_intensity_per_nucleus[i] for i in range(1, num_nuclei + 1)
    ]
    nucleus_final['agg_nuc_ratio'] = nucleus_final['total_aggregate_intensity'] / np.maximum(
        nucleus_final['mean_intensity'], 1e-6)

    # Aggregate-to-nuclear AREA ratio
    nucleus_final['total_aggregate_area_pixels'] = [
        total_agg_area_per_nucleus[i] for i in range(1, num_nuclei + 1)
    ]
    nucleus_final['nuclear_area_um2'] = nucleus_final['nuclear_area_pixels'] * (PIXEL_SIZE_UM ** 2)
    nucleus_final['total_aggregate_area_um2'] = nucleus_final['total_aggregate_area_pixels'] * (PIXEL_SIZE_UM ** 2)
    nucleus_final['agg_nuc_area_ratio'] = nucleus_final['total_aggregate_area_um2'] / np.maximum(
        nucleus_final['nuclear_area_um2'], 1e-6)

    nucleus_final['is_live'] = True
    nucleus_final['is_nucleolin_positive'] = nucleus_final['mean_intensity'] > NUCLEOLIN_POSITIVITY_THRESHOLD

    print(
        f"  Aggregate-positive nuclei: {sum(nucleus_has_aggregate)}/{num_nuclei} ({sum(nucleus_has_aggregate) / max(num_nuclei, 1) * 100:.1f}%)")
    print(f"  Total aggregates: {sum(aggregate_count_per_nucleus)}")

    # === QC PLOT 5: FINAL RESULTS OVERLAY ===
    if qc_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'QC: Final Results - {image_name}', fontsize=16)

        # DAPI with nucleus outlines and aggregates
        axes[0, 0].imshow(dapi, cmap='gray', vmax=np.percentile(dapi, 99))

        # Draw nucleus outlines
        for region in measure.regionprops(nuclei_labels_filtered):
            if region.label in good_labels:
                y, x = region.centroid
                axes[0, 0].plot(x, y, 'b+', markersize=5)

        # Draw aggregates
        for idx in range(1, num_nuclei + 1):
            if nucleus_has_aggregate[idx]:
                for agg in nucleus_aggregate_map[idx]:
                    cy, cx = agg['centroid']
                    radius = np.sqrt(agg['area'] / np.pi)
                    circle = Circle((cx, cy), radius, fill=False, color='red', linewidth=1.5)
                    axes[0, 0].add_patch(circle)
                    axes[0, 0].plot(cx, cy, 'ro', markersize=3)

        axes[0, 0].set_title(f'DAPI with {num_nuclei} nuclei, {sum(nucleus_has_aggregate)} with aggregates')
        axes[0, 0].axis('off')

        # Nucleolin with aggregates
        axes[0, 1].imshow(nucleolin_norm, cmap='gray', vmax=np.percentile(nucleolin_norm, 99))

        # Draw aggregates on nucleolin
        for idx in range(1, num_nuclei + 1):
            if nucleus_has_aggregate[idx]:
                for agg in nucleus_aggregate_map[idx]:
                    cy, cx = agg['centroid']
                    radius = np.sqrt(agg['area'] / np.pi)
                    circle = Circle((cx, cy), radius, fill=False, color='yellow', linewidth=1.5)
                    axes[0, 1].add_patch(circle)
                    axes[0, 1].plot(cx, cy, 'yo', markersize=3)

        axes[0, 1].set_title(f'Nucleolin with {len(agg_df)} aggregates')
        axes[0, 1].axis('off')

        # Merge view (DAPI blue, Nucleolin red, Aggregates yellow)
        from skimage.color import label2rgb
        dapi_norm = exposure.rescale_intensity(dapi.astype(np.float32), out_range=(0, 1))
        nucleolin_norm_scaled = exposure.rescale_intensity(nucleolin_norm.astype(np.float32), out_range=(0, 1))

        rgb = np.zeros((dapi.shape[0], dapi.shape[1], 3))
        rgb[:, :, 0] = nucleolin_norm_scaled  # Red channel for nucleolin
        rgb[:, :, 2] = dapi_norm  # Blue channel for DAPI

        axes[1, 0].imshow(rgb)

        # Draw aggregates as yellow circles
        for idx in range(1, num_nuclei + 1):
            if nucleus_has_aggregate[idx]:
                for agg in nucleus_aggregate_map[idx]:
                    cy, cx = agg['centroid']
                    radius = np.sqrt(agg['area'] / np.pi)
                    circle = Circle((cx, cy), radius, fill=False, color='yellow', linewidth=1.5)
                    axes[1, 0].add_patch(circle)
                    axes[1, 0].plot(cx, cy, 'yo', markersize=3)

        axes[1, 0].set_title('Merge: DAPI(blue), Nucleolin(red), Aggregates(yellow)')
        axes[1, 0].axis('off')

        # Statistics
        axes[1, 1].axis('off')
        stats_text = f"""
        Image Statistics:
        -----------------
        Total Nuclei: {num_nuclei}
        Nucleolin Positive: {sum(nucleus_final['is_nucleolin_positive'])}
        Nuclei with Aggregates: {sum(nucleus_has_aggregate)}
        Total Aggregates: {sum(aggregate_count_per_nucleus)}

        Aggregate Stats:
        -----------------
        Mean per +ve nucleus: {sum(aggregate_count_per_nucleus) / max(sum(nucleus_has_aggregate), 1):.2f}
        Max per nucleus: {max(aggregate_count_per_nucleus) if num_nuclei > 0 else 0}
        Min per nucleus: {min(aggregate_count_per_nucleus[1:]) if num_nuclei > 0 else 0}

        Intensity Stats:
        -----------------
        Mean Nucleolin Intensity: {np.mean(nucleus_final['mean_intensity']):.3f}
        Mean Aggregate Intensity: {np.mean(agg_df['mean_intensity']) if not agg_df.empty else 0:.3f}
        """
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plot_qc_figure(fig, filepath, conditions[nd2_files.index(filepath)], image_name, "05_final_results")
        plt.close(fig)

    # Prepare aggregate output metadata
    aggregate_output = pd.DataFrame()
    if not agg_df.empty:
        agg_df['aggregate_area_um2'] = agg_df['area'] * (PIXEL_SIZE_UM ** 2)
        agg_df['image'] = os.path.basename(filepath)
        filename = os.path.basename(filepath)
        if "C9" in filename or "c9" in filename:
            cond = "C9"
        elif "ISO" in filename or "Iso" in filename or "Isogenic" in filename:
            cond = "Isogenic"
        else:
            cond = "Unknown"
        agg_df['condition'] = cond
        aggregate_output = agg_df[['image', 'condition', 'aggregate_area_um2', 'mean_intensity']].copy()

    return nucleus_final, aggregate_output


# Process all files with QC sampling
all_results = []
all_aggregates = []
qc_counter = 0

print(f"\n{'=' * 60}")
print("STARTING ANALYSIS")
print(f"{'=' * 60}")

for i, (f, cond) in enumerate(zip(nd2_files, conditions)):
    # Determine if we should generate QC plots for this image
    # Always do QC for first image, then sample based on QC_SAMPLE_FRACTION
    if i == 0:
        qc_plot = QC_MODE
    else:
        qc_plot = QC_MODE and (np.random.random() < QC_SAMPLE_FRACTION)

    print(f"\nProcessing {cond}: {f}")
    if qc_plot:
        print("  ✓ Generating QC plots")
        qc_counter += 1

    try:
        df_nuclei, df_agg = analyze_nd2_file(f, qc_plot=qc_plot)
        if df_nuclei.empty:
            print("  → No valid nuclei after filtering")
        else:
            df_nuclei['image'] = os.path.basename(f)
            df_nuclei['condition'] = cond
            all_results.append(df_nuclei)
            print(f"  → {len(df_nuclei)} live nuclei analyzed")

        if not df_agg.empty:
            all_aggregates.append(df_agg)

    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        continue

print(f"\n{'=' * 60}")
print(f"QC SUMMARY: Generated QC plots for {qc_counter}/{len(nd2_files)} images")
print(f"{'=' * 60}")

if not all_results:
    raise RuntimeError("No valid results collected.")

df_all = pd.concat(all_results, ignore_index=True)
print(f"\nTotal live nuclei analyzed: {len(df_all)}")

# Rest of the analysis code remains the same...
# [Keep all the existing analysis code from the previous version]
# Only the analyze_nd2_file function and initial setup were modified

# ============================================================================
# The rest of the analysis code remains exactly the same as your original
# ============================================================================

# Sanity check
df_all['has_agg_derived'] = df_all['aggregate_count'] > 0
mismatch = df_all[df_all['has_aggregate'] != df_all['has_agg_derived']]
if not mismatch.empty:
    print("❌ CRITICAL: Mismatch between 'has_aggregate' and 'aggregate_count > 0'")
    df_all['has_aggregate'] = df_all['has_agg_derived']
    print(" Dictate 'has_aggregate' to match 'aggregate_count > 0'")

agg_pos = df_all[df_all['aggregate_count'] > 0]
if not agg_pos.empty:
    min_count = agg_pos['aggregate_count'].min()
    if min_count < 1:
        print(f"❌ IMPOSSIBLE: minimum aggregate_count among positive nuclei is {min_count}")
    else:
        print(f"All aggregate+ nuclei have ≥{min_count} aggregates")

print("Conditions in dataset:", df_all['condition'].unique())


# Image-level Metrics
def compute_image_metrics(g):
    total_nuclei = len(g)
    nucleolin_positive = g['is_nucleolin_positive'].sum()
    cells_with_aggregates = (g['aggregate_count'] > 0).sum()
    total_aggregates = g['aggregate_count'].sum()
    return pd.Series({
        'total_nuclei': total_nuclei,
        'nucleolin_positive_cells': nucleolin_positive,
        'cells_with_aggregates': cells_with_aggregates,
        'total_aggregates': total_aggregates
    })


image_metrics = df_all.groupby(['image', 'condition']).apply(compute_image_metrics).reset_index()

# Population level: Aggregates per Aggregate+ Nucleus
print("\n=== POPULATION-LEVEL: Aggregates per Aggregate+ Nucleus (MUST BE ≥1) ===")
for cond in ['C9', 'Isogenic']:
    cond_data = df_all[df_all['condition'] == cond]
    agg_positive = cond_data[cond_data['aggregate_count'] > 0]
    if len(agg_positive) == 0:
        mean_per_nuc = 0.0
    else:
        total_agg = agg_positive['aggregate_count'].sum()
        num_nuc = len(agg_positive)
        mean_per_nuc = total_agg / num_nuc
        assert mean_per_nuc >= 1.0, f"Physically impossible result for {cond}: {mean_per_nuc}"
    print(f"{cond}: {mean_per_nuc:.2f} aggregates per aggregate+ nucleus "
          f"({agg_positive['aggregate_count'].sum()} aggregates / {len(agg_positive)} nuclei)")

# per-image averages
print("\n=== SUMMARY (Per-Image Averages) ===")
for cond in ['C9', 'Isogenic']:
    data = image_metrics[image_metrics['condition'] == cond]
    if data.empty:
        print(f"\n{cond}: No data")
        continue
    print(f"\n{cond} (n={len(data)} images):")
    print(f"  Avg live cells/image: {data['total_nuclei'].mean():.1f} ± {data['total_nuclei'].std():.1f}")
    print(
        f"  Avg nucleolin-positive cells: {data['nucleolin_positive_cells'].mean():.1f} ± {data['nucleolin_positive_cells'].std():.1f}")
    print(
        f"  Avg cells with aggregates: {data['cells_with_aggregates'].mean():.1f} ± {data['cells_with_aggregates'].std():.1f}")
    print(f"  Avg total aggregates/image: {data['total_aggregates'].mean():.1f} ± {data['total_aggregates'].std():.1f}")

# Perform cell level matrix
c9_data = image_metrics[image_metrics['condition'] == 'C9']
iso_data = image_metrics[image_metrics['condition'] == 'Isogenic']

metrics_to_test = ['nucleolin_positive_cells', 'cells_with_aggregates', 'total_aggregates']
print("\n=== STATISTICAL COMPARISON (C9 vs Isogenic) ===")
for metric in metrics_to_test:
    c9_vals = c9_data[metric]
    iso_vals = iso_data[metric]
    if len(c9_vals) >= 3 and len(iso_vals) >= 3:
        u_stat, p_val = mannwhitneyu(c9_vals, iso_vals, alternative='two-sided')
        d_nucleolin_positive_cells = cliffs_delta(c9_vals, iso_vals)
        d_nucleolin_positive_cells = abs(d_nucleolin_positive_cells)
        print(f"nucleolin positive cells comparison statistics: {d_nucleolin_positive_cells:.3f} ")
        print(f"{metric}: U={u_stat:.1f}, p={p_val:.4f}")
    else:
        print(f"{metric}: insufficient data for test")

# Plot everything serially, Start from the baseline image below, 4 in one
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
plot_metrics = ['nucleolin_positive_cells', 'cells_with_aggregates', 'total_aggregates']
plot_titles = ['Nucleolin-Positive Cells', 'Cells with Aggregates', 'Total Aggregates']

# Define the order for all plots
condition_order = ['Isogenic', 'C9']

for ax, metric, title in zip(axes.flat[:3], plot_metrics, plot_titles):
    # Add order parameter to both boxplot and stripplot
    sns.boxplot(data=image_metrics, x='condition', y=metric, ax=ax,
                palette=['lightgreen', 'salmon'], order=condition_order)
    sns.stripplot(data=image_metrics, x='condition', y=metric, ax=ax,
                  color='black', size=4, order=condition_order)
    ax.set_title(title)
    ax.set_xlabel("")

# Aggregates per nucleus (population-level) - Reverse the order
axes.flat[3].bar(['Isogenic', 'C9'],  # Changed order here
                 [df_all[df_all['condition'] == 'Isogenic']['aggregate_count'].replace(0, np.nan).mean(),
                  df_all[df_all['condition'] == 'C9']['aggregate_count'].replace(0, np.nan).mean()],
                 color=['lightgreen', 'salmon'])  # Adjusted colors to match new order
axes.flat[3].set_title('Aggregates per Aggregate+ Nucleus\n(Population-Level)')
axes.flat[3].set_ylabel('Mean aggregates')

plt.tight_layout()
plt.savefig('nucleolin_aggregate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot everything serially, Start from the baseline image below, 4 in one
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
plot_metrics = ['nucleolin_positive_cells', 'cells_with_aggregates', 'total_aggregates']
plot_titles = ['Nucleolin-Positive Cells', 'Cells with Aggregates', 'Total Aggregates']

# Define the order for all plots
condition_order = ['Isogenic', 'C9']

for ax, metric, title in zip(axes.flat[:3], plot_metrics, plot_titles):
    # Add order parameter to both boxplot and stripplot
    sns.boxplot(data=image_metrics, x='condition', y=metric, ax=ax,
                palette=['lightgreen', 'salmon'], order=condition_order)
    sns.stripplot(data=image_metrics, x='condition', y=metric, ax=ax,
                  color='black', size=4, order=condition_order)
    ax.set_title(title)
    ax.set_xlabel("")

# Aggregates per nucleus (population-level) - Reverse the order
axes.flat[3].bar(['Isogenic', 'C9'],  # Changed order here
                 [df_all[df_all['condition'] == 'Isogenic']['aggregate_count'].replace(0, np.nan).mean(),
                  df_all[df_all['condition'] == 'C9']['aggregate_count'].replace(0, np.nan).mean()],
                 color=['lightgreen', 'salmon'])  # Adjusted colors to match new order
axes.flat[3].set_title('Aggregates per Aggregate+ Nucleus\n(Population-Level)')
axes.flat[3].set_ylabel('Mean aggregates')

plt.tight_layout()
plt.savefig('nucleolin_aggregate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Fraction nucleolin-positive
image_metrics['frac_nucleolin_positive'] = (
        image_metrics['nucleolin_positive_cells'] / image_metrics['total_nuclei']
)

plt.figure(figsize=(6, 5))
sns.boxplot(data=image_metrics, x='condition', y='frac_nucleolin_positive',
            palette=['lightgreen', 'salmon'], order=condition_order)
sns.stripplot(data=image_metrics, x='condition', y='frac_nucleolin_positive',
              color='black', size=4, order=condition_order)
plt.ylabel("Fraction of Nucleolin-Positive Nuclei")
plt.title("Nucleolar Stress: Fraction of Nucleolin-Positive Cells")
plt.tight_layout()
plt.savefig('nucleolin_fraction_positive.png', dpi=300, bbox_inches='tight')
plt.show()


# Fraction, nucleolin aggregate positive
image_metrics['frac_cells_with_aggregates'] = (
        image_metrics['cells_with_aggregates'] / image_metrics['total_nuclei']
)

# Fraction of cells with aggregates plot
plt.figure(figsize=(6, 5))
sns.boxplot(data=image_metrics, x='condition', y='frac_cells_with_aggregates',
            palette=['lightgreen', 'salmon'], order=condition_order)
sns.stripplot(data=image_metrics, x='condition', y='frac_cells_with_aggregates',
              color='black', size=4, order=condition_order)
plt.ylabel("Fraction of Cells with Aggregates")
plt.title("Nucleolar Stress: Fraction of Cells Containing Nucleolin Aggregates")
plt.tight_layout()
plt.savefig('fraction_cells_with_aggregates.png', dpi=300, bbox_inches='tight')
plt.show()

c9_frac = image_metrics[image_metrics['condition'] == 'C9']['frac_nucleolin_positive']
iso_frac = image_metrics[image_metrics['condition'] == 'Isogenic']['frac_nucleolin_positive']
if len(c9_frac) >= 3 and len(iso_frac) >= 3:
    u_stat, p_val = mannwhitneyu(c9_frac, iso_frac, alternative='two-sided')
    d_frac = cliffs_delta(c9_frac, iso_frac)
    d_frac = abs(d_frac)
    print(f" fraction of nucleolin positive Cliff's d = {d_frac:.1f}")
    print(f"Fraction nucleolin-positive: U={u_stat:.1f}, p={p_val:.4f}")
else:
    print("Fraction nucleolin-positive: insufficient data for test")

# Aggregate count distribution
agg_df_plot = df_all[df_all['aggregate_count'] > 0]
if not agg_df_plot.empty:
    max_aggs = int(agg_df_plot['aggregate_count'].max()) + 1
    bins = np.arange(0.5, max_aggs + 1.5, 1.0)

    plt.figure(figsize=(6, 5))
    sns.histplot(
        data=agg_df_plot,
        x='aggregate_count',
        hue='condition',
        element='step',
        stat='density',
        common_norm=False,
        bins=bins,
        hue_order=condition_order
    )
    plt.xlabel("Aggregates per Nucleus")
    plt.ylabel("Density")
    plt.title("Distribution of Aggregates per Nucleus")
    plt.xticks(range(1, max_aggs + 1))
    plt.tight_layout()
    plt.savefig('aggregate_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Analysis of Aggregate size
# Always define aggregates_df
if all_aggregates:
    aggregates_df = pd.concat(all_aggregates, ignore_index=True)
    print(f"Total aggregates collected: {len(aggregates_df)}")
else:
    aggregates_df = pd.DataFrame()  # ← Safe empty DataFrame
    print("No aggregates detected in any image.")

# Now safely check before using it
if not aggregates_df.empty and 'mean_intensity' in aggregates_df.columns:
    # ✅ AGGREGATE INTENSITY: STATISTICS + RATIO
    c9_agg_int = aggregates_df[aggregates_df['condition'] == 'C9']['mean_intensity']
    iso_agg_int = aggregates_df[aggregates_df['condition'] == 'Isogenic']['mean_intensity']

    if len(c9_agg_int) >= 3 and len(iso_agg_int) >= 3:
        u_stat_int, p_val_int = mannwhitneyu(c9_agg_int, iso_agg_int, alternative='two-sided')
        median_c9 = np.median(c9_agg_int)
        median_iso = np.median(iso_agg_int)
        ratio_median = median_c9 / median_iso if median_iso != 0 else np.nan
        mean_c9 = np.mean(c9_agg_int)
        mean_iso = np.mean(iso_agg_int)
        ratio_mean = mean_c9 / mean_iso if mean_iso != 0 else np.nan
        d_agg_int = cliffs_delta(c9_agg_int, iso_agg_int)
        d_agg_int = abs(d_agg_int)
        print(f" aggregate intensity Cliff's d = {d_agg_int:.3f}")
        print(f"\nAggregate intensity:")
        print(f"  C9 median: {median_c9:.3f}, Isogenic median: {median_iso:.3f}")
        print(f"  Median ratio (C9/Isogenic): {ratio_median:.2f}")
        print(f"  Mean ratio (C9/Isogenic): {ratio_mean:.2f}")
        print(f"  Mann-Whitney U={u_stat_int:.1f}, p={p_val_int:.4f}")
    else:
        print("\nAggregate intensity: insufficient data for test")

    # Mean aggregate size per image
    if 'aggregate_area_um2' in aggregates_df.columns:
        agg_size_per_image = (
            aggregates_df.groupby(['image', 'condition'])['aggregate_area_um2']
            .mean()
            .reset_index(name='mean_aggregate_size')
        )

        # Check if agg_size_per_image is not empty and has valid numeric data
        if not agg_size_per_image.empty and 'mean_aggregate_size' in agg_size_per_image.columns:
            # Drop any NaN values
            agg_size_per_image = agg_size_per_image.dropna(subset=['mean_aggregate_size'])

            if not agg_size_per_image.empty:
                # Check which conditions are actually present in the data
                available_conditions = agg_size_per_image['condition'].unique()

                if len(available_conditions) > 0:
                    plt.figure(figsize=(6, 5))

                    # Determine the order for plotting
                    plot_order = []
                    color_palette = []

                    # Always put Isogenic first if it exists
                    if 'Isogenic' in available_conditions:
                        plot_order.append('Isogenic')
                        color_palette.append('lightgreen')

                    if 'C9' in available_conditions:
                        plot_order.append('C9')
                        color_palette.append('salmon')

                    # Plot only if we have valid order and palette
                    if plot_order and color_palette:
                        # Use hue parameter if we have multiple conditions
                        if len(plot_order) > 1:
                            sns.boxplot(data=agg_size_per_image, x='condition', y='mean_aggregate_size',
                                        palette=color_palette, order=plot_order)
                            sns.stripplot(data=agg_size_per_image, x='condition', y='mean_aggregate_size',
                                          color='black', size=4, order=plot_order)
                        else:
                            # Single condition - simpler plotting
                            sns.boxplot(data=agg_size_per_image, x='condition', y='mean_aggregate_size',
                                        color=color_palette[0])
                            sns.stripplot(data=agg_size_per_image, x='condition', y='mean_aggregate_size',
                                          color='black', size=4)

                        plt.ylabel("Mean Aggregate Size (µm²)")
                        plt.title("Aggregate Size per Image")
                        plt.tight_layout()
                        plt.savefig('aggregate_size_analysis.png', dpi=300, bbox_inches='tight')
                        plt.show()

                        # Only run stats if both conditions are present with enough data
                        if 'C9' in available_conditions and 'Isogenic' in available_conditions:
                            c9_vals = agg_size_per_image[agg_size_per_image['condition'] == 'C9'][
                                'mean_aggregate_size'].dropna()
                            iso_vals = agg_size_per_image[agg_size_per_image['condition'] == 'Isogenic'][
                                'mean_aggregate_size'].dropna()

                            if len(c9_vals) >= 3 and len(iso_vals) >= 3:
                                u_stat, p_val = mannwhitneyu(c9_vals, iso_vals, alternative='two-sided')
                                d_agg_size = cliffs_delta(c9_vals, iso_vals)
                                d_agg_size = abs(d_agg_size)
                                print(f" mean aggregate size Cliff's d = {d_agg_size:.3f}")
                                print(f"\nMean aggregate size: U={u_stat:.1f}, p={p_val:.4f}")
                            else:
                                print(
                                    f"\nMean aggregate size: insufficient data for test (C9: {len(c9_vals)}, Isogenic: {len(iso_vals)})")
                        else:
                            print(f"\nMean aggregate size: only one condition present ({available_conditions})")
                    else:
                        print("\nNo valid conditions for plotting aggregate size")
                else:
                    print("\nNo conditions available in aggregate size data")
            else:
                print("\nNo valid aggregate size data after cleaning")
        else:
            print("\nNo aggregate size data available")
    else:
        print("\nSkipping aggregate size analysis: 'aggregate_area_um2' column not found")

    # Aggregate/Nuclear Area ratio analysis
    print("\n=== AGGREGATE / NUCLEAR AREA RATIO ===")
    agg_positive_area = df_all[df_all['aggregate_count'] > 0].copy()

    if not agg_positive_area.empty:
        agg_positive_area = agg_positive_area.replace([np.inf, -np.inf], np.nan).dropna(subset=['agg_nuc_area_ratio'])

        if not agg_positive_area.empty and 'condition' in agg_positive_area.columns:
            # Check available conditions
            available_conditions = agg_positive_area['condition'].unique()

            if len(available_conditions) > 0:
                # Perform statistical test if both conditions exist
                c9_area_ratio = agg_positive_area[agg_positive_area['condition'] == 'C9']['agg_nuc_area_ratio']
                iso_area_ratio = agg_positive_area[agg_positive_area['condition'] == 'Isogenic']['agg_nuc_area_ratio']

                # Initialize variables
                p = None
                d_agg_nuc_area = None
                stats_text = ""

                if len(c9_area_ratio) >= 3 and len(iso_area_ratio) >= 3:
                    u, p = mannwhitneyu(c9_area_ratio, iso_area_ratio, alternative='two-sided')
                    d_agg_nuc_area = cliffs_delta(c9_area_ratio, iso_area_ratio)
                    d_agg_nuc_area = abs(d_agg_nuc_area)
                    print(f"\nAggregate/Nuclear Area Ratio:")
                    print(f"  C9 median: {np.median(c9_area_ratio):.4f}")
                    print(f"  Isogenic median: {np.median(iso_area_ratio):.4f}")
                    print(f" aggregate to nuclear area ratio Cliff's d = {d_agg_nuc_area:.3f}")
                    print(f"  Mann-Whitney U={u:.1f}, p={p:.4f}")
                else:
                    print("\nInsufficient data for area ratio test.")

                plt.figure(figsize=(6, 5))
                ax = plt.gca()  # Get current axis

                # Determine order for plotting
                plot_order = []
                color_palette = []

                if 'Isogenic' in available_conditions:
                    plot_order.append('Isogenic')
                    color_palette.append('lightgreen')

                if 'C9' in available_conditions:
                    plot_order.append('C9')
                    color_palette.append('salmon')

                if plot_order and color_palette:
                    if len(plot_order) > 1:
                        # Plot with seaborn
                        sns.boxplot(data=agg_positive_area, x='condition', y='agg_nuc_area_ratio',
                                    palette=color_palette, order=plot_order, ax=ax)
                        sns.stripplot(data=agg_positive_area, x='condition', y='agg_nuc_area_ratio',
                                      color='black', size=4, order=plot_order, ax=ax)

                        # Add significance annotation if we have p-value
                        if p is not None and d_agg_nuc_area is not None:
                            # Get y-axis limits for positioning
                            y_min, y_max = ax.get_ylim()
                            y_range = y_max - y_min

                            # Position for significance bar
                            bar_height = y_max + 0.02 * y_range
                            bar_y = bar_height
                            text_y = bar_y + 0.02 * y_range

                            # Draw significance bar
                            ax.plot([0, 0, 1, 1], [bar_y, bar_y + 0.01 * y_range, bar_y + 0.01 * y_range, bar_y],
                                    color='black', lw=1.5)

                            # Format p-value with stars
                            if p < 0.001:
                                p_stars = "***"
                                p_display = "p < 0.001"
                            elif p < 0.01:
                                p_stars = "**"
                                p_display = f"p = {p:.3f}"
                            elif p < 0.05:
                                p_stars = "*"
                                p_display = f"p = {p:.3f}"
                            else:
                                p_stars = "ns"
                                p_display = f"p = {p:.3f}"

                            # Add p-value text
                            ax.text(0.5, text_y, p_stars,
                                    ha='center', va='bottom', fontsize=12, fontweight='bold')

                            # Format Cliff's d magnitude
                            d_abs = abs(d_agg_nuc_area)
                            if d_abs < 0.147:
                                d_magnitude = "negligible"
                            elif d_abs < 0.33:
                                d_magnitude = "small"
                            elif d_abs < 0.474:
                                d_magnitude = "medium"
                            else:
                                d_magnitude = "large"

                            # Add statistics annotation in a box
                            stats_text = f"MW-U {p_display}\nCliff's d = {d_agg_nuc_area:.2f} ({d_magnitude})"
                            ax.text(0.95, 0.95, stats_text,
                                    transform=ax.transAxes,
                                    fontsize=9,
                                    verticalalignment='top',
                                    horizontalalignment='right',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

                            # Adjust y-limit to accommodate the significance bar
                            ax.set_ylim(y_min, y_max + 0.1 * y_range)

                    else:
                        # Single condition - simpler plotting
                        sns.boxplot(data=agg_positive_area, x='condition', y='agg_nuc_area_ratio',
                                    color=color_palette[0], ax=ax)
                        sns.stripplot(data=agg_positive_area, x='condition', y='agg_nuc_area_ratio',
                                      color='black', size=4, ax=ax)

                    plt.ylabel("Aggregate / Nuclear Area Ratio")
                    plt.title("Fraction of Nuclear Area Occupied by Aggregates")
                    plt.ylim(0, None)
                    plt.tight_layout()
                    plt.savefig('agg_nuc_area_ratio.png', dpi=300, bbox_inches='tight')
                    plt.show()
                else:
                    print("\nNo valid conditions for area ratio plot")
            else:
                print("\nNo conditions available for area ratio analysis")
        else:
            print("\nNo valid data for area ratio analysis after cleaning")
    else:
        print("\nNo aggregate-positive nuclei for area ratio analysis.")
c9_frac_agg = image_metrics[image_metrics['condition'] == 'C9']['frac_cells_with_aggregates']
iso_frac_agg = image_metrics[image_metrics['condition'] == 'Isogenic']['frac_cells_with_aggregates']

if len(c9_frac_agg) >= 3 and len(iso_frac_agg) >= 3:
    u_stat, p_val = mannwhitneyu(c9_frac_agg, iso_frac_agg, alternative='two-sided')
    d_cell_with_agg = cliffs_delta(c9_frac_agg, iso_frac_agg)
    d_cell_with_agg = abs(d_cell_with_agg)
    print(f" fraction of cells with aggregate Cliff's d = {d_cell_with_agg:.3f}")
    print(f"\nFraction of cells with aggregates: U={u_stat:.1f}, p={p_val:.4f}")
else:
    print("\nFraction of cells with aggregates: insufficient data for test")

# Final, save results
df_all.to_csv('nucleolin_per_nucleus.csv', index=False)
image_metrics.to_csv('image_level_nucleolin_metrics.csv', index=False)

# Create summary QC figure
if QC_MODE and all_results:
    print("\nGenerating summary QC figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QC Summary: Processing Pipeline', fontsize=16)

    # Example images from first analyzed file
    sample_image = nd2_files[0]
    with nd2.ND2File(sample_image) as ndfile:
        img_4d = ndfile.to_xarray().values
        dapi_sample = img_4d[:, 0, :, :].max(axis=0)
        nucleolin_sample = img_4d[:, 1, :, :].max(axis=0)

    axes[0, 0].imshow(dapi_sample, cmap='gray', vmax=np.percentile(dapi_sample, 99))
    axes[0, 0].set_title('1. Raw DAPI')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(nucleolin_sample, cmap='gray', vmax=np.percentile(nucleolin_sample, 99))
    axes[0, 1].set_title('2. Raw Nucleolin')
    axes[0, 1].axis('off')

    # Process a small region for demonstration
    sample_region = dapi_sample[100:300, 100:300]
    dapi_for_seg_sample = exposure.equalize_adapthist(
        exposure.rescale_intensity(sample_region.astype(np.float32), out_range=(0, 1)),
        clip_limit=0.03
    )
    axes[0, 2].imshow(dapi_for_seg_sample, cmap='gray')
    axes[0, 2].set_title('3. Processed DAPI')
    axes[0, 2].axis('off')

    # Binary threshold example
    dapi_thresh_sample = filters.threshold_otsu(dapi_for_seg_sample)
    binary_sample = dapi_for_seg_sample > dapi_thresh_sample
    axes[1, 0].imshow(binary_sample, cmap='gray')
    axes[1, 0].set_title('4. Binary Mask')
    axes[1, 0].axis('off')

    # Aggregate detection example
    nucleolin_norm_sample = exposure.rescale_intensity(nucleolin_sample[100:300, 100:300], out_range=(0, 1))
    nucleolin_tophat_sample = morphology.white_tophat(nucleolin_norm_sample, morphology.disk(3))
    agg_binary_sample = nucleolin_tophat_sample > AGGREGATE_INTENSITY_THRESHOLD
    axes[1, 1].imshow(nucleolin_norm_sample, cmap='gray', vmax=np.percentile(nucleolin_norm_sample, 99))
    axes[1, 1].imshow(agg_binary_sample, cmap='Reds', alpha=0.5)
    axes[1, 1].set_title('5. Aggregate Detection')
    axes[1, 1].axis('off')

    # Final overlay
    rgb_sample = np.zeros((sample_region.shape[0], sample_region.shape[1], 3))
    rgb_sample[:, :, 0] = nucleolin_norm_sample
    rgb_sample[:, :, 2] = exposure.rescale_intensity(sample_region.astype(np.float32), out_range=(0, 1))
    axes[1, 2].imshow(rgb_sample)
    axes[1, 2].imshow(agg_binary_sample, cmap='Reds', alpha=0.5)
    axes[1, 2].set_title('6. Final Overlay\nDAPI(blue), Nucleolin(red)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('QC_Figures/QC_Processing_Pipeline_Summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

print("\n✅ Analysis complete. Results saved.")
# =============================================================================
# MAIN COMPARISON PLOTS: Aggregate Intensity, Nucleolin Intensity, Nuclear Size

# 2. NUCLEOLIN INTENSITY (per nucleus, all nuclei)
if not df_all.empty and 'mean_intensity' in df_all.columns:
    # Check which conditions exist in df_all
    available_conditions = df_all['condition'].unique()

    if len(available_conditions) > 0:
        plt.figure(figsize=(6, 5))

        # Determine plotting order and palette
        plot_order = []
        color_palette = []

        if 'Isogenic' in available_conditions:
            plot_order.append('Isogenic')
            color_palette.append('lightgreen')

        if 'C9' in available_conditions:
            plot_order.append('C9')
            color_palette.append('salmon')

        # Plot based on available conditions
        if len(plot_order) > 1:
            sns.boxplot(data=df_all, x='condition', y='mean_intensity',
                        palette=color_palette, width=0.6, order=plot_order)
            sns.stripplot(data=df_all, x='condition', y='mean_intensity',
                          color='black', size=3, alpha=0.5, order=plot_order)
        else:
            sns.boxplot(data=df_all, x='condition', y='mean_intensity',
                        color=color_palette[0], width=0.6)
            sns.stripplot(data=df_all, x='condition', y='mean_intensity',
                          color='black', size=3, alpha=0.5)

        plt.ylabel("Nucleolin Intensity per Nucleus (A.U.)", fontsize=12)
        plt.xlabel("Condition", fontsize=12)
        plt.title("Nucleolin Intensity: C9 vs Isogenic", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig('main_plot_nucleolin_intensity.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("⚠ Skipping Nucleolin Intensity plot: no conditions found in data")
else:
    print("⚠ Skipping Nucleolin Intensity plot: df_all missing or no 'mean_intensity' column")

# 3. NUCLEAR SIZE (per nucleus, in µm²)
if not df_all.empty and 'nuclear_area_um2' in df_all.columns:
    # Check which conditions exist in df_all
    available_conditions = df_all['condition'].unique()

    if len(available_conditions) > 0:
        plt.figure(figsize=(6, 5))

        # Determine plotting order and palette
        plot_order = []
        color_palette = []

        if 'Isogenic' in available_conditions:
            plot_order.append('Isogenic')
            color_palette.append('lightgreen')

        if 'C9' in available_conditions:
            plot_order.append('C9')
            color_palette.append('salmon')

        # Plot based on available conditions
        if len(plot_order) > 1:
            sns.boxplot(data=df_all, x='condition', y='nuclear_area_um2',
                        palette=color_palette, width=0.6, order=plot_order)
            sns.stripplot(data=df_all, x='condition', y='nuclear_area_um2',
                          color='black', size=3, alpha=0.5, order=plot_order)
        else:
            sns.boxplot(data=df_all, x='condition', y='nuclear_area_um2',
                        color=color_palette[0], width=0.6)
            sns.stripplot(data=df_all, x='condition', y='nuclear_area_um2',
                          color='black', size=3, alpha=0.5)

        plt.ylabel("Nuclear Area (µm²)", fontsize=12)
        plt.xlabel("Condition", fontsize=12)
        plt.title("Nuclear Size: C9 vs Isogenic", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig('main_plot_nuclear_size.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("⚠ Skipping Nuclear Size plot: no conditions found in data")
else:
    print("⚠ Skipping Nuclear Size plot: 'nuclear_area_um2' column missing")

print("\n✅ Main comparison plots saved:")
print("   - main_plot_aggregate_intensity.png")
print("   - main_plot_nucleolin_intensity.png")
print("   - main_plot_nuclear_size.png")

# =============================================================================
# NEW: Mann-Whitney U Tests for Nuclear Size and Nucleolin Intensity
# =============================================================================

# 1. NUCLEAR SIZE COMPARISON (C9 vs Isogenic)
print("\n=== NUCLEAR SIZE COMPARISON (C9 vs Isogenic) ===")
if 'nuclear_area_um2' in df_all.columns and 'condition' in df_all.columns:
    # Check which conditions actually exist
    available_conditions = df_all['condition'].unique()

    if 'C9' in available_conditions and 'Isogenic' in available_conditions:
        c9_nuc_size = pd.to_numeric(df_all[df_all['condition'] == 'C9']['nuclear_area_um2'], errors='coerce').dropna()
        iso_nuc_size = pd.to_numeric(df_all[df_all['condition'] == 'Isogenic']['nuclear_area_um2'],
                                     errors='coerce').dropna()

        if len(c9_nuc_size) >= 3 and len(iso_nuc_size) >= 3:
            u_size, p_size = mannwhitneyu(c9_nuc_size, iso_nuc_size, alternative='two-sided')
            d_nuclear_size = cliffs_delta(c9_nuc_size, iso_nuc_size)
            d_nuclear_size = abs(d_nuclear_size)
            print(f"Nuclear size (µm²):")
            print(f"  C9 median: {c9_nuc_size.median():.2f}, Isogenic median: {iso_nuc_size.median():.2f}")
            print(f"  Mann-Whitney U={u_size:.1f}, p={p_size:.4f}")
            print(f" nuclear area Cliff's d = {d_nuclear_size:.3f}")

            # Plot
            plt.figure(figsize=(6, 5))
            ax = sns.boxplot(data=df_all, x='condition', y='nuclear_area_um2',
                             palette=['lightgreen', 'salmon'], order=['Isogenic', 'C9'])
            sns.stripplot(data=df_all, x='condition', y='nuclear_area_um2',
                          color='black', size=4, alpha=0.6, order=['Isogenic', 'C9'])
            plt.ylabel("Nuclear Area (µm²)")
            plt.title("Nuclear Size: C9 vs Isogenic")

            # Add significance bars with p-value and Cliff's d
            # Get y-axis limits for positioning
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min

            # Position for significance bar
            bar_height = y_max + 0.02 * y_range
            bar_y = bar_height
            text_y = bar_y + 0.02 * y_range

            # Draw significance bar
            ax.plot([0, 0, 1, 1], [bar_y, bar_y + 0.01 * y_range, bar_y + 0.01 * y_range, bar_y],
                    color='black', lw=1.5)

            # Format p-value with stars
            if p_size < 0.001:
                p_stars = "***"
                p_display = "p < 0.001"
            elif p_size < 0.01:
                p_stars = "**"
                p_display = f"p = {p_size:.3f}"
            elif p_size < 0.05:
                p_stars = "*"
                p_display = f"p = {p_size:.3f}"
            else:
                p_stars = "ns"
                p_display = f"p = {p_size:.3f}"

            # Add p-value text
            ax.text(0.5, text_y, p_stars,
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

            # Format Cliff's d magnitude
            d_abs = abs(d_nuclear_size)
            if d_abs < 0.147:
                d_magnitude = "negligible"
            elif d_abs < 0.33:
                d_magnitude = "small"
            elif d_abs < 0.474:
                d_magnitude = "medium"
            else:
                d_magnitude = "large"

            # Add statistics annotation in a box
            stats_text = f"MW-U {p_display}\nCliff's d = {d_nuclear_size:.2f} ({d_magnitude})"
            ax.text(0.95, 0.95, stats_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

            # Adjust y-limit to accommodate the significance bar
            ax.set_ylim(y_min, y_max + 0.1 * y_range)

            plt.tight_layout()
            plt.savefig('nuclear_size_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Nuclear size: insufficient data for test")
    else:
        print(f"Nuclear size: missing one or both conditions (available: {available_conditions})")
else:
    print("Nuclear size: required columns not found")

# 2. NUCLEOLIN INTENSITY COMPARISON (C9 vs Isogenic)
print("\n=== NUCLEOLIN INTENSITY COMPARISON (C9 vs Isogenic) ===")
if 'mean_intensity' in df_all.columns and 'condition' in df_all.columns:
    c9_nuc_int = pd.to_numeric(df_all[df_all['condition'] == 'C9']['mean_intensity'], errors='coerce').dropna()
    iso_nuc_int = pd.to_numeric(df_all[df_all['condition'] == 'Isogenic']['mean_intensity'], errors='coerce').dropna()

    if len(c9_nuc_int) >= 3 and len(iso_nuc_int) >= 3:
        u_int, p_int = mannwhitneyu(c9_nuc_int, iso_nuc_int, alternative='two-sided')
        d_nucleolin_int = cliffs_delta(c9_nuc_int, iso_nuc_int)
        d_nucleolin_int = abs(d_nucleolin_int)
        print(f"  C9 median: {c9_nuc_int.median():.3f}, Isogenic median: {iso_nuc_int.median():.3f}")
        print(f"  Mann-Whitney U={u_int:.1f}, p={p_int:.4f}")
        print(f"Nucleolin intensity per nucleus Cliff's d:{d_nucleolin_int:.3f}")

        # Plot
        plt.figure(figsize=(6, 5))
        ax = sns.boxplot(data=df_all, x='condition', y='mean_intensity',
                         palette=['lightgreen', 'salmon'], order=['Isogenic', 'C9'])
        sns.stripplot(data=df_all, x='condition', y='mean_intensity',
                      color='black', size=4, alpha=0.6, order=['Isogenic', 'C9'])
        plt.ylabel("Mean Nucleolin Intensity per Nucleus (A.U.)")
        plt.title("Nucleolin Intensity: C9 vs Isogenic")

        # Add significance bars with p-value and Cliff's d
        # Get y-axis limits for positioning
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        # Position for significance bar
        bar_height = y_max + 0.02 * y_range
        bar_y = bar_height
        text_y = bar_y + 0.02 * y_range

        # Draw significance bar
        ax.plot([0, 0, 1, 1], [bar_y, bar_y + 0.01 * y_range, bar_y + 0.01 * y_range, bar_y],
                color='black', lw=1.5)

        # Format p-value with stars
        if p_int < 0.001:
            p_stars = "***"
            p_display = "p < 0.001"
        elif p_int < 0.01:
            p_stars = "**"
            p_display = f"p = {p_int:.3f}"
        elif p_int < 0.05:
            p_stars = "*"
            p_display = f"p = {p_int:.3f}"
        else:
            p_stars = "ns"
            p_display = f"p = {p_int:.3f}"

        # Add p-value text
        ax.text(0.5, text_y, p_stars,
                ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Add statistics annotation in a box
        stats_text = f"MW-U {p_display}\nCliff's d = {d_nucleolin_int:.2f}"
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

        # Adjust y-limit to accommodate the significance bar
        ax.set_ylim(y_min, y_max + 0.1 * y_range)

        plt.tight_layout()
        plt.savefig('nucleolin_intensity_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("Nucleolin intensity: insufficient data for test")
else:
    print("Nucleolin intensity: column 'mean_intensity' not found")

# final remark
print("Alessio eats Worms for fun every afternoon!! 🐛🤓")