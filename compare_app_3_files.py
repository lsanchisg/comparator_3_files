import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page to wide mode
st.set_page_config(layout="wide")
st.title("🔬 2D Colormap 3-Way Comparison Tool")
st.write("For Mesh Convergence Analysis")

# --- 1. File Definitions ---
# Define the files and their labels
FILES_TO_LOAD = {
    '0.045': 'EXPORT_tbl2_0_045.csv',
    '0.05': 'EXPORT_tbl2_0_05.csv',
    '0.07': 'EXPORT_tbl2_0_07.csv'
}

# --- 2. Data Loading and Caching ---
@st.cache_data
def load_data(file_dict):
    """
    Loads, merges, and processes data from multiple CSV files.
    """
    dfs = {}
    base_params = None
    
    try:
        for label, filename in file_dict.items():
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip()
            
            # Get parameter columns (all except axes)
            params = [col for col in df.columns if col not in ['h_fib', 'lda0']]
            
            # Set base_params from the first file loaded
            if base_params is None:
                base_params = sorted(params) # e.g., ['Absorptance', 'Reflectance_port_1', ...]
            
            # Rename columns to include the file label
            # e.g., 'Reflectance_port_1' -> 'Reflectance_port_1_0.045'
            rename_dict = {p: f"{p}_{label}" for p in params}
            dfs[label] = df.rename(columns=rename_dict)
            
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Make sure all CSV files are in the same directory as the app.")
        return None, [], None, None
    except Exception as e:
        st.error(f"An error occurred during file loading: {e}")
        return None, [], None, None

    # Merge all dataframes
    df_merged = None
    for df in dfs.values():
        if df_merged is None:
            df_merged = df
        else:
            df_merged = pd.merge(df_merged, df, on=['h_fib', 'lda0'])
    
    # Calculate difference columns
    for p in base_params:
        # p is now the full name, e.g., 'Reflectance_port_1'
        p_045 = f"{p}_0.045" # 'Reflectance_port_1_0.045'
        p_05 = f"{p}_0.05"   # 'Reflectance_port_1_0.05'
        p_07 = f"{p}_0.07"   # 'Reflectance_port_1_0.07'
        
        # Check if all columns exist before calculating diffs
        if all(col in df_merged.columns for col in [p_045, p_05, p_07]):
            # Pairwise differences
            df_merged[f'{p}_Diff_05_vs_045'] = (df_merged[p_05] - df_merged[p_045]).abs()
            df_merged[f'{p}_Diff_07_vs_045'] = (df_merged[p_07] - df_merged[p_045]).abs()
            df_merged[f'{p}_Diff_07_vs_05'] = (df_merged[p_07] - df_merged[p_05]).abs()
            
            # Max-Min Range
            df_merged[f'{p}_Range'] = df_merged[[p_045, p_05, p_07]].max(axis=1) - df_merged[[p_045, p_05, p_07]].min(axis=1)
    
    # Get axis values
    x_vals = sorted(df_merged['lda0'].unique())
    y_vals = sorted(df_merged['h_fib'].unique())
    
    return df_merged, base_params, x_vals, y_vals

# Load data
df, params, x_vals, y_vals = load_data(FILES_TO_LOAD)

if df is not None:
    st.sidebar.info(f"""
    **Files Loaded:**
    - `{FILES_TO_LOAD['0.045']}` (0.045)
    - `{FILES_TO_LOAD['0.05']}` (0.05)
    - `{FILES_TO_LOAD['0.07']}` (0.07)
    """)

    # --- 3. Sidebar Controls ---
    # The 'params' list now correctly contains the full names
    selected_param = st.sidebar.selectbox("Select Parameter to Plot:", params)
    
    display_mode = st.sidebar.radio(
        "Select Heatmap Display:",
        (
            "File 1 (0.045)", "File 2 (0.05)", "File 3 (0.07)", 
            "Diff: 0.05 vs 0.045", "Diff: 0.07 vs 0.045", "Diff: 0.07 vs 0.05",
            "Max-Min Range"
        ),
        index=6  # Default to "Max-Min Range"
    )
    
    st.sidebar.markdown("---")
    
    # --- 4. Sliders for Cross-Sections ---
    y_slider_val = st.sidebar.slider(
        "Fiber Height (h_fib) Cross-Section",
        min_value=min(y_vals),
        max_value=max(y_vals),
        value=y_vals[len(y_vals) // 2],
        step=(y_vals[1] - y_vals[0]) if len(y_vals) > 1 else 0.0
    )
    
    x_slider_val = st.sidebar.slider(
        "Wavelength (lda0) Cross-Section",
        min_value=min(x_vals),
        max_value=max(x_vals),
        value=x_vals[len(x_vals) // 2],
        step=(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 0.0
    )

    # Find the closest index in our data for the slider values
    y_slider_idx = (np.abs(np.array(y_vals) - y_slider_val)).argmin()
    x_slider_idx = (np.abs(np.array(x_vals) - x_slider_val)).argmin()

    # --- 5. Data Pivoting ---
    @st.cache_data
    def pivot_data(_df, value_col): # Changed df to _df to avoid shadowing
        try:
            return _df.pivot(index='h_fib', columns='lda0', values=value_col).values
        except Exception as e:
            # Provide a more informative error message
            st.error(f"Error pivoting data for column '{value_col}': {e}")
            if value_col not in _df.columns:
                st.error(f"Column '{value_col}' was not found in the DataFrame.")
            return np.zeros((len(y_vals), len(x_vals)))

    # Pivot all necessary data
    # 'selected_param' is now 'Reflectance_port_1' (or similar)
    z_data_1 = pivot_data(df, f"{selected_param}_0.045")
    z_data_2 = pivot_data(df, f"{selected_param}_0.05")
    z_data_3 = pivot_data(df, f"{selected_param}_0.07")
    
    z_data_diff_05_045 = pivot_data(df, f"{selected_param}_Diff_05_vs_045")
    z_data_diff_07_045 = pivot_data(df, f"{selected_param}_Diff_07_vs_045")
    z_data_diff_07_05 = pivot_data(df, f"{selected_param}_Diff_07_vs_05")
    z_data_range = pivot_data(df, f"{selected_param}_Range")

    # --- 6. Plotting ---
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        specs=[
            [{"type": "xy"}, {}],  # Top plot (row 1, col 1)
            [{"type": "heatmap"}, {"type": "xy"}]  # Main (2,1), Right (2,2)
        ],
        horizontal_spacing=0.01,
        vertical_spacing=0.01
    )

    # --- Plot 1: Top Cross-Section (vs. lda0) ---
    fig.add_trace(go.Scatter(
        x=x_vals, y=z_data_1[y_slider_idx, :], name='Mesh 0.045',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_vals, y=z_data_2[y_slider_idx, :], name='Mesh 0.05',
        line=dict(color='red', dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_vals, y=z_data_3[y_slider_idx, :], name='Mesh 0.07',
        line=dict(color='green', dash='dashdot')
    ), row=1, col=1)

    # --- Plot 2: Main Heatmap ---
    if display_mode == "File 1 (0.045)":
        z_data, colorscale, heatmap_title = z_data_1, 'Viridis', f"{selected_param} (Mesh 0.045)"
    elif display_mode == "File 2 (0.05)":
        z_data, colorscale, heatmap_title = z_data_2, 'Viridis', f"{selected_param} (Mesh 0.05)"
    elif display_mode == "File 3 (0.07)":
        z_data, colorscale, heatmap_title = z_data_3, 'Viridis', f"{selected_param} (Mesh 0.07)"
    elif display_mode == "Diff: 0.05 vs 0.045":
        z_data, colorscale, heatmap_title = z_data_diff_05_045, 'Reds', "Abs. Diff (0.05 vs 0.045)"
    elif display_mode == "Diff: 0.07 vs 0.045":
        z_data, colorscale, heatmap_title = z_data_diff_07_045, 'Reds', "Abs. Diff (0.07 vs 0.045)"
    elif display_mode == "Diff: 0.07 vs 0.05":
        z_data, colorscale, heatmap_title = z_data_diff_07_05, 'Reds', "Abs. Diff (0.07 vs 0.05)"
    else: # "Max-Min Range"
        z_data, colorscale, heatmap_title = z_data_range, 'Hot', "Max-Min Range (Convergence)"

    fig.add_trace(go.Heatmap(
        x=x_vals, y=y_vals, z=z_data,
        colorscale=colorscale,
        colorbar_title=selected_param,
        zmin=0 if "Diff" in display_mode or "Range" in display_mode else None,
        zmax=z_data_range.max() if "Diff" in display_mode or "Range" in display_mode else (z_data_1.max() if "File" in display_mode else None) # Better auto-range for diffs
    ), row=2, col=1)
    
    # Add slider lines to heatmap
    fig.add_trace(go.Scatter(
        x=x_vals, y=[y_slider_val] * len(x_vals),
        mode='lines', line=dict(color='white', dash='dash', width=1), name='h_fib cut'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=[x_slider_val] * len(y_vals), y=y_vals,
        mode='lines', line=dict(color='white', dash='dash', width=1), name='lda0 cut'
    ), row=2, col=1)

    # --- Plot 3: Right Cross-Section (vs. h_fib) ---
    fig.add_trace(go.Scatter(
        y=y_vals, x=z_data_1[:, x_slider_idx], name='Mesh 0.045',
        line=dict(color='blue'), showlegend=False
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        y=y_vals, x=z_data_2[:, x_slider_idx], name='Mesh 0.05',
        line=dict(color='red', dash='dot'), showlegend=False
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        y=y_vals, x=z_data_3[:, x_slider_idx], name='Mesh 0.07',
        line=dict(color='green', dash='dashdot'), showlegend=False
    ), row=2, col=2)

    # --- 7. Layout Updates ---
    fig.update_layout(
        height=700,
        title_text=heatmap_title,
        xaxis1=dict(showticklabels=False),
        yaxis1=dict(title=selected_param),
        xaxis2=dict(title='lda0 (Wavelength)'),
        yaxis2=dict(title='h_fib (Fiber Height)'),
        xaxis3=dict(title=selected_param),
        yaxis3=dict(showticklabels=False),
        xaxis1_matches='x2',
        yaxis3_matches='y2',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
