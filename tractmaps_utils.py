######### UTILITY FUNCTIONS FOR TRACTMAPS #########
import os
import hcp_utils as hcp
import nilearn.plotting as plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats as sstats
from statistics import mean
from PIL import Image
from neuromaps import nulls, images
from neuromaps.datasets import fetch_annotation
from neuromaps.parcellate import Parcellater
import nibabel as nib
from matplotlib.gridspec import GridSpec


### Load a list of cortical maps from Neuromaps ###
def load_surface_maps(maps_list = None, maps_names = None):
    
    """
    Load surface maps into a dictionary.

    Parameters:
        maps_list (list, optional): A list of loaded surface maps. Default is None (default list of maps is loaded from a csv).
        maps_names (list, optional): A list of map names corresponding to the loaded maps. Default is None (default list of map names is loaded from a csv).

    Returns:
        dict: A dictionary where keys are map names and values are the loaded maps based on the list.

    Description:
    This function loads surface maps into a dictionary. By default, it loads maps from a local CSV file (./inputs/selected_neuromaps.csv), 
    using 'annotation' as the map list and 'name' as the map names. Alternatively, the user can provide custom lists for maps_list and maps_names. 
    The function verifies that the provided lists have matching lengths and returns a dictionary with map names as keys and the corresponding loaded maps as values.
    
    """
    
    #initalize dict to store maps
    neuromap_dict = {}
    
    if maps_list is not None:
         # Check if the user-provided list is a list of tuples with 5 elements
        if not all(isinstance(row, tuple) and len(row) == 4 for row in maps_list):
            print("Error: User-provided list should be a list of tuples with 4 elements each.")
            selected_maps = []
            
        # Check that maps_names exists
        if maps_names is None:
            print("Error: when maps_list is provided, corresponding maps_names must be provided as well for each brain map.")
            return None
        else:
            # Check if user-provided arguments are lists and have matching lengths
            if not isinstance(maps_list, list) or not isinstance(maps_names, list) or not len(maps_list) == len(maps_names):
                print("Error: maps_list and maps_names must be lists and must have the same length.")
                return None
    
    # if no maps list provided, load neuromaps csv
    else:
        
        # Load the default CSV into a DataFrame
        all_neuromaps = pd.read_csv('./inputs/selected_neuromaps.csv')
        
        # If maps_names are provided, load them from the csv
        if maps_names is not None:
            all_neuromaps[all_neuromaps['use'] == 1]['annotation']
            
            # Filter the DataFrame based on 'name' column matching maps_names
            selected_maps = all_neuromaps.loc[all_neuromaps['name'].isin(maps_names)]
            
            maps_list = selected_maps['annotation'].apply(lambda x: tuple(word.strip("()''") for word in x.split(', '))).tolist()
            maps_names = selected_maps['name'].tolist()
        
        # if no maps list and map names are provided, load default selected maps from csv
        else:
    
            # Filter rows based on the 'use' column (where 1 means we definitely want to use the map); save as a list of tuples
            maps_list = all_neuromaps[all_neuromaps['use'] == 1]['annotation'].apply(lambda x: tuple(word.strip("()''") for word in x.split(', '))).tolist()
            
            # use this filter to include additional maps (defined as == 2, i.e. "maybe")
            # selected_maps = all_neuromaps[all_neuromaps['use'].isin([1, 2])][['name', 'annotation']].apply(lambda x: tuple(word.strip("()''") for word in x.split(', '))).tolist()
            
            # select map names
            maps_names = all_neuromaps[all_neuromaps['use'] == 1]['name'].tolist()      
        
    # Generate a dictionary with maps_names as keys and maps_list as values
    annotations_dict = dict(zip(maps_names, maps_list))

    # fetch maps
    for map_name, annotation in annotations_dict.items():
        print(f'Map: {map_name}, Annotation: {annotation}')
        
        # fetch the map using Neuromaps
        fetched_map = fetch_annotation(source = annotation[0], 
                                       desc = annotation[1],
                                       space = annotation[2], 
                                       den = annotation[3])
        
        # store fetched maps and corresponding names in results dictionary
        neuromap_dict[map_name] = {'annotation': annotation, 
                                   'map': fetched_map}
        
    return neuromap_dict

### Get Glasser parcellation ###
def get_glasser(lh_glasser = None, rh_glasser = None):
    """
    Generate GIFTI objects for the left and right hemispheres with optional custom input data.

    This function creates GIFTI objects for the left and right hemispheres using the Glasser atlas labels and data. GIFTI objects are outputted as a tuple. 
    
    Args:
        lh_glasser (optional): GIFTI object for the left hemisphere. If provided, custom data will be used for the left hemisphere.
        rh_glasser (optional): GIFTI object for the right hemisphere. If provided, custom data will be used for the right hemisphere.
    
    Returns:
        tuple: A tuple containing GIFTI objects for the left and right hemispheres.

    If custom GIFTI objects are not provided, the function generates GIFTI objects based on the Glasser atlas labels and data.

    Example Usage:
        glasser = get_glasser()
    """
    
    # select region labels
    labels = list(hcp.mmp.labels.values())

    # redefine the medial wall label to match one of the elements in PARCIGNORE (required for generating spin samples in the alexander_bloch nulls function)
    # PARCIGNORE = ['unknown', 'corpuscallosum', 'Background+FreeSurfer_Defined_Medial_Wall','???', 'Unknown', 'Medial_wall', 'Medial wall', 'medial_wall']
    labels[0] = 'medial_wall'

    # select left and right hemisphere labels
    lh_labels = labels[0:181] # the first 0-180 are unassigned (0) and 1-180 are lh labels
    rh_labels = [labels[0]]  + labels[181:361] # the first (0) unassigned label and 181-360 are rh labels
    
    # if left and right data are not provided, use the hcp glasser data
    if lh_glasser is None:
        # Transform array of glasser labels in grayordinates (hcp.mmp.map_all, size=59412) to labels in vertices (size=32492). 
        # The unused vertices are filled with a constant (zero by default). Order is Left, Right hemisphere.
        lh_glasser_verts = hcp.left_cortex_data(hcp.mmp.map_all)
        
        # create gifti
        lh_glasser = images.construct_shape_gii(lh_glasser_verts, labels = lh_labels,
                                           intent = 'NIFTI_INTENT_LABEL')
        
    if rh_glasser is None:
        rh_glasser_verts = hcp.right_cortex_data(hcp.mmp.map_all)
        rh_glasser = images.construct_shape_gii(rh_glasser_verts, labels = rh_labels,
                                       intent = 'NIFTI_INTENT_LABEL')

    # if needed, update GIFTI images so label IDs are consecutive across hemispheres
    # glasser = images.relabel_gifti((lh_glasser, rh_glasser))
    
    # create tuple of left and right hemisphere GIFTI images
    glasser = (lh_glasser, rh_glasser)
    
    return glasser

### Apply Glasser parcellation to cortical maps ###
def glasserize(cortical_maps, zscore = True):
    """
    Parcellate cortical maps using the Glasser parcellation and store the parcellated maps in a dictionary.

    Args:
        cortical_maps (dict): A dictionary of cortical maps to be parcellated.
        zscore (bool, optional): Whether to z-score the cortical maps. Defaults to True.

    Returns:
        dict: A dictionary containing the parcellated maps.
    """
    
    # get glasser object (a tuple containing Glasser labels as GIFTI objects for the left and right hemispheres)
    glasser = get_glasser()
    
    # create parcellater object
    glasser_parc = Parcellater(parcellation = glasser,
                               space = 'fsLR', # space in which the parcellation is defined
                               resampling_target = 'parcellation') # cortical maps provided later will be resampled to the space + resolution of the data, if needed
    # glasser_parc.parcellation[0].darrays[0] #.data

    
    # Dictionary to store parcellated maps
    glasser_maps = {}  

    for map_name, value in cortical_maps.items():
        print(f'Map: {map_name}')

        # Apply Glasser parcellation to the map
        parcellated_map = glasser_parc.fit_transform(data=value['map'], space=value['annotation'][2])

        # Z-score the map if zscore is True
        if zscore:
            parcellated_map = sstats.zscore(parcellated_map, nan_policy = 'omit')

        # Assign the parcellated map to the new dictionary
        glasser_maps[f'{map_name}_glasser'] = parcellated_map

        # Load non-parcellated maps to compute stats for comparison with parcellated maps
        data = images.load_data(value['map'])
        print(f'Original map shape: {data.shape}, parcellated shape: {parcellated_map.shape}')
#         print(f' Min original value: {np.min(data)}, min parcellated: {np.min(parcellated_map)}', 
#               f'\n Mean value: {np.mean(data)}, mean parcellated: {np.mean(parcellated_map)}', 
#               f'\n Max value: {np.max(data)}, max parcellated: {np.max(parcellated_map)}')  

    return glasser_maps


### Plot parcellated brain map ###
def plot_parc_map(brain_map, map_name, colors, hemisphere = None, mode = 'interactive'):
    """
    Plots parcellated regions of a `brain_map` using nilearn plotting functions
    
    Parameters
    ----------
    brain_map : (N,) array_like
        Surface data for N parcels
    map_name: str
        Name of the brain map that is being plotted (for plot title).
    colors: str, optional
        Colormap (from matplotlib default options) used to display brain regions. Default is 'Spectral'.
    hemisphere: str, optional
        For static plots, which hemisphere should be displayed. Options are 'left' or 'right'.
    mode: str, optional
        Whether the figure should be interactive or static. Default is 'interactive'.
    
    Returns
    -------
    plot : nilearn interactive or static plot showing map values for all brain parcels.
    
    """
    
    # subset hcp Glasser parcellation to only contain cortical regions (remove subcortical areas 361-380)
    hcp.mmp_short = hcp.mmp
    hcp.mmp_short.map_all = hcp.mmp.map_all[(hcp.mmp.map_all >= 0) & (hcp.mmp.map_all <= 360)]
    hcp.mmp_short.labels = subset_dict = {key: hcp.mmp.labels[key] for key in list(hcp.mmp.labels.keys())[:361]}
    hcp.mmp_short.ids = hcp.mmp.ids[0:361]
    hcp.mmp_short.ids = hcp.mmp.ids[0:361]
    hcp.mmp_short.nontrivial_ids = hcp.mmp.nontrivial_ids[0:361]
    hcp.mmp_short.rgba = subset_dict = {key: hcp.mmp.rgba[key] for key in list(hcp.mmp.rgba.keys())[:361]}
    
    # assign brain map values of each parcels to corresponding vertices on full brain surface
    unparcellated_map = hcp.unparcellate(brain_map, hcp.mmp_short)
    
    if mode == 'static' and hemisphere is None:
        raise ValueError("If mode is 'static', hemisphere must be provided.")
    
    if mode == 'static':
        
        # define brain meshes to plot on (left or right)            
        if hemisphere == 'left':
            brain_mesh = hcp.mesh.inflated_left
            background_im = hcp.mesh.sulc_left
            input_parc_map = hcp.left_cortex_data(unparcellated_map) # create full cortex array for plotting
            
        else: # right hemisphere
            brain_mesh = hcp.mesh.inflated_right
            background_im = hcp.mesh.sulc_right
            input_parc_map = hcp.right_cortex_data(unparcellated_map)
            
        # assign nan to medial wall
        input_parc_map[input_parc_map == 0]  = np.nan 

        # Create a figure with a grid of 3D subplots (for lateral and medial views)
        fig, axes = plt.subplots(1, 2, figsize = (7, 4), subplot_kw = {'projection': '3d'}) # 3d projection to correctly insert nilearn plots in grid

        # plot lateral regions
        plotting.plot_surf(brain_mesh,
                           input_parc_map,
                           cmap = colors,
#                            colorbar = True, # this gets placed on top of the brain surface, would need fixing if want to display colorbar
                           vmin = np.min(brain_map),
                           vmax = np.max(brain_map),
                           symmetric_cmap = False,
                           bg_map = background_im,
                           view = 'lateral',
                           axes=axes[0] # assign to left plot in grid
                          )

        # plot medial regions
        plotting.plot_surf(brain_mesh,
                           input_parc_map,
                           cmap = colors,
#                           colorbar = True,
                           vmin = np.min(brain_map),
                           vmax = np.max(brain_map),
                           symmetric_cmap = False,
                           bg_map = background_im,
                           view = 'medial',
                           axes = axes[1] # assign to right plot in grid
                          )

        # plot title
        plt.suptitle(f'{map_name} - Glasser 360 parcellation', fontsize = 13)

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.show()
        
    else: # interactive plot
    
        # create full cortex array for plotting
        input_parc_map = hcp.cortex_data(unparcellated_map) 

        # assign nan to medial wall
        input_parc_map[input_parc_map == 0] = np.nan 
        
        return plotting.view_surf(hcp.mesh.inflated,
                                  input_parc_map,
                                  cmap = colors,
                                  vmin = np.min(brain_map),
                                  vmax = np.max(brain_map),
                                  symmetric_cmap = False,
                                  bg_map = hcp.mesh.sulc,
                                  title = f'{map_name} - Glasser 360 parcellation',
                                  title_fontsize = 20)

### Plot subset of structurally connected regions for a given brain map and tract ###
def plot_parc_subset(brain_map, tract, map_name, tract_name, colors = 'Spectral', mode = 'interactive', connection_threshold = 0.95):
    
    """
    Plots a subset of parcellated regions in a `brain_map` using nilearn plotting functions

    Parameters
    ----------
    brain_map : (N,) array_like
        Surface data for N parcels
    tract: str
        Name of the tract for which structurally connected brain regions should be displayed. 
    map_name: str
        Name of the brain map that is being plotted (for plot title).
    tract_name: str
        Name of the tract name for which structurally connected brain regions are displayed (for plot title).
    colors: str, optional
        Colormap (from matplotlib default options) used to display brain regions. Default is 'Spectral'.
    mode: str, optional
        Whether the figure should be interactive or static. Default is 'interactive'.
    connection_threshold: str, optional
        Defines the connection threshold used to select brain regions structurally connected to the input tract. Default is 95% probability of connection (0.95).

    Returns
    -------
    plot : nilearn interactive or static plot showing map values for a subset of structurally connected brain parcels.
    
    """
    
    # subset hcp Glasser parcellation to only contain cortical regions (remove subcortical areas 361-380)
    hcp.mmp_short = hcp.mmp
    hcp.mmp_short.map_all = hcp.mmp.map_all[(hcp.mmp.map_all >= 0) & (hcp.mmp.map_all <= 360)]
    hcp.mmp_short.labels = subset_dict = {key: hcp.mmp.labels[key] for key in list(hcp.mmp.labels.keys())[:361]}
    hcp.mmp_short.ids = hcp.mmp.ids[0:361]
    hcp.mmp_short.ids = hcp.mmp.ids[0:361]
    hcp.mmp_short.nontrivial_ids = hcp.mmp.nontrivial_ids[0:361]
    hcp.mmp_short.rgba = subset_dict = {key: hcp.mmp.rgba[key] for key in list(hcp.mmp.rgba.keys())[:361]}

    # read in csv with tracts and Glasser region IDs
    tracts_regs_ids = pd.read_csv('./outputs/tracts_regs_Glasser.csv')

    # select brain regions structurally connected to the tract
    tract_df = tracts_regs_ids.loc[tracts_regs_ids[f'{tract}'] >= connection_threshold, ['parcel_name', 'regionLongName', 'cortex', 'regionID']]

    # define region IDs of structurally connected regions for the tract
    region_ids = np.array(tract_df['regionID'].astype('int'))

    # select region ids across full surface map vertices
    region_ids_all = np.isin(hcp.mmp_short.map_all, region_ids)

    # assign brain map values of each parcels to correspinding vertices on full brain surface
    unparcellated_map = hcp.unparcellate(brain_map, hcp.mmp_short)
    
    if 'left' in tract:
        hemisphere = 'left'
    elif 'right' in tract:
        hemisphere = 'right'
    else:
        raise ValueError("The tract must be either 'left' or 'right'.")
    
    # plot regions
    if mode == 'static': # static plot
        # define brain meshes to plot on (left or right)
        if hemisphere == 'left':
            brain_mesh = hcp.mesh.inflated_left
            background_im = hcp.mesh.sulc_left
            input_parc_map = hcp.left_cortex_data(hcp.mask(unparcellated_map, region_ids_all)) # plot a subset of region IDs
            
        elif hemisphere == 'right': # right hemisphere
            brain_mesh = hcp.mesh.inflated_right
            background_im = hcp.mesh.sulc_right
            input_parc_map = hcp.right_cortex_data(hcp.mask(unparcellated_map, region_ids_all)) # plot a subset of region IDs
            
        # assign nan to medial wall
        input_parc_map[input_parc_map == 0]  = np.nan 

        # Create a figure with a grid of 3D subplots (for lateral and medial views)
        fig, axes = plt.subplots(1, 2, figsize = (7, 4), subplot_kw = {'projection': '3d'}) # 3d projection to correctly insert nilearn plots in grid

        # plot lateral regions
        plotting.plot_surf(brain_mesh,
                           input_parc_map,
                           cmap = colors,
        #                    colorbar = True, # this gets placed on top of the brain surface, would need fixing if want to display colorbar
                           vmin = np.min(brain_map),
                           vmax = np.max(brain_map),
                           symmetric_cmap = False,
                           bg_map = background_im,
                           view = 'lateral',
                           axes=axes[0] # assign to left plot in grid
                          )

        # plot medial regions
        plotting.plot_surf(brain_mesh,
                           input_parc_map,
                           cmap = colors,
        #                   colorbar = True,
                           vmin = np.min(brain_map),
                           vmax = np.max(brain_map),
                           symmetric_cmap = False,
                           bg_map = background_im,
                           view = 'medial',
                           axes = axes[1] # assign to right plot in grid
                          )

        # plot title
        plt.suptitle(f'{map_name} in {len(tract_df)} regions structurally connected to the {tract_name}', fontsize = 13)

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.show()

    else: # interactive plot
        
        # plot a subset of region IDs (on the entire brain surface, both hemispheres)
        input_parc_map = hcp.cortex_data(hcp.mask(unparcellated_map, region_ids_all))

        # assign nan to medial wall
        input_parc_map[input_parc_map == 0]  = np.nan 

        # plotting interactive view
        return plotting.view_surf(hcp.mesh.inflated,
                                  input_parc_map,
                                  cmap = colors,
                                  vmin = np.min(brain_map),
                                  vmax = np.max(brain_map),
                                  symmetric_cmap = False,
                                  bg_map = hcp.mesh.sulc,
                                  title = f'{map_name} in {len(tract_df)} regions structurally connected to the {tract_name}',
                                  title_fontsize = 15
                                 )
    


##### Generate a custom heatmap #######
def generate_heatmap(df, brain_maps_col, tracts_col, result_value_col, p_value_col, significance_threshold = 0.05, cmap = 'coolwarm', title = None):
    
    # pivot the results dataframe
    
    pivot_df = df.pivot_table(index = brain_maps_col, columns = tracts_col, values = result_value_col)

    # Create a wider heatmap
    plt.figure(figsize=(20, 4))  # Adjust the width and height as needed

    # Create a mask to hide non-significant values
    mask = df.pivot_table(index = brain_maps_col, columns = tracts_col, values = p_value_col) > significance_threshold

    # define a custom colormap
    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    # Create the heatmap with empty white boxes for non-significant values
    sns.heatmap(pivot_df, mask = mask, cmap = colormap, linewidths = 0.5, linecolor = 'grey') # remove the mask if you want to see all values (including non-significant ones)
    
    # Add a title to the heatmap if provided
    if title:
        plt.title(title, fontsize = 16)
    
    # Show the heatmap
    plt.show()
    
    
### Run linear regression #####
def run_linear_regression(df, x, y, separate_by_group=False, group_column=None, plot=True):
    """
    Perform a linear regression on the DataFrame to test the effect of 'x' on 'y'.
    Optionally, run separate regressions based on a grouping variable.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - x (str): The name of the independent variable.
    - y (str): The name of the dependent variable.
    - separate_by_group (bool): If True, run separate regressions based on the 'group_column'.
                               If False (default), run a single regression for the entire dataset.
    - group_column (str or None): The name of the grouping variable (if separate_by_group is True).
    - plot (bool): If True (default), plot the relationship between 'x' and 'y' with regression lines.
                  If False, do not plot.

    Returns:
    - None
    """
    if separate_by_group:
        if group_column is None:
            raise ValueError("You must specify a 'group_column' when 'separate_by_group' is True.")

        # Get the unique values of the group_column
        unique_groups = df[group_column].unique()

        # Initialize a color palette for plotting
        palette = sns.color_palette("husl", n_colors=len(unique_groups))

        # Create a figure for plotting
        if plot:
            plt.figure(figsize=(10, 6))

        # Iterate through each unique group
        for i, group in enumerate(unique_groups):
            # Filter the data for the specific group
            group_data = df[df[group_column] == group]

            # Define the dependent (y) and independent (X) variables
            y_values = group_data[y]
            x_values = group_data[[x]]
            x_values = sm.add_constant(x_values)  # Add a constant term (intercept) to the model

            # Fit the linear regression model
            model = sm.OLS(y_values, x_values).fit()

            # Print the p-value for the 'x' coefficient
            print(f"Group: {group}")
            print(f"p-value for {x}: {model.pvalues[x]:.4f}")
            print("\n")

            if plot:
                # Plot the relationship between 'x' and 'y' with regression line
                plt.scatter(x_values[x], y_values, label=f"Group: {group}", color=palette[i])
                plt.plot(x_values[x], model.predict(x_values), color=palette[i])

        if plot:
            # Customize the plot
            plt.title(f"Relationship between {x} and {y}")
            plt.xlabel(x)
            plt.ylabel(y)
            plt.legend(title=group_column)

            # Show the plot
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    else:
        # Define the dependent (y) and independent (X) variables for the entire dataset
        y_values = df[y]
        x_values = df[[x]]
        x_values = sm.add_constant(x_values)  # Add a constant term (intercept) to the model

        # Fit the linear regression model for the entire dataset
        model = sm.OLS(y_values, x_values).fit()

        # Print the p-value for the 'x' coefficient
        print("Entire Dataset:")
        print(f"p-value for {x}: {model.pvalues[x]:.4f}")

        if plot:
            # Create a figure for plotting
            plt.figure(figsize=(10, 6))

            # Plot the relationship between 'x' and 'y' with regression line
            plt.scatter(x_values[x], y_values)
            plt.plot(x_values[x], model.predict(x_values), color='red')

            # Customize the plot
            plt.title(f"Relationship between {x} and {y}")
            plt.xlabel(x)
            plt.ylabel(y)

            # Show the plot
            plt.grid(True)
            plt.tight_layout()
            plt.show()

### plot individual tract results (null + empirical) #####
def plot_density(map_name, tract, result_value, density_color):
    
    # Load the nulls CSV file based on map_name
    nulls_file = f"./outputs/statistical_testing/nulls_{map_name}.csv"
    nulls_df = pd.read_csv(nulls_file)

    # Filter rows based on the specified tract_name
    tract_data = nulls_df[nulls_df['tract_name'] == tract]

    if tract_data.empty:
        print(f"No data found for tract '{tract}' in map '{map_name}'.")
        return

    # Load the empirical t-tests CSV file
    empirical_file = "./outputs/statistical_testing/empirical_t_tests.csv"
    empirical_df = pd.read_csv(empirical_file)
    
    # Filter rows based on tract_name and map_name
    empirical_data = empirical_df[(empirical_df['tract_name'] == tract) & (empirical_df['map_name'] == map_name)]

    if empirical_data.empty:
        print(f"No empirical data found for tract '{tract}' in map '{map_name}'.")
        return
    
    # Create a new figure for each plot
    fig = plt.figure()

    # Create a probability density plot using KDE
    null_results = tract_data[f'null_{result_value}']
    sns.kdeplot(null_results, shade = True, color = density_color)

    # Add a vertical line at the empirical mean difference (i.e. for the non-rotated original map)
    empirical_results = empirical_data[f'empirical_{result_value}'].values[0]
    plt.axvline(x = empirical_results, color = 'orange', linestyle='--', label = f'Empirical {result_value}')
    
    # get FDR corrected p-value
    fdr_pval = round(empirical_data['fdr_corrected_p_val'].values[0], 3)

    # Add labels and title
    plt.xlabel(f'{result_value}')
    plt.ylabel('Probability Density')
    plt.suptitle(f'Density Plot for Tract {tract} in {map_name}')
    plt.title(f'FDR corrected p-value: {fdr_pval}')
    plt.legend()

    # Show the plot
    plt.show()

    # Save the figure as an image
    image_path = f'./outputs/statistical_testing/plot_{map_name}_{tract}.png'
    fig.savefig(image_path)
    plt.close(fig)
    

### Compute empirical results ####
def compute_empirical(maps_list, map_names, tracts, tracts_regs_ids):
    """
    Compute empirical data for each combination of map and tract.

    Parameters:
    - maps_list (list): A list of map names to iterate over.
    - tracts (list): A list of tract names to iterate over.
    - map_names (list): A list of map names for reference.

    Returns:
    - result_df (DataFrame): A DataFrame containing computed empirical data with FDR-corrected p-values.
    """
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # loop over all selected brain maps
    for index, mp in enumerate(maps_list):

        # map name
        map_name = map_names[index]
        print(f'Running empirical testing for {map_name}...')

        # generate results lists
        tract_names = [] # tract names
        tract_size = [] # tract size (i.e., number of connected regions)
        tracts_mean_map = [] # mean cortical map value across tracts
        tracts_emps_mean_diffs = [] # empirical results across tracts
        tracts_emps_t_vals = [] # empirical t-values across tracts
        tracts_emps_p_vals = [] # empirical p-values across tracts

        # loop over tracts
        for index, tract in enumerate(tracts):
    #         print(f'Computing {tract}...')

            ### EMPIRICAL RESULTS FOR EACH MAP AND TRACT ###
      
            # select brain region indices
            tract_regs_idx = tracts_regs_ids.index[tracts_regs_ids[f'{tract}'] > 0.95].tolist()

            # count number of connected regions
            nb_connected = len(tract_regs_idx)

            # select tract hemisphere
            if 'left' in tract:
                hem = 'L_'
            elif 'right' in tract:
                hem = 'R_'
            else:
                raise ValueError(f"hemisphere not found for: {tract}")

            # select all brain region indices corresponding to the tract's hemisphere
            hem_all_idx = tracts_regs_ids.index[tracts_regs_ids['parcel_name'].str.contains(f'{hem}')].tolist()

            # create array with map values of connected brain regions
            connected = [mp[i] for i in tract_regs_idx]
#             print(f'Number of brain regions structurally connected to the {tract}: {len(connected)}') 

            # create array with map values of non-connected regions (includes regions in both hemispheres)
#             non_connected = [mp[i] for i in range(len(mp)) if i not in tract_regs_idx]
#             print(f'Number of brain regions NOT structurally connected to the {tract}: {len(non_connected)}') 

            # create array with map values of non-connected regions of the SAME hemisphere as the tract 
            non_connected = [mp[i] for i in range(len(mp)) if i not in tract_regs_idx and i in hem_all_idx]

            # two sample two-tailed t-test to compare the difference of the two distributions (structurally connected vs non-connected regions)
            empirical_t_statistic, empirical_p_value = sstats.ttest_ind(connected, non_connected)

            # compute the difference in the means (will be used for plotting)
            empirical_mean_diff = mean(connected) - mean(non_connected)

            # Print the results
#             print(f"{tract} - Empirical T-Statistic:", round(empirical_t_statistic, 3))
#             print(f"{tract} - Empirical P-Value:", round(empirical_p_value, 3))
#             print(f"{tract} - Empirical difference in means:", round(empirical_mean_diff, 3))

            # save the results
            tract_names.append(f'{tract}')
            tract_size.append(nb_connected)
            tracts_mean_map.append(round(mean(connected), 3))
            tracts_emps_mean_diffs.append(round(empirical_mean_diff, 3))
            tracts_emps_t_vals.append(round(empirical_t_statistic, 3))
            tracts_emps_p_vals.append(round(empirical_p_value, 3))

        ### STORE MAP RESULTS (ACROSS ALL TRACTS) ###

        # Create a temporary DataFrame for this map
        map_df = pd.DataFrame({
            'tract_name': tract_names,
            'tract_size': tract_size,
            'cortical_map_mean': tracts_mean_map,
            'empirical_t_statistic': tracts_emps_t_vals,
            'empirical_mean_diff': tracts_emps_mean_diffs,
            'empirical_p_val': tracts_emps_p_vals
        })

        # Add a map_name column to the temporary DataFrame
        map_df['map_name'] = map_name

        # Append the temporary DataFrame to the list of DataFrames
        dfs.append(map_df)

    # Concatenate the list of DataFrames into one DataFrame
    result_df = pd.concat(dfs, ignore_index=True)

    # Apply FDR correction
    _, pvals_corrected, _, _ = sm.stats.multipletests(result_df['empirical_p_val'], alpha=0.05, method='fdr_bh')

    # Replace the original 'P_Value' column with the corrected values
    result_df['fdr_corrected_p_val'] = pvals_corrected

    # Write the result to a CSV file
    result_df.to_csv(f'./outputs/statistical_testing/empirical_t_tests.csv', index=False, header=True)

    print('Done!')
    return result_df  


### Compute null distribution ###
def compute_nulls(maps_list, tracts, map_names, tracts_regs_ids, parcellation):
    """
    Compute null data for each combination of map and tract.

    Parameters:
    - maps_list (list): A list of map names to iterate over.
    - tracts (list): A list of tract names to iterate over.
    - map_names (list): A list of map names for reference.
    - tract_regs_ids (dataframe): A dataframe of Glasser parcels (rows) by tracts (columns).
    - parcellation (tuple): A tuple of 2 gifti objects contains left and right hemisphere Glasser labels 
                            (where each gifti contains an array of 32492 vertices with assigned labels).

    Returns:
    - null_map_dict (dict): A dictionary containing null DataFrames for each map_name.
    """
    
    # Initialize an empty dictionary to store null DataFrames
    null_map_dict = {}
    
    # loop over all selected brain maps
    for index, mp in enumerate(maps_list):

        # map name
        map_name = map_names[index]
        print(f'Running nulls for {map_name}...')

        # create nulls by shuffling the parcellated map - this gives a new randomly assigned map values (for instance, myelin) for each brain region
        rotated_maps = nulls.alexander_bloch(mp, atlas = 'fsLR', density = '32k',
                                        n_perm = 100, seed = 1234, parcellation = parcellation)

        # generate results lists
        tract_names = [] # tract names
        tracts_nulls_rotated_map_names = [] # rotated null map names
        nulls_tract_names = [] # tract names for null results
        tracts_nulls_mean_diffs = [] # null results across tracts
        tracts_nulls_t_vals = [] # null t-values across tracts
        tracts_nulls_p_vals = [] # null p-values across tracts

        # loop over tracts
        for index, tract in enumerate(tracts):
#             print(f'Computing {tract}...')

            # select brain region indices
            tract_regs_idx = tracts_regs_ids.index[tracts_regs_ids[f'{tract}'] > 0.95].tolist()

            # count number of connected regions
            nb_connected = len(tract_regs_idx)

            # select tract hemisphere
            if 'left' in tract:
                hem = 'L_'
            elif 'right' in tract:
                hem = 'R_'
            else:
                raise ValueError(f"hemisphere not found for: {tract}")

            # select all brain region indices corresponding to the tract's hemisphere
            hem_all_idx = tracts_regs_ids.index[tracts_regs_ids['parcel_name'].str.contains(f'{hem}')].tolist()

            ### NULL RESULTS ###

            # null result lists
            rotated_map_names = []
            null_tract_name = []
            null_t_stats = []
            null_p_vals = []
            null_mean_diffs = []

            # generate null distribution (using neuromaps spatial nulls)
            for map_index, rotated_map in enumerate(rotated_maps.T): # transposing to iterate through columns (i.e. maps)

                if map_index < 100: # loop until the 99th iteration, which corresponds to the 100th and last column

                    # assign rotated map name
                    rotated_map_name = f'rotated_map_{map_index}'

                    # assign the brain regions as connected vs non-connected for each of the 100 permuted maps (columns), except now the map values have been shuffled
                    connected = [rotated_maps[i, map_index] for i in tract_regs_idx]

                    # subset non-connected brain regions (includes regions of both hemispheres)
#                     non_connected = [rotated_maps[i, map_index] for i in range(len(rotated_maps)) if i not in tract_regs_idx]

                    # subset only non-connected brain regions of the SAME hemisphere as the tract 
                    non_connected = [rotated_maps[i, map_index] for i in range(len(rotated_maps)) if i not in tract_regs_idx and i in hem_all_idx]

#                     print(f'Number of brain regions structurally connected to the tract : {len(connected)}') 
#                     print(f'Number of brain regions NOT structurally connected to the tract: {len(non_connected)}') 

                    # two sample two-tailed t-test to compare the difference of the two distributions (structurally connected vs non-connected regions)
                    t_statistic, p_value = sstats.ttest_ind(connected, non_connected)

                    # compute the difference in the means (will be used for plotting)
                    mean_diff = mean(connected) - mean(non_connected)

                    # print the results
#                     print(f'Map number: {map_index}')
#                     print("T-Statistic:", round(t_statistic, 3))
#                     print("P-Value:", round(p_value, 3))
#                     print("Difference in means:", round(mean_diff, 3))

                    # save the results (for each tract x null map)
                    rotated_map_names.append(f'{rotated_map_name}')
                    null_tract_name.append(f'{tract}')
                    null_t_stats.append(round(t_statistic, 3))
                    null_p_vals.append(round(p_value, 3))
                    null_mean_diffs.append(round(mean_diff, 3))

            # save the null results (across tracts)
            tracts_nulls_rotated_map_names.extend(rotated_map_names)
            nulls_tract_names.extend(null_tract_name)
            tracts_nulls_t_vals.extend(null_t_stats)
            tracts_nulls_p_vals.extend(null_p_vals)
            tracts_nulls_mean_diffs.extend(null_mean_diffs)

        ### STORE MAP RESULTS (ACROSS ALL TRACTS) ###

        # create a temporary null DataFrame for this map
        null_map_df = pd.DataFrame({
            'tract_name': nulls_tract_names,
            'rotated_map_name': tracts_nulls_rotated_map_names,
            'null_t_statistic': tracts_nulls_t_vals, 
            'null_mean_diff': tracts_nulls_mean_diffs,
            'null_p_val': tracts_nulls_p_vals,
        })

        null_map_df['map_name'] = map_name

        # create outputs folder if doesn't yet exist
        outputs_folder = './outputs/statistical_testing'
        if not os.path.exists(outputs_folder):
            os.makedirs(outputs_folder)

        # write as csv
        null_map_df.to_csv(f'./outputs/statistical_testing/nulls_{map_name}.csv', index = False, header = True)
        
         # Add the null_map_df to the dictionary with the map_name as the key
        null_map_dict[map_name] = null_map_df

    print('Done!')
    return null_map_dict

### Display tract, map gradients and tract mean, SD #####
def create_gradient_bar(tract, tracts_regs_ids, maps_list, map_names, colormap = 'viridis'):
    
    '''
    Create a horizontal bar with a color gradient and standard deviation indicators.

    Parameters
    -----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes on which to create the bar.
   maps_list : list of numpy.ndarray 
       A list of data arrays for which statistics are calculated and displayed.
    map_name : str
        The name of the map to be displayed.
    colormap : str, optional
        The name of the colormap to use for the color gradient. Default is 'viridis'.

    Returns
    --------
    plot : The function generates a grid of horizontal bar plots.
    '''
    ### Define tract information ###

    # select brain region indices
    tract_regs_idx = tracts_regs_ids.index[tracts_regs_ids[f'{tract}'] > 0.95].tolist()
    
    # select tract hemisphere
    if 'left' in tract:
        hem = 'L_'
        hem_folder = 'left_hem'
    elif 'right' in tract:
        hem = 'R_'
        hem_folder = 'right_hem'
    else:
        raise ValueError(f"hemisphere not found for: {tract}")
    
    ### Set up plotting grid ###
    
    num_rows = len(maps_list)
    num_columns = 2
    
    # Calculate the figure height based on the number of rows
    fig_height = num_rows * 1.1  # Adjust the multiplier as needed for desired visual output

    # Create a grid with n rows and 2 columns
    fig = plt.figure(figsize=(12, fig_height))
    gs = GridSpec(num_rows, num_columns, width_ratios=[1, 2])

    # Load the JPG image
    tract_files = pd.read_csv('./inputs/tract_labels.csv')
    filename = tract_files[tract_files['tract_names'] == tract]['file_names'].iloc[0]
    image_path = f'./tracts_figures/{hem_folder}/{filename}'
    img = plt.imread(image_path)

    # Add the JPG tract image to the left column (spanning all n rows)
    ax_img = plt.subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.axis('off')
    
    # generate gradient plot for all brain maps
    for i, mp in enumerate(maps_list):
        
        # define map name
        map_name = map_names[i]

        # create array with map values of connected brain regions
        connected = [mp[reg_id] for reg_id in tract_regs_idx]
        
        # Calculate mean and SD of connected regions
        mean_val = np.mean(connected)
        std_dev = np.std(connected)

        # Create a gradient color map
        cmap = plt.get_cmap(colormap)
        
        # get range (min and max of the entire brain map) to make gradient
        min_val = np.min(mp)
        max_val = np.max(mp)

        # Create a range of values for the color gradient
        gradient_values = np.linspace(min_val, max_val, 1000)

        # Create a horizontal bar with a gradient of colors
        ax_bar = plt.subplot(gs[i, 1])
        ax_bar.imshow([gradient_values], cmap = cmap, aspect = 'auto', extent = [min_val, max_val, 0, 1])

        # Add a vertical black line for the mean
        ax_bar.axvline(x = mean_val, color = 'indigo', linestyle = '-', linewidth = 5)

        # Calculate the confidence interval boundaries
        lower_bound = mean_val - std_dev
        upper_bound = mean_val + std_dev

        # Plot the standard deviation as two vertical gray lines
        ax_bar.axvline(x = lower_bound, color = 'tab:purple', linestyle = '-', linewidth = 4)
        ax_bar.axvline(x = upper_bound, color = 'tab:purple', linestyle = '-', linewidth = 4)

        # Set the x-axis limits and label
        ax_bar.set_xlim(min_val, max_val)
        ax_bar.set_xlabel('map values')
        ax_bar.set_title(f'{map_name}')

        # Remove y-axis labels and ticks
        ax_bar.set_yticks([])
        ax_bar.set_yticklabels([])

    plt.tight_layout()
    plt.show()

