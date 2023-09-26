######### UTILITY FUNCTIONS FOR TRACTMAPS #########
import os
import hcp_utils as hcp
import nilearn.plotting as plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind
from statistics import mean
from PIL import Image
from neuromaps import nulls, images
import nibabel as nib

# subset hcp Glasser parcellation to only contain cortical regions (remove subcortical areas 361-380)
hcp.mmp_short = hcp.mmp
hcp.mmp_short.map_all = hcp.mmp.map_all[(hcp.mmp.map_all >= 0) & (hcp.mmp.map_all <= 360)]
hcp.mmp_short.labels = subset_dict = {key: hcp.mmp.labels[key] for key in list(hcp.mmp.labels.keys())[:361]}
hcp.mmp_short.ids = hcp.mmp.ids[0:361]
hcp.mmp_short.ids = hcp.mmp.ids[0:361]
hcp.mmp_short.nontrivial_ids = hcp.mmp.nontrivial_ids[0:361]
hcp.mmp_short.rgba = subset_dict = {key: hcp.mmp.rgba[key] for key in list(hcp.mmp.rgba.keys())[:361]}


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
    
    # assign brain map values of each parcels to correspinding vertices on full brain surface
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
    plt.figure(figsize=(20, 2))  # Adjust the width and height as needed

    # Create a mask to hide non-significant values
    mask = result_df.pivot_table(index= brain_maps_col, columns = tracts_col, values = p_value_col) > significance_threshold

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
            empirical_t_statistic, empirical_p_value = ttest_ind(connected, non_connected)

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
def compute_nulls(maps_list, tracts, map_names, tracts_regs_ids):
    """
    Compute null data for each combination of map and tract.

    Parameters:
    - maps_list (list): A list of map names to iterate over.
    - tracts (list): A list of tract names to iterate over.
    - map_names (list): A list of map names for reference.

    Returns:
    - null_map_dict (dict): A dictionary containing null DataFrames for each map_name.
    """
    
    # Initialize an empty dictionary to store null DataFrames
    null_map_dict = {}
    
    # get Glasser labels
    lh_glasser = nib.load('./inputs/glasser_360_L.label.gii')
    rh_glasser = nib.load('./inputs/glasser_360_R.label.gii')

    # generate neuromaps fsLR based Glasser 360 parcellation (needs relabeling to have consecutive region IDs)
    glasser = images.relabel_gifti((lh_glasser, rh_glasser), background=['Medial_wall'])
    
    # loop over all selected brain maps
    for index, mp in enumerate(maps_list):

        # map name
        map_name = map_names[index]
        print(f'Running nulls for {map_name}...')

        # create nulls by shuffling the parcellated map - this gives a new randomly assigned map values (for instance, myelin) for each brain region
        rotated_maps = nulls.alexander_bloch(mp, atlas = 'fsLR', density = '32k',
                                        n_perm = 100, seed = 1234, parcellation = glasser)

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
                    t_statistic, p_value = ttest_ind(connected, non_connected)

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

    