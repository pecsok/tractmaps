######### UTILITY FUNCTIONS FOR TRACTMAPS #########
import hcp_utils as hcp
import nilearn.plotting as plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
       