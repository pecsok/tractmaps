######### UTILITY FUNCTIONS FOR TRACTMAPS #########
import hcp_utils as hcp
import nilearn.plotting as plotting
import numpy as np
import pandas as pd

# subset hcp Glasser parcellation to only contain cortical regions (remove subcortical areas 361-380)
hcp.mmp_short = hcp.mmp
hcp.mmp_short.map_all = hcp.mmp.map_all[(hcp.mmp.map_all >= 0) & (hcp.mmp.map_all <= 360)]
hcp.mmp_short.labels = subset_dict = {key: hcp.mmp.labels[key] for key in list(hcp.mmp.labels.keys())[:361]}
hcp.mmp_short.ids = hcp.mmp.ids[0:361]
hcp.mmp_short.ids = hcp.mmp.ids[0:361]
hcp.mmp_short.nontrivial_ids = hcp.mmp.nontrivial_ids[0:361]
hcp.mmp_short.rgba = subset_dict = {key: hcp.mmp.rgba[key] for key in list(hcp.mmp.rgba.keys())[:361]}


### Plot parcellated brain map ###
def plot_parc_map(brain_map, map_name, colors):
    """
    Inputs: Takes a parcellated brain map, its corresponding name (which will be used for plot titles), and color map (see matplotlib standard options).
    Output: Returns an interactive plot showing the parcellated brain map.
    
    """
    
    # assign brain map values of each parcels to correspinding vertices on full brain surface
    unparcellated_map = hcp.unparcellate(brain_map, hcp.mmp_short)
    
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
def plot_parc_subset(brain_map, tract, map_name, tract_name, colors, connection_threshold=0.95):
    """
    Inputs: Takes a parcellated brain map, a tract of interest, the tract-to-region probability of connection threshold (default=0.95); brain map name, tract name and colors are for plotting.
    Output: Returns an interactive plot showing map values for regions structurally connected to the given tract.
    
    """
    # read in csv with tracts and Glasser region IDs
    tracts_regs_ids = pd.read_csv('./outputs/tracts_regs_Glasser.csv')
    
    # select brain regions structurally connected to the tract
    tract_df = tracts_regs_ids.loc[tracts_regs_ids[f'{tract}'] >= connection_threshold, ['parcel_name', 'regionLongName', 'cortex', 'regionID']]
#     print(f'Number of brain regions structurally connected to the {tract}: {len(tract_df)}')
    
    # define region IDs of structurally connected regions for the tract
    region_ids = np.array(tract_df['regionID'].astype('int'))

    # select region ids across full surface map vertices
    region_ids_all = np.isin(hcp.mmp_short.map_all, region_ids)

    # assign brain map values of each parcels to correspinding vertices on full brain surface
    unparcellated_map = hcp.unparcellate(brain_map, hcp.mmp_short)
    
    # plot a subset of region IDs
    input_parc_map = hcp.cortex_data(hcp.mask(unparcellated_map, region_ids_all))
    input_parc_map[input_parc_map == 0]  = np.nan # assign nan to medial wall

    # plot regions
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