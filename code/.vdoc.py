# type: ignore
# flake8: noqa
#
import matplotlib.pyplot as plt
import pandas as pd
from my_functions import *
#
#
#
#
#
#
#
## Set variables for the title and paths
title = 'Borneo Total Forest Loss 2001 - 2021'
input_tif = '../../07_data_prepared/deforestation/LAEA_gfw_2000_2022_borneo.tif'
output_csv = 'results/tables/gfc_deforestation_total_yearly.csv'
output_png = 'results/plots/borneo_total_forest_loss.png'

with rasterio.open(input_tif) as src:
    ## Get the spatial resolution (in meters) from the TIF file metadata
    spatial_resolution = src.res[0]

    ## Calculate the conversion factor from pixels to hectares
    conversion_factor = (spatial_resolution ** 2) / 10000

    band1 = src.read(1)
    unique, counts = np.unique(band1, return_counts=True)
    data = dict(zip(unique[unique != 0], counts[unique != 0] * conversion_factor))
    df = pd.DataFrame(list(data.items()), columns=['Year', 'Area (ha)'])
    df['Year'] = df['Year'] + 2000
    df.to_csv(output_csv, index=False)

plt.style.use('bmh') ## set the style to 'bmh'

df = pd.read_csv(output_csv)

ax = df.plot(x='Year', y='Area (ha)', kind='bar')
for i in ax.containers:
    ax.bar_label(i, label_type='edge', rotation=90, padding=3, fmt=lambda x: f'{x:.0f}')

## add the x-axis line back
ax.spines['bottom'].set_visible(True)

## remove the vertical grid lines and add horizontal grid lines
ax.xaxis.grid(False)
ax.yaxis.grid(True)

## extend the y-axis by 15%
y_max = df['Area (ha)'].max()
ax.set_ylim(0, y_max * 1.15)

## add a title to the plot
ax.set_title(title)

## save the plot as a .png file with extra space around the edges
plt.savefig(output_png, bbox_inches='tight')
#
#
#
#
## Set variables for the title and paths
title = 'Borneo Total Forest Loss (Forest Fires) 2001 - 2021'
input_tif = '../../07_data_prepared/deforestation/LAEA_forest_fires.tif'
output_csv = 'results/tables/forest_fires_loss_total_yearly.csv'
output_png = 'results/plots/forest_fires_loss_total_yearly.png'

with rasterio.open(input_tif) as src:
    ## Get the spatial resolution (in meters) from the TIF file metadata
    spatial_resolution = src.res[0]

    ## Calculate the conversion factor from pixels to hectares
    conversion_factor = (spatial_resolution ** 2) / 10000

    band1 = src.read(1)
    unique, counts = np.unique(band1, return_counts=True)
    data = dict(zip(unique[unique != 0], counts[unique != 0] * conversion_factor))
    df = pd.DataFrame(list(data.items()), columns=['Year', 'Area (ha)'])
    df['Year'] = df['Year'] + 2000
    df.to_csv(output_csv, index=False)

plt.style.use('bmh') ## set the style to 'bmh'

df = pd.read_csv(output_csv)

ax = df.plot(x='Year', y='Area (ha)', kind='bar')
for i in ax.containers:
    ax.bar_label(i, label_type='edge', rotation=90, padding=3, fmt=lambda x: f'{x:.0f}')

## add the x-axis line back
ax.spines['bottom'].set_visible(True)

## remove the vertical grid lines and add horizontal grid lines
ax.xaxis.grid(False)
ax.yaxis.grid(True)

## extend the y-axis by 15%
y_max = df['Area (ha)'].max()
ax.set_ylim(0, y_max * 1.15)

## add a title to the plot
ax.set_title(title)

## save the plot as a .png file with extra space around the edges
plt.savefig(output_png, bbox_inches='tight')
#
#
#
#
#
input_tif = '../../07_data_prepared/deforestation/LAEA_gfw_2000_2022_borneo.tif'
mask_tif = '../../07_data_prepared/deforestation/LAEA_primary_forest_2001.tif'
output_tif = '../../08_data_processed/deforestation/deforestation_primary_forest.tif'

mask_tif_select_nodata(input_tif, mask_tif, output_tif)
#
#
#
#
## Set variables for the title and paths
title = 'Borneo Total Primary Forest Loss 2001 - 2021'
input_tif = '../../08_data_processed/deforestation/deforestation_primary_forest.tif'
output_csv = 'results/tables/primary_forest_loss_total_yearly.csv'
output_png = 'results/plots/primary_loss_total_yearly.png'

with rasterio.open(input_tif) as src:
    ## Get the spatial resolution (in meters) from the TIF file metadata
    spatial_resolution = src.res[0]

    ## Calculate the conversion factor from pixels to hectares
    conversion_factor = (spatial_resolution ** 2) / 10000

    band1 = src.read(1)
    unique, counts = np.unique(band1, return_counts=True)
    data = dict(zip(unique[unique != 0], counts[unique != 0] * conversion_factor))
    df = pd.DataFrame(list(data.items()), columns=['Year', 'Area (ha)'])
    df['Year'] = df['Year'] + 2000
    df.to_csv(output_csv, index=False)

plt.style.use('bmh') ## set the style to 'bmh'

df = pd.read_csv(output_csv)

ax = df.plot(x='Year', y='Area (ha)', kind='bar')
for i in ax.containers:
    ax.bar_label(i, label_type='edge', rotation=90, padding=3, fmt=lambda x: f'{x:.0f}')

## add the x-axis line back
ax.spines['bottom'].set_visible(True)

## remove the vertical grid lines and add horizontal grid lines
ax.xaxis.grid(False)
ax.yaxis.grid(True)

## extend the y-axis by 15%
y_max = df['Area (ha)'].max()
ax.set_ylim(0, y_max * 1.15)

## add a title to the plot
ax.set_title(title)

## save the plot as a .png file with extra space around the edges
plt.savefig(output_png, bbox_inches='tight')
#
#
#
#
#

input_tif = '../../07_data_prepared/deforestation/LAEA_gfw_2000_2022_borneo.tif'
mask_tif = '../../07_data_prepared/infrastructure/LAEA_build_up_area.tif'
output_tif = '../../08_data_processed/combined/deforestation_new_build_up_area.tif'

mask_tif_select_value(input_tif, mask_tif, output_tif, mask_values= [1])
#
#
#
#
## Set variables for the title and paths
title = 'Borneo Forest Loss to build up areas 2001 - 2021'
input_tif = '../../08_data_processed/combined/deforestation_new_build_up_area.tif'
output_csv = 'results/tables/forest_loss_build_up_areas_yearly.csv'
output_png = 'results/plots/forest_loss_build_up_areas_yearly.png'

with rasterio.open(input_tif) as src:
    ## Get the spatial resolution (in meters) from the TIF file metadata
    spatial_resolution = src.res[0]

    ## Calculate the conversion factor from pixels to hectares
    conversion_factor = (spatial_resolution ** 2) / 10000

    band1 = src.read(1)
    unique, counts = np.unique(band1, return_counts=True)
    data = dict(zip(unique[unique != 0], counts[unique != 0] * conversion_factor))
    df = pd.DataFrame(list(data.items()), columns=['Year', 'Area (ha)'])
    df['Year'] = df['Year'] + 2000
    df.to_csv(output_csv, index=False)

plt.style.use('bmh') ## set the style to 'bmh'

df = pd.read_csv(output_csv)

ax = df.plot(x='Year', y='Area (ha)', kind='bar')
for i in ax.containers:
    ax.bar_label(i, label_type='edge', rotation=90, padding=3, fmt=lambda x: f'{x:.0f}')

## add the x-axis line back
ax.spines['bottom'].set_visible(True)

## remove the vertical grid lines and add horizontal grid lines
ax.xaxis.grid(False)
ax.yaxis.grid(True)

## extend the y-axis by 15%
y_max = df['Area (ha)'].max()
ax.set_ylim(0, y_max * 1.15)

## add a title to the plot
ax.set_title(title)

## save the plot as a .png file with extra space around the edges
plt.savefig(output_png, bbox_inches='tight')
#
#
#
#
#
#

input_tif = '../../07_data_prepared/deforestation/LAEA_forest_fires.tif'
mask_tif = '../../07_data_prepared/infrastructure/LAEA_build_up_area.tif'
output_tif = '../../08_data_processed/combined/forest_fires_new_build_up_area.tif'

mask_tif_select_value(input_tif, mask_tif, output_tif, mask_values= [1])
#
#
#
#
## Set variables for the title and paths
title = 'Borneo Forest Loss to build up areas by forest fires 2001 - 2021'
input_tif = '../../08_data_processed/combined/forest_fires_new_build_up_area.tif'
output_csv = 'results/tables/forest_fires_new_build_up_area_yearly.csv'
output_png = 'results/plots/forest_fires_new_build_up_area_yearly.png'

with rasterio.open(input_tif) as src:
    ## Get the spatial resolution (in meters) from the TIF file metadata
    spatial_resolution = src.res[0]

    ## Calculate the conversion factor from pixels to hectares
    conversion_factor = (spatial_resolution ** 2) / 10000

    band1 = src.read(1)
    unique, counts = np.unique(band1, return_counts=True)
    data = dict(zip(unique[unique != 0], counts[unique != 0] * conversion_factor))
    df = pd.DataFrame(list(data.items()), columns=['Year', 'Area (ha)'])
    df['Year'] = df['Year'] + 2000
    df.to_csv(output_csv, index=False)

plt.style.use('bmh') ## set the style to 'bmh'

df = pd.read_csv(output_csv)

ax = df.plot(x='Year', y='Area (ha)', kind='bar')
for i in ax.containers:
    ax.bar_label(i, label_type='edge', rotation=90, padding=3, fmt=lambda x: f'{x:.0f}')

## add the x-axis line back
ax.spines['bottom'].set_visible(True)

## remove the vertical grid lines and add horizontal grid lines
ax.xaxis.grid(False)
ax.yaxis.grid(True)

## extend the y-axis by 15%
y_max = df['Area (ha)'].max()
ax.set_ylim(0, y_max * 1.15)

## add a title to the plot
ax.set_title(title)

## save the plot as a .png file with extra space around the edges
plt.savefig(output_png, bbox_inches='tight')
#
#
#
#
#

input_tif = '../../07_data_prepared/deforestation/LAEA_forest_fires.tif'
mask_shp = '../../07_data_prepared/protected_areas/LAEA_protected_areas_borneo.shp'
output_tif = '../../08_data_processed/combined/deforestation_protected_areas.tif'

mask_tif_with_shapefile(input_tif, mask_shp, output_tif)
#
#
#
#
## Set variables for the title and paths
title = 'Forest loss in protected areas'
input_tif = '../../08_data_processed/combined/deforestation_protected_areas.tif'
output_csv = 'results/tables/deforestation_protected_areas_yearly.csv'
output_png = 'results/plots/deforestation_protected_areas_yearly.png'

with rasterio.open(input_tif) as src:
    ## Get the spatial resolution (in meters) from the TIF file metadata
    spatial_resolution = src.res[0]

    ## Calculate the conversion factor from pixels to hectares
    conversion_factor = (spatial_resolution ** 2) / 10000

    band1 = src.read(1)
    unique, counts = np.unique(band1, return_counts=True)
    data = dict(zip(unique[unique != 0], counts[unique != 0] * conversion_factor))
    df = pd.DataFrame(list(data.items()), columns=['Year', 'Area (ha)'])
    df['Year'] = df['Year'] + 2000
    df.to_csv(output_csv, index=False)

plt.style.use('bmh') ## set the style to 'bmh'

df = pd.read_csv(output_csv)

ax = df.plot(x='Year', y='Area (ha)', kind='bar')
for i in ax.containers:
    ax.bar_label(i, label_type='edge', rotation=90, padding=3, fmt=lambda x: f'{x:.0f}')

## add the x-axis line back
ax.spines['bottom'].set_visible(True)

## remove the vertical grid lines and add horizontal grid lines
ax.xaxis.grid(False)
ax.yaxis.grid(True)

## extend the y-axis by 15%
y_max = df['Area (ha)'].max()
ax.set_ylim(0, y_max * 1.15)

## add a title to the plot
ax.set_title(title)

## save the plot as a .png file with extra space around the edges
plt.savefig(output_png, bbox_inches='tight')
#
#
#
#
#
#
#
## Set variables for the title and paths
title = 'Borneo new oil palm plantations (2001 - 2017)'
input_tif = '../../07_data_prepared/oil_palm/LAEA_detection_year_2000_2017.tif'
output_csv = 'results/tables/new_oil_palm_detection_yearly.csv'
output_png = 'results/plots/new_oil_palm_detection_yearly.png'

with rasterio.open(input_tif) as src:
    ## Get the spatial resolution (in meters) from the TIF file metadata
    spatial_resolution = src.res[0]

    ## Calculate the conversion factor from pixels to hectares
    conversion_factor = (spatial_resolution ** 2) / 10000

    band1 = src.read(1)
    unique, counts = np.unique(band1, return_counts=True)
    data = dict(zip(unique[unique != 0], counts[unique != 0] * conversion_factor))
    df = pd.DataFrame(list(data.items()), columns=['Year', 'Area (ha)'])
    df['Year'] = df['Year']
    df.to_csv(output_csv, index=False)

plt.style.use('bmh') ## set the style to 'bmh'

df = pd.read_csv(output_csv)

ax = df.plot(x='Year', y='Area (ha)', kind='bar')
for i in ax.containers:
    ax.bar_label(i, label_type='edge', rotation=90, padding=3, fmt=lambda x: f'{x:.0f}')

## add the x-axis line back
ax.spines['bottom'].set_visible(True)

## remove the vertical grid lines and add horizontal grid lines
ax.xaxis.grid(False)
ax.yaxis.grid(True)

## extend the y-axis by 15%
y_max = df['Area (ha)'].max()
ax.set_ylim(0, y_max * 1.15)

## add a title to the plot
ax.set_title(title)

## save the plot as a .png file with extra space around the edges
plt.savefig(output_png, bbox_inches='tight')
#
#
#
#
#
#

snap_tif = '../../07_data_prepared/deforestation/LAEA_gfw_2000_2022_borneo.tif'
input_tif = '../../07_data_prepared/oil_palm/LAEA_detection_year_2000_2017.tif'
output_tif = '../../07_data_prepared/oil_palm/LAEA_snap_gfw_detection_year_2000_2017.tif'


snap_raster(input_tif, snap_tif, output_tif)
#
#
#
#
#
input_tif = '../../07_data_prepared/deforestation/LAEA_gfw_2000_2022_borneo.tif'
mask_tif = '../../08_data_processed/combined/snap_oil_palm_deforestation.tif'
output_tif = '../../08_data_processed/combined/oil_palm_deforestation.tif'

mask_tif_select_nodata(input_tif, mask_tif, output_tif)
#
#
#
#
#

input_tif = '../../07_data_prepared/infrastructure/LAEA_build_up_area.tif'
output_tif = '../../07_data_prepared/infrastructure/LAEA_buffer_build_up_area_borneo.tif'
buffer_distances = [500, 1000, 2000]

for buffer_distance in buffer_distances:
    buffer_tif(input_tif, buffer_distance, output_tif.replace("_buffer", "_buffer_" + str(buffer_distance) + "_output.tif"))
#
#
#
#
#
