import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
import _osx_support
import rasterio
from rasterio import features
from rasterio.merge import merge
from rasterio.errors import RasterioIOError
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from shapely.geometry import shape, mapping, box
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import disk

# check crs
def check_crs(tif_file):
    with rasterio.open(tif_file) as src:
        crs = src.crs
        print(f"CRS of {tif_file}: {crs}")

# combine tiff's
def merge_tif_files(tif_files, output_file):
    src_files_to_mosaic = []
    
    for file in tif_files:
        try:
            src = rasterio.open(file)
            src_files_to_mosaic.append(src)
        except RasterioIOError:
            print(f"Skipping non-existing file: {file}")
    
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "compress": 'lzw'})
    
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    for src in src_files_to_mosaic:
        src.close()
       
        
# change raster values (reclassify has more options)
def change_raster_values(input_file, output_file, values, new_value):
    # Open the input TIFF file and read the first band as a NumPy array
    with rasterio.open(input_file) as src:
        raster = src.read(1)

        # Convert values to a list if a single integer is provided
        if not isinstance(values, list):
            values = [values]

        # Change pixel values in the raster array
        mask = np.isin(raster, values)
        raster = np.where(mask, new_value, raster)

        # Copy the metadata from the source dataset
        meta = src.meta.copy()

        # Update the metadata for the modified dataset
        meta.update(dtype=rasterio.int32, nodata=new_value, compress='lzw')

        # Save the modified raster array as a new TIFF file
        with rasterio.open(output_file, "w", **meta) as dest:
            dest.write(raster.astype(rasterio.int32), 1)
            

# change values outside mask to chosen value
def change_values_outside_mask(input_file, output_file, mask, new_value = 0):
    # Load the input TIF file
    with rasterio.open(input_file) as src:
        # Read the input TIF file as an array
        data = src.read(1)

        # Extract the first geometry from the GeoDataFrame
        mask_geometry = shape(mask.iloc[0]['geometry'])

        # Create a mask using the input TIF file's georeferencing and the GDF mask
        mask_array = geometry_mask([mapping(mask_geometry)], src.shape, transform=src.transform, invert=True)

        # Replace values outside the mask with the new value
        data = np.where(mask_array, data, new_value)

        # Update the metadata and write the modified array to the output TIF file
        meta = src.meta
        meta.update(compress='lzw')
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(data, 1)
            
            
# clip to bbox
def clip_tif_to_bbox(tif_path, gdf_boundary, output_path):
    from shapely.geometry import box
    # Open the TIFF file
    tif = rasterio.open(tif_path)

    # Extract the bounding box coordinates
    bounds = gdf_boundary.geometry.total_bounds
    bbox = box(*bounds)

    # Clip the TIFF file to the bounding box
    clipped, transform = mask(tif, [bbox], crop=True)

    # Update the metadata of the clipped TIFF
    clipped_meta = tif.meta.copy()
    clipped_meta.update({
        "height": clipped.shape[1],
        "width": clipped.shape[2],
        "transform": transform,
        "compress": 'lzw'
    })

    # Save the clipped TIFF to the output path
    with rasterio.open(output_path, "w", **clipped_meta) as output:
        output.write(clipped)
        

# reproject
## raster to epsg
def reproject_raster(epsg, input_file_path, output_file_path):
    #open source raster
    srcRst = rasterio.open(input_file_path)

    dstCrs = {'init': epsg}

    #calculate transform array and shape of reprojected raster
    transform, width, height = calculate_default_transform(
            srcRst.crs, dstCrs, srcRst.width, srcRst.height, *srcRst.bounds)

    #working of the meta for the destination raster
    kwargs = srcRst.meta.copy()
    kwargs.update({
            'crs': dstCrs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'lzw'
        })

    #open destination raster
    dstRst = rasterio.open(output_file_path, 'w', **kwargs)

    #reproject and save raster band data
    for i in range(1, srcRst.count + 1):
        reproject(
            source=rasterio.band(srcRst, i),
            destination=rasterio.band(dstRst, i),
            #src_transform=srcRst.transform,
            src_crs=srcRst.crs,
            #dst_transform=transform,
            dst_crs=dstCrs,
            resampling=Resampling.nearest)
    #close destination raster
    dstRst.close()

## raster to custom crs in wkt format
def reproject_raster_to_wktcrs(input_file_path, output_file_path, wkt_crs = (
    'PROJCS["Custom Lambert Azimuthal Equal Area",'
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],'
    'UNIT["Degree",0.017453292519943295],'
    'AUTHORITY["EPSG","4326"]],'
    'PROJECTION["Lambert_Azimuthal_Equal_Area"],'
    'PARAMETER["latitude_of_center",0],'
    'PARAMETER["longitude_of_center",115],'
    'UNIT["Meter",1],'
    'AUTHORITY["Custom_CRS","1001"]]'
    )):
    # Open source raster
    srcRst = rasterio.open(input_file_path)

    # Define the custom CRS using WKT
    custom_crs = CRS.from_string(wkt_crs)

    # Calculate transform array and shape of reprojected raster
    transform, width, height = calculate_default_transform(
        srcRst.crs, custom_crs, srcRst.width, srcRst.height, *srcRst.bounds)

    # Update the metadata for the destination raster
    kwargs = srcRst.meta.copy()
    kwargs.update({
        'crs': custom_crs,
        'transform': transform,
        'width': width,
        'height': height,
        'compress': 'lzw'
    })

    # Open destination raster
    dstRst = rasterio.open(output_file_path, 'w', **kwargs)

    # Reproject and save raster band data
    for i in range(1, srcRst.count + 1):
        reproject(
            source=rasterio.band(srcRst, i),
            destination=rasterio.band(dstRst, i),
            src_transform=srcRst.transform,
            src_crs=srcRst.crs,
            dst_transform=transform,
            dst_crs=custom_crs,
            resampling=Resampling.nearest)

    # Close destination raster
    dstRst.close()

## shapefile to custom wkt crs
def reproject_shapefile_to_wktcrs(input_shapefile, output_shapefile, custom_wkt_crs = (
    'PROJCS["Custom Lambert Azimuthal Equal Area",'
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],'
    'UNIT["Degree",0.017453292519943295],'
    'AUTHORITY["EPSG","4326"]],'
    'PROJECTION["Lambert_Azimuthal_Equal_Area"],'
    'PARAMETER["latitude_of_center",0],'
    'PARAMETER["longitude_of_center",115],'
    'UNIT["Meter",1],'
    'AUTHORITY["Custom_CRS","1001"]]'
    )):
    # Read the input shapefile
    gdf = gpd.read_file(input_shapefile)

    # Define the custom CRS using WKT
    custom_crs = CRS.from_string(custom_wkt_crs)

    # Reproject the GeoDataFrame to the custom CRS
    gdf_reprojected = gdf.to_crs(custom_crs)

    # Save the reprojected GeoDataFrame to a new shapefile
    gdf_reprojected.to_file(output_shapefile)

# reclassify
def reclassify(input_file, output_file, reclassify_values):
    # Open the input TIFF file and read the first band as a NumPy array
    with rasterio.open(input_file) as src:
        raster = src.read(1)

        # Create a copy of the raster array to store the reclassified values
        reclassified_raster = np.copy(raster)

        # Loop through the reclassification dictionary and update the pixel values
        for new_value, values in reclassify_values.items():
            if isinstance(values, int):
                values = [values]
            mask = np.isin(raster, values)
            reclassified_raster[mask] = new_value

        # Copy the metadata from the source dataset
        meta = src.meta.copy()

        # Update the metadata for the modified dataset
        meta.update(dtype=rasterio.int32, compress='lzw')

        # Save the modified raster array as a new TIFF file
        with rasterio.open(output_file, "w", **meta) as dest:
            dest.write(reclassified_raster.astype(rasterio.int32), 1)


def combine_geotiffs(file1, file2, output_file):
    # Open the two GeoTIFF files
    with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
        # Read the data into numpy arrays
        band1 = src1.read(1)
        band2 = src2.read(1)

    # Use numpy's logical_or function to combine the arrays
    combined = np.logical_or(band1, band2).astype(rasterio.uint8)

    # Get the metadata from the first file
    meta = src1.meta

    # Update the metadata for the new file
    meta.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )

    # Write the combined array to a new GeoTIFF file
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(combined, 1)
       
            
def replace_nan_with_zero(input_file, output_file):
    with rasterio.open(input_file) as src:
        img = src.read(1, masked=True)  # read the first band with masking

        # Replace NaN and nodata values with 0
        img = np.where(img.mask, 0, img.filled(0))

        # Define the profile for the output file
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw'
        )

        # Write the output file
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(img.astype(rasterio.float32), 1)    
            
                            
def mask_tif_select_nodata(input_tif, mask_tif, output_tif, nodata_value=[0]):
    with rasterio.open(input_tif) as src:
        with rasterio.open(mask_tif) as mask:
            mask_data = mask.read()
            out_image = src.read()
            
            # Reshape the mask data to match the dimensions of the output image
            mask_data = np.broadcast_to(mask_data, out_image.shape)
            
            if isinstance(nodata_value, list):
                nodata_value = np.array(nodata_value)
            
            # Create a boolean mask of pixels equal to nodata_value
            mask_pixels = np.all(mask_data == nodata_value, axis=0)
            
            # Apply the mask to the output image
            out_image[:, mask_pixels] = nodata_value
            
            out_meta = src.meta
            out_meta.update({"nodata": nodata_value, "compress": "lzw", "predictor": 2})
            
            with rasterio.open(output_tif, "w", **out_meta) as dest:
                dest.write(out_image)
                
 
def change_sub_zero_to_zero(input_tif_path, output_tif_path):
    try:
        # Open the input GeoTIFF
        with rasterio.open(input_tif_path) as src:
            # Read the raster data (assuming a single-band raster)
            raster = src.read(1)
            
            # Replace values below 0 with 0
            modified_raster = np.maximum(raster, 0)
            
            # Create a copy of the source dataset with the modified data
            with rasterio.open(
                output_tif_path,
                'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=src.transform,
                compress='lzw'  # Use LZW compression
            ) as dst:
                dst.write(modified_raster, 1)
    
    except Exception as e:
        print(f"Error: {e}")

def multiply_values_in_geotiff(input_tif_path, output_tif_path, multiplier):
    try:
        # Open the input GeoTIFF
        with rasterio.open(input_tif_path) as src:
            # Read the raster data (assuming a single-band raster)
            raster = src.read(1)
            
            # Multiply the values by the specified multiplier
            modified_raster = raster * multiplier
            
            # Create a copy of the source dataset with the modified data
            with rasterio.open(
                output_tif_path,
                'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=src.transform,
                compress = 'lzw'
            ) as dst:
                dst.write(modified_raster, 1)
        
    
    except Exception as e:
        print(f"Error: {e}") 
                
def mask_tif_with_shapefile(input_tif, shapefile, output_tif, nodata_value=None):
    with rasterio.open(input_tif) as src:
        shapes = gpd.read_file(shapefile)
        mask = geometry_mask(shapes.geometry, out_shape=src.shape, transform=src.transform, invert=True)
        out_image = src.read()
        if nodata_value is None:
            nodata_value = src.nodata
        if nodata_value is None:
            nodata_value = 0
        out_image[:, ~mask] = nodata_value
        out_meta = src.meta
        out_meta.update({"nodata": nodata_value, "compress": "lzw"})  # Add compression setting
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)      

def mask_tif_with_shapefile_select_feature_value(input_tif, shapefile, output_tif, nodata_value=None, feature_name=None, feature_value=None):
    with rasterio.open(input_tif) as src:
        shapes = gpd.read_file(shapefile)

        # Check if both feature_name and feature_value are provided
        if feature_name is not None and feature_value is not None:
            shapes = shapes[shapes[feature_name] == feature_value]

        mask = geometry_mask(shapes.geometry, out_shape=src.shape, transform=src.transform, invert=True)
        out_image = src.read()
        if nodata_value is None:
            nodata_value = src.nodata
        if nodata_value is None:
            nodata_value = 0
        out_image[:, ~mask] = nodata_value
        out_meta = src.meta
        out_meta.update({"nodata": nodata_value, "compress": "lzw"})  # Add compression setting
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)


def mask_tif_select_value(input_tif, mask_tif, output_tif, mask_values, inside=False, nodata_value=None):
    with rasterio.open(input_tif) as src:
        with rasterio.open(mask_tif) as mask:
            mask_data = mask.read()
            out_image = src.read()
            if nodata_value is None:
                nodata_value = src.nodata
            if nodata_value is None:
                nodata_value = 0
            if inside:
                mask = np.isin(mask_data, mask_values)
            else:
                mask = ~np.isin(mask_data, mask_values)
            if len(mask_data.shape) == 2:
                out_image[:, mask] = nodata_value
            else:
                out_image[mask] = nodata_value
            out_meta = src.meta
            out_meta.update({"nodata": nodata_value, "compress": "lzw"})
            with rasterio.open(output_tif, "w", **out_meta) as dest:
                dest.write(out_image)
   
def mask_tif_larger_smaller_value(input_tif, mask_tif, output_tif, is_greater=True, nodata_value=None):
    with rasterio.open(input_tif) as src_input:
        with rasterio.open(mask_tif) as src_mask:
            input_data = src_input.read()
            mask_data = src_mask.read()
            if nodata_value is None:
                nodata_value = src_input.nodata
            if nodata_value is None:
                nodata_value = src_mask.nodata  # Use nodata value from mask TIFF
            if nodata_value is None:
                nodata_value = 0
            if is_greater:
                mask_condition = (mask_data > input_data) | (mask_data == 0)
            else:
                mask_condition = (mask_data < input_data) | (mask_data == 0)
            output_data = np.where(mask_condition, nodata_value, input_data)
            out_meta = src_input.meta
            out_meta.update({"nodata": nodata_value, "compress": "lzw"})
            with rasterio.open(output_tif, "w", **out_meta) as dest:
                dest.write(output_data)
            
                
def get_pixel_size(filename):
    # Open the GeoTIFF file
    with rasterio.open(filename) as src:
        # Get the affine transform of the file
        transform = src.transform

    # Calculate the pixel size
    pixel_size_x = transform.a
    pixel_size_y = -transform.e

    return pixel_size_x, pixel_size_y


def print_tif_dimensions(tif_path):
    with rasterio.open(tif_path) as src:
        width = src.width
        height = src.height
        count = src.count  # Number of bands
        print(f"Width: {width}, Height: {height}, Bands: {count}")
        
        
def snap_raster(input_tif, snap_tif, output_tif):
    with rasterio.open(input_tif) as src:
        profile = src.profile
        metadata = src.read(1)  # Read metadata from the input raster
        
    with rasterio.open(snap_tif) as snap_src:
        snap_profile = snap_src.profile
        
        # Adjust the extent of the input raster to match the snap raster
        profile["width"] = snap_profile["width"]
        profile["height"] = snap_profile["height"]
        
        # Adjust the resolution of the input raster to match the snap raster
        profile["transform"] = snap_profile["transform"]
        
    # Configure output TIFF compression with LWZ method
    profile["compress"] = "lzw"
    
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(metadata, indexes=1)


def buffer_tif(input_file, buffer_distance, output_file, exclude_value=0, included_values=None):
    with rasterio.open(input_file) as src:
        A = src.read(1)
        nodata_value = src.nodata
        A = np.where(A == nodata_value, 0, A)  # Replace NoData values with 0

        # Calculate the buffer distance in pixels
        buffer_pixels = int(buffer_distance / src.res[0])  # Assuming square pixels

        # Apply binary dilation to non-zero values
        unique_vals = np.unique(A)
        if included_values is not None:
            included_vals = np.asarray(included_values)
            unique_vals = np.intersect1d(unique_vals, included_vals)
        unique_vals = unique_vals[unique_vals != exclude_value]

        # Create the buffer mask for non-zero values
        buffer_mask = np.isin(A, unique_vals)

        # Calculate the Euclidean distance transform
        dist_transform = distance_transform_edt(~buffer_mask)

        # Create the buffer result by comparing with the buffer distance
        buffer_result = dist_transform <= buffer_pixels

        # Write the buffered array to a new TIFF file with LZW compression
        profile = src.profile
        profile.update(compress='lzw')

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(buffer_result.astype(np.uint8), 1)
            
            
def find_tif_files_with_polygon(root_folder, gdf_boundary, keyword, minimum_area=0):
    tif_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.tif') and keyword in file:
                tif_path = os.path.join(root, file)
                with rasterio.open(tif_path) as src:
                    # Read the raster data and transform to the boundary's CRS
                    raster = src.read(1)
                    transform = src.transform
                    boundary_geometry = gdf_boundary.unary_union
                    # Create a mask of the raster within the boundary
                    mask = geometry_mask([boundary_geometry], out_shape=raster.shape, transform=transform, invert=True)
                    # Check if there is at least one pixel with a value larger than 0 within the mask
                    if np.any(np.where(mask, raster > 0, False)):
                        tif_files.append(tif_path)
    return tif_files


def calculate_sum_within_geotiff(tif_path):
    total_sum = 0
    
    with rasterio.open(tif_path) as src:
        raster = src.read(1)  # Read the raster data (assuming a single-band raster)
        total_sum = raster.sum()
    
    return total_sum




def create_high_res_tif_from_mask(input_tif, mask_tif, output_tif):
    with rasterio.open(input_tif) as src:
        input_data = src.read(1)
        input_transform = src.transform
        input_crs = src.crs

    with rasterio.open(mask_tif) as msk:
        mask_data = msk.read(1)
        mask_transform = msk.transform

    # Calculate the scaling factors between the two rasters
    scale_x = int(np.ceil(input_transform[0] / mask_transform[0]))
    scale_y = int(np.ceil(input_transform[4] / mask_transform[4]))

    # Calculate the percentage of pixels with value 1 in each region
    region_sums = np.add.reduceat(np.add.reduceat(mask_data, np.arange(0, mask_data.shape[0], scale_y), axis=0),
                                  np.arange(0, mask_data.shape[1], scale_x), axis=1)
    percentages = region_sums / (scale_x * scale_y)

    # Pad percentages array with ones if necessary
    pad_y = input_data.shape[0] - percentages.shape[0]
    pad_x = input_data.shape[1] - percentages.shape[1]
    if pad_y > 0 or pad_x > 0:
        percentages = np.pad(percentages, ((0, pad_y), (0, pad_x)), constant_values=1)

    # Calculate the new values for each region
    new_values = np.nan_to_num(input_data / percentages)

    # Assign these values to the corresponding pixels in the output raster
    output_data = new_values.repeat(scale_y, axis=0).repeat(scale_x, axis=1) * (mask_data == 1)

    # Write the output raster
    with rasterio.open(output_tif, 'w', driver='GTiff', height=output_data.shape[0],
                       width=output_data.shape[1], count=1, dtype=str(output_data.dtype),
                       crs=input_crs, transform=mask_transform, compress='lzw') as dst:
        dst.write(output_data, 1)

def resample_different_extent(high_res_tif_path, low_res_tif_path, output_tif_path):
    # Open the low-resolution GeoTIFF
    with rasterio.open(low_res_tif_path) as low_res_src:
        low_res_data = low_res_src.read()
        low_res_profile = low_res_src.profile

        # Get the extent and resolution of the low-resolution GeoTIFF
        extent = low_res_src.bounds
        resolution = low_res_src.res

        # Open the high-resolution GeoTIFF
        with rasterio.open(high_res_tif_path) as high_res_src:
            # Read the high-resolution data
            high_res_data = high_res_src.read()

            # Resample the high-resolution data to match the extent and resolution
            # of the low-resolution GeoTIFF
            resampled_data = np.empty(low_res_data.shape, dtype=high_res_data.dtype)
            rasterio.warp.reproject(
                source=high_res_data,
                destination=resampled_data,
                src_transform=high_res_src.transform,
                src_crs=high_res_src.crs,
                dst_transform=low_res_src.transform,
                dst_crs=low_res_src.crs,
                resampling=Resampling.nearest  # You can choose a different resampling method if needed
            )

            # Update the profile for the output GeoTIFF
            low_res_profile.update(
                width=low_res_data.shape[2],
                height=low_res_data.shape[1],
                transform=low_res_src.transform,
                dtype=resampled_data.dtype,
                compress="lzw"
            )

            # Write the resampled data to the output GeoTIFF
            with rasterio.open(output_tif_path, 'w', **low_res_profile) as dst:
                dst.write(resampled_data)


def resample_geotiff(input_path_low_res, input_path_high_res, output_path):
    # Open the high-resolution GeoTIFF to get its extent and resolution
    with rasterio.open(input_path_high_res) as high_res_ds:
        high_res_data = high_res_ds.read()
        high_res_extent = high_res_ds.bounds
        high_res_resolution = high_res_ds.res

    with rasterio.open(input_path_low_res) as low_res_ds:
        low_res_extent = low_res_ds.bounds
        low_res_resolution = low_res_ds.res

    # Calculate the target resolution based on the low-resolution and high-resolution resolutions
    target_resolution = low_res_resolution[0] / (np.ceil(low_res_resolution[0] / high_res_resolution[0]))

    # Create an empty array with the same shape as the low-resolution GeoTIFF but with the target resolution
    data = np.empty(shape=(low_res_ds.count,
                           int(low_res_ds.height * low_res_resolution[0] / target_resolution),
                           int(low_res_ds.width * low_res_resolution[1] / target_resolution)),
                    dtype=np.float32)

    # Reproject the high-resolution data to match the low-resolution data
    reproject(
        source=high_res_data,
        destination=data,
        src_transform=high_res_ds.transform,
        src_crs=high_res_ds.crs,
        dst_transform=low_res_ds.transform * low_res_ds.transform.scale(
            (low_res_ds.width / data.shape[-1]),
            (low_res_ds.height / data.shape[-2])
        ),
        dst_crs=low_res_ds.crs,
        resampling=Resampling.nearest
    )

    # Update the transform of the output dataset to reflect the new resolution
    out_transform = rasterio.Affine(target_resolution, 0, low_res_ds.bounds.left, 0, -target_resolution, low_res_ds.bounds.top)

    # Write the reprojected data to the output path
    with rasterio.open(output_path, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2], count=data.shape[0],
                       dtype=str(data.dtype), crs=low_res_ds.crs, transform=out_transform, compress='lzw') as dest:
        dest.write(data)

def process_tifs(input_tif, mask_tif, output_path):
    with rasterio.open(input_tif) as src:
        input_data = src.read(1)
        input_transform = src.transform
        input_crs = src.crs

    with rasterio.open(mask_tif) as msk:
        mask_data = msk.read(1)
        mask_transform = msk.transform

    # Calculate the scaling factors between the two rasters
    scale_x = int(input_transform[0] / mask_transform[0])
    scale_y = int(input_transform[4] / mask_transform[4])

    # Calculate the area of each output pixel in square meters
    mask_x, mask_y = get_pixel_size(mask_tif)
    pixel_area = mask_x*mask_y/10000

    # Reshape the input data to match the mask data resolution
    input_data_rescaled = np.repeat(np.repeat(input_data, scale_y, axis=0), scale_x, axis=1)

    # Calculate the percentage of pixels with value 1 in each region
    region_sums = np.add.reduceat(np.add.reduceat(mask_data, np.arange(0, mask_data.shape[0], scale_y), axis=0),
                                  np.arange(0, mask_data.shape[1], scale_x), axis=1)
    percentages = region_sums / (scale_x * scale_y)

    # Calculate the new values for each region
    new_values = np.nan_to_num(input_data/((scale_x * scale_y)* percentages))

    # Only insert the pixel area if the new value is larger
    new_values_with_area = np.where(new_values > pixel_area, pixel_area, new_values)

    # Assign these values to the corresponding pixels in the output raster
    output_data = new_values_with_area.repeat(scale_y, axis=0).repeat(scale_x, axis=1) * (mask_data == 1)

    # Write the output raster
    with rasterio.open(output_path, 'w', driver='GTiff', height=output_data.shape[0],
                       width=output_data.shape[1], count=1, dtype=str(output_data.dtype),
                       crs=input_crs, transform=mask_transform, compress='lzw') as dst:
        dst.write(output_data, 1)


def resample_and_process(input_path_low_res, input_path_high_res, output_path):
    resample_geotiff(input_path_low_res, input_path_high_res, output_path)
    process_tifs(input_path_low_res, output_path, output_path)


def convert_data_type(file_path, output_file, data_type = np.int8, multiply_factor = 1):
    with rasterio.open(file_path) as src:
        # Read the raster data into a 2D array
        data = src.read(1)
        
        # Multiply every pixel value by multiply_factor and change the data type
        data = (data * multiply_factor).astype(data_type)
        
        # Update the metadata to reflect the new data type and compression method
        meta = src.meta
        meta.update(dtype=data_type, compress='lzw')

        # Write the result to a new file
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(data, 1)


def calculate_sum_within_geotiff(tif_path):
    total_sum = 0
    
    with rasterio.open(tif_path) as src:
        raster = src.read(1)  # Read the raster data (assuming a single-band raster)
        total_sum = np.round(raster.sum()).astype(int)  # Round and convert to integer
    
    return total_sum

def get_top_sum_tif_paths(root_folder, num_top_rows):
    tif_files = [file for file in os.listdir(root_folder) if file.endswith('.tif')]
    data = {'File': [], 'Summed_Value': []}
    
    for tif_file in tif_files:
        for spam_name in spam_names:
            if spam_name in tif_file:
                tif_path = os.path.join(root_folder, tif_file)
                total_sum = calculate_sum_within_geotiff(tif_path)
                
                data['File'].append(tif_path)
                data['Summed_Value'].append(total_sum)
                break  # Exit the loop if a match is found
    
    df = pd.DataFrame(data)
    
    # Sort the DataFrame by 'Summed_Value' in descending order
    df = df.sort_values(by='Summed_Value', ascending=False)
    
    # Get the top 'num_top_rows' rows
    top_rows = df.head(num_top_rows)
    
    # Replace non-finite values (NaN or inf) with a default value (e.g., 0)
    summed_values = top_rows['Summed_Value'].fillna(0)
    
    # Cast the values to integers and then convert to string to remove decimal points
    top_rows['Summed_Value'] = summed_values.astype(int).astype(str)
    
    # Calculate the total sum of 'Summed_Value' in the top rows
    total_top_sum = top_rows['Summed_Value'].astype(int).sum()
    
    return top_rows, total_top_sum


def process_raster_to_csv_and_plot(title, input_tif, output_csv, output_png, included_values=None, show_sum=False):
    with rasterio.open(input_tif) as src:
        spatial_resolution = src.res[0]
        conversion_factor = (spatial_resolution ** 2) / 10000

        band1 = src.read(1)
        unique, counts = np.unique(band1, return_counts=True)
        
        if included_values is not None:
            included_vals = np.asarray(included_values)
            unique = np.intersect1d(unique, included_vals)
        
        data = dict(zip(unique[unique != 0], counts[unique != 0] * conversion_factor))
        df = pd.DataFrame(list(data.items()), columns=['Year', 'Area (ha)'])
        df['Year'] = df['Year']
        df.to_csv(output_csv, index=False)
    
    total_area = df['Area (ha)'].sum()
    plt.style.use('bmh')
    df = pd.read_csv(output_csv)

    ax = df.plot(x='Year', y='Area (ha)', kind='bar')
    
    if show_sum:
        total_area_rounded = round(total_area)
        for i in ax.containers:
            ax.bar_label(i, label_type='edge', rotation=0 if len(df) <= 5 else 90, padding=5, fmt=lambda x: f'{x:.0f}')
        
        ax.text(0.5, 1.03 if len(df) <= 5 else 1.05, f'Total Area: {total_area_rounded} ha', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    else:  # When show_sum is False, still show bar labels
        for i in ax.containers:
            ax.bar_label(i, label_type='edge', rotation=0 if len(df) <= 5 else 90, padding=5, fmt=lambda x: f'{x:.0f}')
    
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    y_max = df['Area (ha)'].max()
    ax.set_ylim(0, y_max * 1.15)

    # Adjust the vertical position of the title and box if show_sum is True
    title_pad = 50 if show_sum and len(df) > 5 else 30  # Adjust the value as needed
    ax.set_title(title, pad=title_pad)

    plt.savefig(output_png, bbox_inches='tight')



def get_tif_area(input_tif, included_values=None):
    with rasterio.open(input_tif) as src:
        spatial_resolution = src.res[0]
        conversion_factor = (spatial_resolution ** 2) / 10000

        band1 = src.read(1)
        unique, counts = np.unique(band1, return_counts=True)
        
        if included_values is not None:
            included_vals = np.asarray(included_values)
            unique = np.intersect1d(unique, included_vals)
        
        data = dict(zip(unique[(unique != 0) & (unique != 255)], counts[(unique != 0) & (unique != 255)] * conversion_factor))
        df = pd.DataFrame(list(data.items()), columns=['Year', 'Area (ha)'])
        df['Year'] = df['Year']
    
    total_area = df['Area (ha)'].sum()
    return total_area
