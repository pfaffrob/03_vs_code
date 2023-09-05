import geopandas as gpd
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
        out_meta.update({"nodata": nodata_value})
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
            out_meta.update({"nodata": nodata_value})
            with rasterio.open(output_tif, "w", **out_meta) as dest:
                dest.write(out_image)           
            
                
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