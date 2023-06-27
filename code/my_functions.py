import rasterio
from rasterio.errors import RasterioIOError
from rasterio.merge import merge


def merge_tiff_files(tif_files, output_file):
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
                     "transform": out_trans})
    
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    for src in src_files_to_mosaic:
        src.close()


        
def change_raster_values(input_file, output_file, values, new_value):

    # Open the input TIFF file and read the first band as a NumPy array
    with rasterio.open(input_file) as src:
        raster = src.read(1)

        # Change pixel values in the raster array
        for value in values:
            raster[raster == value] = new_value

        # Copy the metadata from the source dataset
        meta = src.meta.copy()

        # Update the metadata for the modified dataset
        meta.update(dtype=rasterio.int32, nodata=new_value)

        # Save the modified raster array as a new TIFF file
        with rasterio.open(output_file, "w", **meta) as dest:
            dest.write(raster.astype(rasterio.int32), 1)
        
        
