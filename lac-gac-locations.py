# -*- coding: utf-8 -*-
#
# This script is used to process LAC and GAC scenes for a comparison analysis. It will create csv-files with mean-values
# from selected regions which can be plotted using the plot_lac_gac.ipynb (Jupyter Notebook) script.
#
# Dir structure:    root_dir/data/lac/year/lac_scene.nc
#                   root_dir/data/gac/year/scfv/gac_scfv_scene.nc
#                   root_dir/data/gac/year/scfg/gac_scfg_scene.nc
# File:             lac_gac.py
# Synopsis:         python lac_gac.py
#
# Author:
# Elias Frey, RSGB/Unibe
# Date: 19.06.2024

""" Compare ESA Local Area Coverage (LAC) and Global Coverage (GAC) data """

import os
import csv
from datetime import datetime
import numpy as np
import xarray as xr
from collections import defaultdict


def get_area(area='all'):
    """
    Select area(s)
    Author: Elias Frey, RSGB/UniBE, 20240404
    :param area: All or specific region
    :return: Dictionary with coordinates of selected regions
    """
    coords_dict = {
        'Alaska_USA': [66.16, -153.37],                # Gates of the Arctic National Park, forest
        'Auyuittuq_Canada': [67.5, -65],       # Auyuittuq National Park, mountains
        'San_Rafael_South_America': [-46.76, 73,55],           # Parque Nacional Laguna San Rafael, glacier
        'Berlevåg_Norway': [70.85, 29.08],      # urban, desert
        'Salekhard_Russia': [66.32, 66.37],           # urban, forest
        'Siberia_Russia': [63.32, 115.21],         # forest
    }
    if area == 'all':
        return coords_dict
    else:
        selected_area = coords_dict[area]
        return selected_area


def get_coords(input_dict, area):
    """
    Extract coordinate(s)
    Author: Elias Frey, RSGB/UniBE, 20240404
    :param input_dict: Dictionary containing areas and coordinates
    :param area: Specified region
    :return: Min latitude and longitude
    """
    area_coordinates = input_dict[area]
    min_lat = area_coordinates[0]
    min_lon = area_coordinates[1]
    return min_lat, min_lon


def load_data(data_path):
    """
    Load datasets using xarray
    Author: Elias Frey, RSGB/UniBE, 20240404
    :param data_path: Path of dataset
    :return: Xarray dataset
    """
    data = xr.open_dataset(data_path, decode_coords='all')

    return data


def reproject_lac(lac):
    """
    Reproject LAC data into GAC projection
    Author: Elias Frey, RSGB/UniBE, 20240404
    :param lac: Lac dataset
    :return: Reprojected LAC dataset
    """
    # Assign propre CRS using rioxarray
    lac = lac.rio.write_crs(lac.spatial_ref.attrs['crs_wkt'])
    # Reproject into GAC EPSG 4326
    lac_reprojected = lac.rio.reproject("EPSG:4326")
    # Update LAC dims
    lac_reprojected = lac_reprojected.rename_dims({'y': 'lat', 'x': 'lon'})
    lac_reprojected = lac_reprojected.rename_vars({'y': 'lat', 'x': 'lon'})

    return lac_reprojected


def crop_gac(gac, lac_reprojected):
    """
    Crop GAC data into LAC lat/lon extent
    Author: Elias Frey, RSGB/UniBE, 20240404
    :param gac: GAC dataset
    :param lac_reprojected: Reprojected LAC dataset
    :return GAC dataset cropped to LAC extent
    """
    # Define LAC coordinate boundaries
    min_lon = lac_reprojected.lon.min().values
    min_lat = lac_reprojected.lat.min().values
    max_lon = lac_reprojected.lon.max().values
    max_lat = lac_reprojected.lat.max().values
    # Clip GAC to LAC coordinate boundaries
    gac_cropped = gac.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

    return gac_cropped


def resample_lac(lac_reprojected, gac_template, method='nearest'):
    """
    Resample LAC (1km) into GAC (4km) resolution
    Author: Elias Frey, RSGB/UniBE, 20240404
    :param lac_reprojected: Reprojected LAC dataset
    :param gac_template: GAC file as template to get resolution
    :param method: Method for interpolation
    :return: Resampled LAC dataset
    """
    # lon_bins = gac_cropped['lon'].values
    # lat_bins = gac_cropped['lat'].values
    lac_resampled = lac_reprojected.interp(lon=gac_template['lon'], lat=gac_template['lat'], method=method)

    return lac_resampled


def lac_flags(ds):
    """
    Unpack byte packed flags using the flag_meaning attribute
    Author: Helga Weber, RSGB/UniBE, 20230201
    Adapted: Elias Frey, RSGB/UniBE, 20240404
    :param ds: Xarray LAC dataset
    :return: LAC dataset mask
    """
    flags = ds.quality_flags.attrs['flag_meaning'].split()
    masks = np.unpackbits(ds.quality_flags.astype('uint8'), axis=0)
    out = xr.Dataset(data_vars={f: (('lat', 'lon'), masks[i] == 1) for i, f in enumerate(flags)},
                     coords={'lon': ds.quality_flags['lon'], 'lat': ds.quality_flags['lat']})
    return out


def gac_flags(da):
    """
    Unpack byte packed flags using the flag_meaning attribute
    Author: Helga Weber, RSGB/UniBE, 20230201
    Adapted: Elias Frey, RSGB/UniBE, 20240404
    :param da: Xarray GAC data array
    :return: GAC data array mask
    """
    mask_all = (
        # (da != 0) &
        (da != 205) &
        (da != 206) &
        (da != 215) &
        (da != 210) &
        (da != 250) &
        (da != 251) &
        (da != 252) &
        (da != 253) &
        (da != 254) &
        (da != 255)
    )
    da_mask = xr.where(mask_all, da, float('nan'))
    return da_mask


def apply_mask(dataset, data_type):
    """
    Mask dataset with quality_flag condition
    Author: Elias Frey, RSGB/UniBE, 20240404
    :param dataset: Xarray dataset
    :param data_type: LAC or GAC data type
    :return: Masked dataset LAC or GAC
    """
    if data_type == 'lac':
        flag_mask = lac_flags(dataset)
        # Apply mask on dataset according to defined condition (default: good)
        ds_masked = dataset.where(flag_mask['good'] == 1)
        # ds_masked = xr.where(flag_mask['good'] == 1, dataset, float('nan'))
        # ds_masked = xr.where(flag_mask['questionable'] == 1, dataset, ds_masked)
    elif data_type == 'gac':
        ds_masked = gac_flags(dataset)
    else:
        raise ValueError("Unsupported data_type. Use 'lac' or 'gac'.")

    return ds_masked


def filter_scf(da, con, scf):
    """
    Mask dataset with quality_flag condition (not used)
    Author: Elias Frey, RSGB/UniBE, 20240404
    :param da: Dataset
    :param con: Condition
    :param scf: SCF value threshold
    :return: Filtered dataset based on conditions
    """
    if con == '>=':
        da_filtered = da.where(da >= scf)
    elif con == '>':
        da_filtered = da.where(da > scf)
    elif con == '<=':
        da_filtered = da.where(da <= scf)
    elif con == '<':
        da_filtered = da.where(da < scf)
    elif con == '==':
        da_filtered = da.where(da == scf)
    elif con == '==':
        da_filtered = da.where(da != scf)
    else:
        raise ValueError("Unsupported condition. Use '>=', '>', '<=', '<', or '=='.")

    return da_filtered


def create_extent(ds, min_latitude, min_longitude, area):
    """
    Create extent of dataset representing the selected area
    Author: Elias Frey, RSGB/UniBE, 20240405
    :param ds: Dataset
    :param min_latitude: Min latitude of selected area
    :param min_longitude: Min longitude of selected area
    :param area: Selected area
    :return: Dataset cropped to extent of selected area
    """
    latitude_max = min_latitude + (0.05 * area)
    longitude_max = min_longitude + (0.05 * area)
    extent = ds.sel(lon=slice(min_longitude, longitude_max), lat=slice(min_latitude, latitude_max))
    extent = extent.isel(lon=slice(0, area), lat=slice(0, area))

    return extent


def extract_date(datatype, filename):
    """
    Extract date from file name
    Author: Elias Frey, RSGB/UniBE, 20240405
    :param datatype: LAC or GAC
    :param filename: File name of scene
    :return: Date and satellite type of scene
    """
    if datatype == 'lac':
        parts = filename.split('_')
        data_date = parts[5][:8]
        data_satellite = parts[3]

    elif datatype == 'gac':
        parts = filename.split('-')
        data_date = parts[0]
        data_satellite = parts[4].split('_')[1].lower()

    else:
        raise ValueError(f'Wrong datatype <{datatype}>, must be lac or gac')

    return data_date, data_satellite


def build_datadict(data_directory):
    """
    Create data dictionary of all available LAC/GAC scenes
    Author: Elias Frey, RSGB/UniBE, 20240405
    :param data_directory: Main directory of scenes
    :return: Data dictionary containing all scenes
    """
    scenes_dict = defaultdict(lambda: {'lac': defaultdict(list), 'gac': defaultdict(list)})

    for datatype in os.listdir(data_directory):
        datatype_dir = os.path.join(data_directory, datatype)

        for year_dir in os.listdir(datatype_dir):
            year_path = os.path.join(datatype_dir, year_dir)
            if not os.path.isdir(year_path):
                continue

            if datatype_dir.endswith('lac'):
                subdir = year_path
            elif datatype_dir.endswith('gac') and 'scfv' in os.listdir(year_path):
                subdir = os.path.join(year_path, 'scfv')
            else:
                continue

            sat_files = [file for file in os.listdir(subdir) if extract_date(datatype, file)[1] in ('noaa11', 'noaa14')]
            for file in sat_files:
                data_date, satellite = extract_date(datatype, file)
                scenes_dict[data_date][datatype][satellite].append(os.path.normpath(os.path.join(subdir, file)))

    return scenes_dict


def write_csv(values_dictionary, area_dictionary, csv_timestamp):
    """
    Write values from dictionary to a csv file
    :param values_dictionary: Dictionary containing processed values
    :param area_dictionary: Area dictionary with coordinates
    :param csv_timestamp: Processing Timestamp
    """
    for area, coordinates in area_dictionary.items():
        csv_filename = area + "_values_" + csv_timestamp + ".csv"
        os.makedirs(f'csv/{csv_timestamp}', exist_ok=True)
        csv_path = os.path.join(f'csv/{csv_timestamp}', csv_filename)
        # Check if the file exists
        file_exists = os.path.isfile(csv_path)

        # Write the data to the CSV file
        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write header row if the file is newly created
            if not file_exists:
                writer.writerow(['date', 'datatype', 'satellite', 'region', 'data'])
            # Write data rows
            for data_date, platforms in values_dictionary.items():
                for platform, satellites in platforms.items():
                    for satellite, regions in satellites.items():
                        # data_str = ','.join(str(val) for val in values)
                        writer.writerow([data_date, platform, satellite, area, regions[area]])


def read_csv(csv_path):
    """
    Read csv file (not used)
    Author: Elias Frey, RSGB/UniBE, 20240405
    :param csv_path: Path to csv file
    :return: dictionary of csv values
    """
    # Initialize an empty dictionary to store the data
    csv_dict = defaultdict(lambda: defaultdict(dict))

    # Read the data from the CSV file and populate the dictionary
    with open(csv_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header row
        for row in reader:
            date_str, platform, satellite, data_str = row
            # Convert date string to datetime object
            data_date = datetime.strptime(date_str, '%Y%m%d').strftime('%d-%m-%Y')
            # Convert data string to list of floats, handling NaN values
            data = [np.nan if val == 'nan' else float(val) for val in data_str.strip('[]').split(', ')]
            # Update the dictionary
            csv_dict[data_date][platform][satellite] = data

    return csv_dict


# Data directory
data_dir = "data/"
# Create scenes dictionary
selected_scenes = build_datadict(data_directory=data_dir)
# Select area(s)
sel_area = get_area(area='all')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Get GAC template file for interpolation of LAC files
gac_templ = load_data(
    [v for scenes_dictionary in selected_scenes.values() for v in scenes_dictionary.get('gac', {}).values()][0][0])
scene_counter = 0
# Iterate through the original dictionary
for date, data_types in selected_scenes.items():
    values_dict = {date: {}}
    print('Date: ', date)
    for da_type, sats in data_types.items():
        values_dict[date][da_type] = {}
        for sat, file_paths in sats.items():
            values_dict[date][da_type][sat] = {}
            for file_path in file_paths:
                print('Scene counter: --> ', scene_counter)
                # Process LAC scenes
                if da_type == 'lac':
                    lac_ds = load_data(data_path=file_path)
                    lac_rp = reproject_lac(lac=lac_ds)
                    lac_rs = resample_lac(lac_reprojected=lac_rp, gac_template=gac_templ, method='nearest')
                    lac_msk = apply_mask(dataset=lac_rs, data_type='lac')
                    for loc, coords in sel_area.items():
                        if loc not in values_dict[date][da_type][sat]:
                            values_dict[date][da_type][sat][loc] = []
                        # Extract coordinates for selected area
                        lat_min, lon_min = get_coords(input_dict=sel_area, area=loc)
                        # Get values at desired area
                        lac_ext = create_extent(ds=lac_msk,
                                                min_latitude=lat_min,
                                                min_longitude=lon_min,
                                                area=3)
                        # Get mean value of extracted area
                        lac_value = lac_ext['scfv'].mean().values.item()
                        print(loc, lac_value, ' <-- LAC VALUE')
                        # Append values to dictionary
                        values_dict[date][da_type][sat][loc].append(lac_value)
                    lac_ds.close()

                # Process GAG scenes
                elif da_type == 'gac':
                    gac_ds = load_data(data_path=file_path)
                    # gac_cp = crop_gac(gac_ds, lac_templ)
                    gac_msk = apply_mask(dataset=gac_ds, data_type='gac')
                    for loc, coords in sel_area.items():
                        if loc not in values_dict[date][da_type][sat]:
                            values_dict[date][da_type][sat][loc] = []
                        # Extract coordinates for selected area
                        lat_min, lon_min = get_coords(input_dict=sel_area, area=loc)
                        # Get values at desired area
                        gac_ext = create_extent(ds=gac_msk,
                                                min_latitude=lat_min,
                                                min_longitude=lon_min,
                                                area=3)
                        # Get mean value of extracted area
                        gac_value = gac_ext['scfv'].mean().values.item()
                        last_index = 0
                        print(loc, gac_value, ' <-- GAC VALUE')
                        # Append values to dictionary
                        values_dict[date][da_type][sat][loc].append(gac_value)
                    gac_ds.close()
                scene_counter += 1
    # Write values from dictionary to csv file
    write_csv(values_dictionary=values_dict,
              area_dictionary=sel_area,
              csv_timestamp=timestamp)
