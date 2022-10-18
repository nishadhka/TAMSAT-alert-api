#------------------------------------------------------------------------------
# Install relevant packages
#------------------------------------------------------------------------------
# Load relevant packages
import datetime as dtmod
import numpy as np
import xarray as xr
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature
import scipy.stats
import get_file_path

# python filepath forecast_date poi_start poi_end shapefilepath
# python F:/Ewan_SM_data_260121/TAMSAT_ALERT_API/NDMA_forecasts.py 20210319 20210301 20210531 C:/Users/vlbou/Dropbox/TAMSAT/GIS/KEN_adm2.shp 33.5 42.0 -4.8 5.5
#
#------------------------------------------------------------------------------
# Setup path to data and storage
#------------------------------------------------------------------------------
# Path of current .py file (all data and outputs will be saved here)
file_path = os.path.dirname(get_file_path.__file__)

#------------------------------------------------------------------------------
# Setup dates
#------------------------------------------------------------------------------
# Define forecast / POI dates
forecast_date = dtmod.datetime.strptime(str(sys.argv[1]), '%Y%m%d')
poi_start = dtmod.datetime.strptime(str(sys.argv[2]), '%Y%m%d')
poi_end = dtmod.datetime.strptime(str(sys.argv[3]), '%Y%m%d')

# Define climatological period
clim_start_year = 2005
clim_end_year = 2019

#------------------------------------------------------------------------------
# Load location data
#------------------------------------------------------------------------------
# Shapefile (county boundaries)
shape = Reader(sys.argv[4])

# Bounding box
lon_min = sys.argv[5]
lon_max = sys.argv[6]
lat_min = sys.argv[7]
lat_max = sys.argv[8]

#------------------------------------------------------------------------------
# Wrapper
#------------------------------------------------------------------------------
def shape_wrapper():
    forecast_stamp, poi_stamp, poi_str, loc_stamp = date_stamps(forecast_date, poi_start, poi_end, lon_min, lon_max, lat_min, lat_max)
    clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr = load_data(poi_stamp, forecast_stamp, loc_stamp)
    anom_map_plot(clim_mean_wrsi_xr, ens_mean_wrsi_xr, poi_stamp, forecast_stamp, clim_start_year, clim_end_year, poi_str, loc_stamp)
    prob_map_plot(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp)
    
#------------------------------------------------------------------------------
# Date stamps for output files
#------------------------------------------------------------------------------
def date_stamps(forecast_date, poi_start, poi_end, lon_min, lon_max, lat_min, lat_max):
    # Forecast date stamps
    forecast_stamp = forecast_date.strftime("%Y%m%d")    
    # POI stamps
    start_month = poi_start.month
    end_month = poi_end.month
    poi_months = np.arange(start_month, end_month + 1, 1)
    poi_year = poi_start.year
    # POI string
    poi_str = ""
    for mo in np.arange(0,len(poi_months)):
        tmp_date = dtmod.datetime(2020,poi_months[mo],1).strftime("%b")[0]
        poi_str += tmp_date
    poi_stamp = poi_str + str(poi_year)
    # Location stamp
    loc_stamp = str(lon_min) + "_" + str(lon_max) + "_" + str(lat_min) + "_" + str(lat_max)
    # Return stamps
    return forecast_stamp, poi_stamp, poi_str, loc_stamp

#------------------------------------------------------------------------------
# Load relevant data
#------------------------------------------------------------------------------
def load_data(poi_stamp, forecast_stamp, loc_stamp):
    # Load forecast files
    clim_mean_wrsi_xr = xr.open_dataarray(file_path+"/outputs/clim_mean_wrsi_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
    clim_sd_wrsi_xr = xr.open_dataarray(file_path+"/outputs/clim_sd_wrsi_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
    ens_mean_wrsi_xr = xr.open_dataarray(file_path+"/outputs/ens_mean_wrsi_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
    ens_sd_wrsi_xr = xr.open_dataarray(file_path+"/outputs/ens_sd_wrsi_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
    # Return stamps
    return clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr

#------------------------------------------------------------------------------
# Make plots
#------------------------------------------------------------------------------
# Plot forecast anomaly map
def anom_map_plot(clim_mean_wrsi_xr, ens_mean_wrsi_xr, poi_stamp, forecast_stamp, clim_start_year, clim_end_year, poi_str, loc_stamp):
    # Extract lons and lats for plotting axies
    lons = clim_mean_wrsi_xr['longitude'].values
    lats = clim_mean_wrsi_xr['latitude'].values
    # Calculate max values to standardised colorbars on both plots
    vmax = np.nanmax([clim_mean_wrsi_xr, ens_mean_wrsi_xr])
    # Calculate percent anomaly    
    percent_anom = (ens_mean_wrsi_xr / clim_mean_wrsi_xr) * 100
    # Colormap setup - make 'bad' values grey
    BrBG_cust = matplotlib.cm.get_cmap("BrBG")
    BrBG_cust.set_bad(color = "silver")
    RdBu_cust = matplotlib.cm.get_cmap("RdBu")
    RdBu_cust.set_bad(color = "silver")
    # Build plot
    fig = plt.figure(figsize = (32,10))
    # Plot climatology
    clim_plt = fig.add_subplot(131, projection = ccrs.PlateCarree())
    clim_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    clim_plt.pcolormesh(lons, lats, clim_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    clim_gl = clim_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    clim_gl.xlabels_top = False
    clim_gl.ylabels_right = False
    clim_gl.xlabel_style = {'size': 18}
    clim_gl.ylabel_style = {'size': 18}
    clim_gl.xformatter = LONGITUDE_FORMATTER
    clim_gl.yformatter = LATITUDE_FORMATTER
    clim_plt.add_geometries(shape.geometries(), crs=ccrs.PlateCarree(), edgecolor='k', linewidth = 0.7, facecolor='none')
    clim_plt.set_title('SM (beta) climatology\n' + poi_str + ' ' + str(clim_start_year) + '-' + str(clim_end_year), fontsize = 20)
    clim_cb = plt.pcolormesh(lons, lats, clim_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    clim_cb = plt.colorbar(clim_cb)
    clim_cb.ax.tick_params(labelsize=18)
    clim_plt.set_aspect("auto", adjustable = None)
    clim_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    clim_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    clim_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Plot forecast
    ens_plt = fig.add_subplot(132, projection = ccrs.PlateCarree())
    ens_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    ens_plt.pcolormesh(lons, lats, ens_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    ens_gl = ens_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    ens_gl.xlabels_top = False
    ens_gl.ylabels_right = False
    ens_gl.xlabel_style = {'size': 18}
    ens_gl.ylabel_style = {'size': 18}
    ens_gl.xformatter = LONGITUDE_FORMATTER
    ens_gl.yformatter = LATITUDE_FORMATTER
    ens_plt.add_geometries(shape.geometries(), crs=ccrs.PlateCarree(), edgecolor='k', linewidth = 0.7, facecolor='none')
    ens_plt.set_title('SM (beta) forecast for ' + poi_stamp + "\nIssued "+ forecast_stamp, fontsize = 20)
    ens_cb = plt.pcolormesh(lons, lats, ens_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    ens_cb = plt.colorbar(ens_cb)
    ens_cb.ax.tick_params(labelsize=18)
    ens_plt.set_aspect("auto", adjustable = None)
    ens_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    ens_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    ens_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Plot anomaly
    anom_plt = fig.add_subplot(133, projection = ccrs.PlateCarree())
    anom_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    anom_plt.pcolormesh(lons, lats, percent_anom.T, vmin = 50, vmax = 150, cmap = RdBu_cust)
    anom_gl = anom_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    anom_gl.xlabels_top = False
    anom_gl.ylabels_right = False
    anom_gl.xlabel_style = {'size': 18}
    anom_gl.ylabel_style = {'size': 18}
    anom_gl.xformatter = LONGITUDE_FORMATTER
    anom_gl.yformatter = LATITUDE_FORMATTER
    anom_plt.add_geometries(shape.geometries(), crs=ccrs.PlateCarree(), edgecolor='k', linewidth = 0.7, facecolor='none')
    anom_plt.set_title('SM (beta) % anomaly for ' + poi_stamp + "\nIssued "+ forecast_stamp, fontsize = 20)
    anom_cb = plt.pcolormesh(lons, lats, percent_anom.T, vmin = 50, vmax = 150, cmap = RdBu_cust)
    anom_cb = plt.colorbar(anom_cb)
    anom_cb.ax.tick_params(labelsize=18)
    anom_plt.set_aspect("auto", adjustable = None)
    anom_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    anom_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    anom_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Save and show
    plt.savefig(file_path+"/outputs/map_plot"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+"_shape.png")
    plt.close()

# Plot probability of lower tercile map
def prob_map_plot(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp):
    # Extract lons and lats for plotting axies
    lons = clim_mean_wrsi_xr['longitude'].values
    lats = clim_mean_wrsi_xr['latitude'].values
    lower_thresh = 0.33
    # Calculate probability of lower tercile soil moisture
    a = scipy.stats.norm(clim_mean_wrsi_xr, clim_sd_wrsi_xr).ppf(0.33)
    b_lower = scipy.stats.norm(ens_mean_wrsi_xr, ens_sd_wrsi_xr).cdf(a)
   # Colormap setup - make 'bad' values grey
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap([c('green'), c('palegreen'), lower_thresh - 0.05, c('white'), c('white'), lower_thresh + 0.05, c('yellow'), c('brown')])
    rvb_cust = matplotlib.cm.get_cmap(rvb)
    rvb_cust.set_bad(color = "silver")
    # Extract lons and lats for plotting axies
    lons = clim_mean_wrsi_xr['longitude'].values
    lats = clim_mean_wrsi_xr['latitude'].values    
    # Build plot
    fig = plt.figure(figsize = (10,10))
    # Plot climatology
    prob_plt = fig.add_subplot(111, projection = ccrs.PlateCarree())
    prob_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    prob_plt.pcolormesh(lons, lats, b_lower.T, vmin = 0, vmax = 1, cmap = rvb_cust)
    prob_gl = prob_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    prob_gl.xlabels_top = False
    prob_gl.ylabels_right = False
    prob_gl.xlabel_style = {'size': 18}
    prob_gl.ylabel_style = {'size': 18}
    prob_gl.xformatter = LONGITUDE_FORMATTER
    prob_gl.yformatter = LATITUDE_FORMATTER
    prob_plt.add_geometries(shape.geometries(), crs=ccrs.PlateCarree(), edgecolor='k', linewidth = 0.7, facecolor='none')
    prob_plt.set_title('Probability of lower tercile SM\n' + poi_stamp + " Issued "+ forecast_stamp, fontsize = 20)
    prob_cb = plt.pcolormesh(lons, lats, b_lower.T, vmin = 0, vmax = 1, cmap = rvb_cust)
    prob_cb = plt.colorbar(prob_cb)
    prob_cb.ax.tick_params(labelsize=18)
    prob_plt.set_aspect("auto", adjustable = None)
    prob_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    prob_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    prob_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Save and show
    plt.savefig(file_path+"/outputs/prob_map_plot"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+"_shape.png")
    plt.close()

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict) 

#------------------------------------------------------------------------------
# Auto-run
#------------------------------------------------------------------------------
if __name__ == '__main__':
    shape_wrapper()