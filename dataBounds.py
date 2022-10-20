import numpy as np
import matplotlib.pyplot as plt
import h5py as h5 
import pandas as pd
# from shapely.geometry import Point
# from geopandas import geoDataFrame 
print ("Imported")
fn = "/workspace/rapidsar_test_data/south_yorkshire/datacrop_20220204.h5"

f = h5.File(fn, 'r')
print (list(f.keys()))

lat = np.array(f['Latitude']).flatten()
lon = np.array(f['Longitude']).flatten()

print (lat[0], lon[0])
print (lat[0], lon[-1])
print (lat[-1], lon[0])
print (lat[-1], lon[-1])

print ("DONE")