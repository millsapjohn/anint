# anint
Anistropic inverse-distance weighted interpolation

## Purpose
This script takes an irregularly-spaced point layer, mask layer, and centerline layer, and returns a regular
grid of interpolated points. Rather than using a simple inverse-distance weighted algorithm, this script first
generates new anistropic coordinates based on the provided centerline (distance along CL, distance to CL), and
interpolates based on those coordinates. Developed specifically for processing bathymetric data, but could be
applied to other uses.

## Dependencies
Geopandas</br>
Pandas</br>
Shapely</br>
progressbar</br>
argparse</br>

## Arguments
    python anint.py <bathy> <mask> <cl> <space> <power> <radius> <min_points> <max_points> <output>

bathy: Shapefile of bathymetric points</br>
mask: Shapefile of existing mask layer(i.e., lake or river outline)</br>
cl: Shapefile of existing centerline</br>
space: float representing the desired spacing of the grid layer to be produced</br>
power: float representing the power(i.e. squared, cubed, etc) to be applied in IDW calculation</br>
radius: default radius to be applied when searching for nearby points</br>
min_points: minimum number of points to be used in IDW calculation(if not met, radius is increased)</br>
max_points: maximum number of points to be used in IDW calculation(if not met, furthest points are removed)</br>
output: name of the Shapefile to be exported</br>

### Example:
    python anint.py 22024_pts.shp 22024_mask.shp 22024_cl.shp 50 2 50 3 20 22024_grid.shp

## Notes
Centerline must extend beyond the first/last points in the bathymetry layer for accurate results.</br>

Currently the search area is actually a rectangle, defined as l = 2x(radius), w = radius/5; or in other
words, a l/w ratio of 10. The idea is to bias toward points that are longitudinally similar (have similar d 
values) over those that are latitudinally similar (have similar m values). For this reason the distance parameter
in the actual IDW calculation, is also divided by 10.

Mask layer will be used to clip grid points - no points will be generated outside mask. Works with polygons
containing holes.</br>

All files must be Shapefiles at this time. Future releases will expand to other file types.</br>

All files must be in the same CRS. Be sure to take into account the distance units of the chosen CRS when
specifying search radius(i.e., feet vs. degrees).</br>

At some point this will be developed into a QGIS plugin.</br>
