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
    python anint.py <bathy> <mask> <cl> <space> <power> <radius> <min_points> <max_points>

bathy: Shapefile of bathymetric points</br>
mask: Shapefile of existing mask layer(i.e., lake or river outline)</br>
cl: Shapefile of existing centerline</br>
space: float representing the desired spacing of the grid layer to be produced</br>
power: float representing the power(i.e. squared, cubed, etc) to be applied in IDW calculation</br>
radius: default radius to be applied when searching for nearby points</br>
min_points: minimum number of points to be used in IDW calculation(if not met, radius is increased)</br>
max_points: maximum number of points to be used in IDW calculation(if not met, furthest points are removed)</br>

### Example:
    python anint.py 22024_pts.shp 22024_mask.shp 22024_cl.shp 50 2 50 3 20

## Notes
Mask layer will be used to clip grid points - no points will be generated outside mask. Works with polygons
containing holes.</br>

All files must be Shapefiles at this time. Future releases will expand to other file types.</br>

All files must be in the same CRS. Be sure to take into account the distance units of the chosen CRS when
specifying search radius(i.e., feet vs. degrees).</br>

At this time, there appears to be speed issues when processing large numbers of points. Future releases will
address these concerns.</br>

At some point this will be developed into a QGIS plugin.</br>
