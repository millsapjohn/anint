"""
/***************************************************************************
 anint

 This plugin performs anisotropic inverse-distance weighted interpolation
of bathymetric points, generating a regularly-spaced grid.
                              -------------------
        begin                : 2023-01-25
        copyright            : (C) 2023 by John Millsap
        email                : jmillsapengineer@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'John Millsap'
__date__ = '2023-02-03'
__copyright__ = '(C) 2023 by John Millsap'

import sys
import os
from collections import OrderedDict
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import shapely
import progressbar
import pathlib
import argparse

def main():
    # parse arguments from the command line
    parser = argparse.ArgumentParser(description='anisotropic inverse-distance weighted point interpolation')
    parser.add_argument('bathy', type=fileCheck, help='Shapefile of existing bathymetry layer')
    parser.add_argument('mask', type=fileCheck, help='Shapefile of existing mask layer')
    parser.add_argument('cl', type=fileCheck, help='Shapefile of existing centerline layer')
    parser.add_argument('space', type=float, help='desired grid spacing')
    parser.add_argument('power', type=float, help='power to be used in IDW calculation')
    parser.add_argument('radius', type=float, help='default search radius for nearby points')
    parser.add_argument('min_points', type=int, help='minimum number of points to be used in IDW calculation')
    parser.add_argument('max_points', type=int, help='maximum number of points to be used in IDW calculation')
    parser.add_argument('output', type=fileCheck, help='output Shapefile')
    args = parser.parse_args()
    grid_space = args.space
    power = args.power
    radius = args.radius
    min_points = args.min_points
    max_points = args.max_points
    output = args.output

    # get data source and feature layer for bathymetry, mask (outline), and centerline
    print('loading layers')
    bathy_lyr = gpd.read_file(args.bathy)
    bathy_lyr.name = 'bathymetry layer'
    mask_lyr = gpd.read_file(args.mask)
    mask_lyr.name = 'mask layer'
    cl_lyr = gpd.read_file(args.cl)
    cl_lyr.name = 'centerline layer'
    
    # check that centerline, mask layers only have one feature
    print('checking centerline and mask layers')
    featureCountCheck(cl_lyr)
    featureCountCheck(mask_lyr)

    # check that bathy points have z value
    for index, row in bathy_lyr.iterrows():
        if row['geometry'].has_z != True:
            sys.exit("one or more features in bathymetry layer does not contain a z value")
        else:
            pass

    # check feature types of all three layers
    print("checking feature types")
    featureTypeCheck(bathy_lyr, 'Point')
    mask_lyr = polyTypeCheck(mask_lyr)
    featureTypeCheck(cl_lyr, 'LineString')

    # compare CRS for the three layers - need to be in the same CRS
    print('checking layer CRS compatibility')
    bathy_crs_code = bathy_lyr.crs.to_epsg(min_confidence=20)
    mask_crs_code = mask_lyr.crs.to_epsg(min_confidence=20)
    cl_crs_code = cl_lyr.crs.to_epsg(min_confidence=20)

    # check that layers have the same CRS code
    if bathy_crs_code != mask_crs_code or bathy_crs_code != cl_crs_code:
        sys.exit("mismatched layer CRS")

    # convert cl layer to single strings (if multistring), trim where it intersects mask layer
    cl_list = []
    for i in range(len(cl_lyr.at[0, 'geometry'].coords) - 1):
        coord = cl_lyr.at[0, 'geometry'].coords[i]
        coord_2 = cl_lyr.at[0, 'geometry'].coords[i + 1]
        line = shapely.LineString([coord, coord_2])
        cl_list.append(line)

    # convert list back to a single multistring after trimming (for calculating m values)
    cl_feat = shapely.union_all([cl_list])

    #create bounding boxes for assigning side
    boxes_left, boxes_right, slivers_left, slivers_right = segmentBoxes(cl_list, mask_lyr)

    # calculate m value, d value for bathy layer, add to bathymetry layer fields
    print('\nassigning m, d values to bathymetry points')
    assignMDValues(bathy_lyr, cl_lyr, boxes_left, boxes_right, slivers_left, slivers_right)

    # get bounding box of mask layer - will clip later
    b_box = mask_lyr.total_bounds
    min_x = float(b_box[0])
    min_y = float(b_box[1])
    max_x = float(b_box[2])
    max_y = float(b_box[3])

    # add points to grid layer based on spacing value provided
    print('\ngenerating grid point layer')
    grid_point_list = []
    for i in range((round((max_x - min_x) / grid_space)) - 1):
        for j in range((round((max_y - min_y) / grid_space)) - 1):
            x_coord = min_x + (i * grid_space)
            y_coord = min_y + (j * grid_space)
            pt = shapely.Point(x_coord, y_coord, 0.00)
            grid_point_list.append({'geometry' : pt})
    grid_lyr = gpd.GeoDataFrame(grid_point_list, geometry='geometry', crs=bathy_lyr.crs)

    # clip grid layer
    print('clipping grid layer')
    new_grid_lyr = multiClip(grid_lyr, mask_lyr)

    # calculate side, m value, d value for grid layer, add to grid layer fields
    print('\nassigning m, d values to grid points')
    assignMDValues(new_grid_lyr, cl_lyr, boxes_left, boxes_right, slivers_left, slivers_right)

    # generate new bathy, grid layers with m, d coordinates
    print('\ngenerating new bathy, grid layers')
    md_bathy_lyr = mdPointLayer(bathy_lyr)
    bathy_index = md_bathy_lyr.sindex
    md_grid_lyr = mdPointLayer(new_grid_lyr)

    # perform the anisotropic IDW calculation
    print('\nperforming IDW interpolation on anisotropic coordinates')
    final_grid_lyr = invDistWeight(new_grid_lyr, md_bathy_lyr, md_grid_lyr, power, radius, min_points, max_points, bathy_index)
    print('\nexporting grid points to Shapefile')
    final_grid_lyr.to_file(output)
    print('processing complete')

# creates bounding boxes of each line segment extended to the extent of the mask layer, for identifying
# which points go with which segment
def segmentBoxes(cl_list, mask_lyr):
    box_list_left = [None] * len(cl_list)
    box_list_right = [None] * len(cl_list)
    sliver_list_left = []
    sliver_list_right = []
    poly = mask_lyr.at[0, 'geometry']
    for i in range(len(cl_list)):
        j = 1 
        k = 1
        seg = cl_list[i]
        offset_left = seg.offset_curve(5)
        offset_right = seg.offset_curve(-5)
        if shapely.contains(poly, offset_left) == True or shapely.intersects(poly, offset_left) == True:
            while shapely.contains(poly, offset_left) == True or shapely.intersects(poly, offset_left) == True:
                offset_left = offset_left.offset_curve(j * 5)
                j += 1
        poly_left = shapely.Polygon([seg.coords[0], seg.coords[1], offset_left.coords[1], offset_left.coords[0]])
        box_list_left[i] = poly_left
        if shapely.contains(poly, offset_right) == True or shapely.intersects(poly, offset_right) == True:
            while shapely.contains(poly, offset_right) == True or shapely.intersects(poly, offset_right) == True:
                offset_right = offset_right.offset_curve(k * -5)
                k += 1
        poly_right = shapely.Polygon([seg.coords[0], seg.coords[1], offset_right.coords[1], offset_right.coords[0]])
        box_list_right[i] = poly_right
    for l in range(len(cl_list) - 2):
        if shapely.overlaps(box_list_left[l], box_list_left[l + 1]) == False:
            sliver = shapely.Polygon([cl_list[l].coords[1],
                                      box_list_left[l].exterior.coords[2],
                                      box_list_left[l + 1].exterior.coords[3]
                                      ])
            sliver_list_left.append(sliver)
        if shapely.overlaps(box_list_right[l], box_list_right[l + 1]) == False:
            sliver = shapely.Polygon([cl_list[l].coords[1],
                                      box_list_right[l].exterior.coords[2],
                                      box_list_right[l + 1].exterior.coords[3],
                                      ])
            sliver_list_right.append(sliver)
    return box_list_left, box_list_right, sliver_list_left, sliver_list_right

# function to calculate the z value for grid points using inverse distance weighted method of m, d coordinates
def invDistWeight(grid_lyr, bathy_md_lyr, grid_md_lyr, power, radius, min_points, max_points, sindex):
    point_list = [None] * len(grid_md_lyr)
    x_coords = [None] * len(grid_lyr)
    y_coords = [None] * len(grid_lyr)
    m_coords = [None] * len(grid_md_lyr)
    d_coords = [None] * len(grid_md_lyr)
    bar = progressbar.ProgressBar(min_value=0).start()
    for i in range(len(grid_lyr) - 1):
        x_coords[i] = grid_lyr.at[i, 'geometry'].x
        y_coords[i] = grid_lyr.at[i, 'geometry'].y
    for l in range(len(grid_md_lyr) - 1):
        m_coords[l] = grid_md_lyr.at[l, 'geometry'].x 
        d_coords[l] = grid_md_lyr.at[l, 'geometry'].y
        point_list[l] = grid_md_lyr.at[l, 'geometry']
    for j in range(len(point_list) - 1):
        if shapely.is_empty(point_list[j]) == True:
            continue
        # generate a polygon corresponding to the search radius specified
        buff_coords = ((m_coords[j] - radius, d_coords[j] + (radius / 50)), 
                       (m_coords[j] - radius, d_coords[j] - (radius / 50)), 
                       (m_coords[j] + radius, d_coords[j] - (radius / 50)), 
                       (m_coords[j] + radius, d_coords[j] + (radius / 50)))
        buff = shapely.Polygon(buff_coords)
        # rough estimate of possible matches based on spatial index
        possible_matches_index = list(sindex.intersection(buff.bounds))
        possible_matches = bathy_md_lyr.iloc[possible_matches_index]
        # exact list of matches
        precise_matches = possible_matches[possible_matches.intersects(buff)]
        # if total matches less than specified minimum, expand search radius until criteria is met
        if len(precise_matches) < min_points:
            k = 1
            while len(precise_matches) < min_points:
                new_rad = radius + (k * 5)
                buff_coords = ((m_coords[j] - new_rad, d_coords[j] + (new_rad / 50)), 
                               (m_coords[j] - new_rad, d_coords[j] - (new_rad / 50)), 
                               (m_coords[j] + new_rad, d_coords[j] - (new_rad / 50)), 
                               (m_coords[j] + new_rad, d_coords[j] + (new_rad / 50)))
                buff = shapely.Polygon(buff_coords)
                possible_matches_index = list(sindex.intersection(buff.bounds))
                possible_matches = bathy_md_lyr.iloc[possible_matches_index]
                precise_matches = possible_matches[possible_matches.intersects(buff)]
                k += 1
        # putting this here because I think this minimizes the number of duplicate operations...
        match_dist_dict = {}
        for index, row in precise_matches.iterrows():
            match_dist_dict.update({row['geometry'].distance(point_list[j]) : index})
        # this has to come second in case expanding the search radius above grabs a ton of points
        if len(match_dist_dict) > max_points:
            extra_rows = len(match_dist_dict) - max_points
            match_dist_dict = OrderedDict(sorted(match_dist_dict.items(), key=lambda t: t[0]))
            count = 0
            while count < extra_rows - 1:
                match_dist_dict.popitem(last=True)
                count += 1
        numerator = 0
        denominator = 0
        # iterate through the range of points within the search radius
        matched_point_list = tuple(match_dist_dict.values())
        dist_list = tuple(match_dist_dict.keys())
        for m in range(len(matched_point_list) - 1):
            # get the z value from the indexed point
            bathy_z_val = bathy_md_lyr.at[matched_point_list[m], 'geometry'].z
            # calculate numerator and denominator values for that point
            temp_num = bathy_z_val / (dist_list[m] ** power)
            temp_den = 1 / (dist_list[m] ** power)
            # sum numerator and denominator values
            numerator = numerator + temp_num
            denominator = denominator + temp_den
        if numerator == 0:
            grid_z_val = 0.00
        else:
            grid_z_val = numerator / denominator
        point_list[j] = shapely.Point(x_coords[j], y_coords[j], grid_z_val)
        bar.update(j)

    grid_lyr = grid_lyr.set_geometry(point_list)
    return(grid_lyr)

# function to create new gdf from m, d values
def mdPointLayer(gdf):
    point_list = [None] * len(gdf)
    for index, row in gdf.iterrows():
        m_val = row['m_val']
        d_val = row['d_val']
        if row['geometry'].has_z == True:
            z_val = row['geometry'].z
        else:
            z_val = 0.00
        point_list[index] = ({'geometry' : shapely.Point(m_val, d_val, z_val)})
    new_layer = gpd.GeoDataFrame(point_list, geometry='geometry')
    return new_layer

# simple check to ensure centerline, mask layers don't have multiple features, which would screw up
# other functions
def featureCountCheck(gdf):
    if len(gdf.index) > 1:
        sys.exit(f"multiple features in {gdf}")
    else:
        pass

# simple check to ensure layers only have the correct geometry types
def featureTypeCheck(gdf, geom_type):
    for index, row in gdf.iterrows():
        if row['geometry'].geom_type != geom_type:
            sys.exit(f'geometry in {gdf.name} is not of type {geom_type}')
        else:
            pass

# function to check file extensions
def fileCheck(file):
    ext = os.path.splitext(file)[1][1:]
    if ext != 'shp':
        parser.error('files must be of type Shapefile')
    return file

# more complex typechecking function for polygons. Polygons with holes come in as multilinestrings, which have to
# be converted to a series of polygons. Geopandas' clip function doesn't handle holes correctly.
def polyTypeCheck(gdf):
    for index, row in gdf.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            # length of layer is checked first, so if row 0 is correct, simply pass on the existing gdf
            new_mask_lyr = gdf
        elif row['geometry'].geom_type == 'MultiLineString':
            poly_list = []
            new_mask_lyr = gdf.explode(index_parts=True)
            new_mask_lyr = new_mask_lyr.reset_index(drop = True)
            for index, row in new_mask_lyr.iterrows():
                poly = shapely.polygons(shapely.linearrings(shapely.get_coordinates(row['geometry']).tolist()))
                poly_list.append(poly)
            new_mask_lyr = new_mask_lyr.set_geometry(poly_list)
            new_mask_lyr = new_mask_lyr.reset_index(drop = True)
        else:
            sys.exit('mask layer cannot be converted to polygons')
    return new_mask_lyr

# more comprehensive clip function, since the geopandas clip function doesn't handle holes correctly. Row 0 is the
# outside border, so all points outside that polygon will be removed. Any further rows are interior holes, so 
# points within those polygons will be removed.
def multiClip(point_lyr, mask_lyr):
    point_lyr = point_lyr.clip(mask_lyr.at[0, 'geometry'])
    drop_list = []
    for index, row in mask_lyr.iterrows():
        # don't want to remove all points within border polygon...
        if index == 0:
            continue
        for index2, row2 in point_lyr.iterrows():
            if shapely.contains(mask_lyr.at[index, 'geometry'], point_lyr.at[index2, 'geometry']) == True:
                drop_list.append(index2)
    # reset the index so other functions that assume row and index are the same don't get messed up
    point_lyr = point_lyr.drop(drop_list)
    point_lyr = point_lyr.reset_index(drop = True)

    return point_lyr

# the "signed triangle area" formula returns an area value that is < 0 when the test point is on the
# right side of the line, > 0 when on the left. This determines the "sidedness" of each point.
# smaller absolute value = closer to the line. 
def signedTriangleArea(test_line, test_point):
    vertex_1 = shapely.Point(test_line.coords[0])
    vertex_2 = shapely.Point(test_line.coords[1])
    test_point = shapely.Point(test_point)
    area = ((vertex_2.x - vertex_1.x) - (test_point.y - vertex_2.y)) - ((test_point.x - vertex_2.x) * (vertex_2.y - vertex_1.y))

    return area
# assigning m (distance along centerline) and d (distance to centerline) for each point in a layer
# uses the signedTriangleArea formula above
def assignMDValues(point_layer, cl_layer, boxes_left, boxes_right, slivers_left, slivers_right):
    m_values = pd.Series(dtype='float64')
    d_values = pd.Series(dtype='float64')
    line_coords = [None] * len(list(cl_layer.at[0, 'geometry'].coords))
    for i in range(len(list(cl_layer.at[0, 'geometry'].coords)) - 1):
        line_coords[i] = shapely.Point(list(cl_layer.at[0, 'geometry'].coords)[i])
    cl_string = shapely.LineString(cl_layer.at[0, 'geometry'])
    bar = progressbar.ProgressBar(min_value=0).start()
    for i in range(len(point_layer) - 1):
        p = shapely.Point(point_layer.at[i, 'geometry'])
        found = False
        for j in range(len(boxes_left) - 1):
            if shapely.contains(boxes_left[j], p) == True and shapely.contains(boxes_left[j + 1], p) == False:
                side = 'left'
                temp_line = shapely.LineString([line_coords[j], line_coords[j + 1]])
                temp_m = temp_line.project(p)
                temp_proj = temp_line.interpolate(temp_m)
                d_val = p.distance(temp_proj)
                m_val = cl_string.project(p)
                found == True
            elif shapely.contains(boxes_left[j], p) == True and shapely.contains(boxes_left[j + 1], p) == True:
                side == 'left'
                d_val = p.distance(line_coords[j])
                m_val = cl_string.project(line_coords[j])
                found == True
            else:
                continue
        if found == False:
            for k in range(len(boxes_right) - 1):
                if shapely.contains(boxes_right[k], p) == True and shapely.contains(boxes_right[k + 1], p) == False:
                    side = 'right'
                    temp_line = shapely.LineString([line_coords[k], line_coords[k + 1]])
                    temp_m = temp_line.project(p)
                    temp_proj = temp_line.interpolate(temp_m)
                    d_val = p.distance(temp_proj) * -1 
                    m_val = cl_string.project(p)
                    found = True
                elif shapely.contains(boxes_right[k], p) == True and shapely.contains(boxes_right[k + 1], p) == True:
                    side = 'right'
                    d_val = p.distance(line_coords[k]) * -1
                    m_val = cl_string.project(line_coords[k])
                    found = True
                else:
                    continue
        if found == False:
            for l in range(len(slivers_left) - 1):
                if shapely.contains(slivers_left[l], p) == True:
                    side = 'left'
                    d_val = p.distance(shapely.Point(slivers_left[l].exterior.coords[0]))
                    m_val = cl_string.project(shapely.Point(slivers_left[l].exterior.coords[0]))
                    found = True
        if found == False:
            for m in range(len(slivers_right) - 1):
                if shapely.contains(slivers_right[m], p) == True:
                    side = 'right'
                    d_val = p.distance(shapely.Point(slivers_right[m].exterior.coords[0])) * -1 
                    m_val = cl_string.project(shapely.Point(slivers_right[m].exterior.coords[0]))
                    found = True
        if found == False:
            avg_dist, index = minAvgDist(line_coords, p)
            if p.distance(line_coords[index]) < p.distance(line_coords[index + 1]):
                d_val = p.distance(line_coords[index])
            else:
                d_val = p.distance(line_coords[index + 1])
            temp_line = shapely.LineString([line_coords[index], line_coords[index + 1]])
            m_val = cl_string.project(line_coords[index])
            area = signedTriangleArea(temp_line, p)
            if area < 0:
                d_val = d_val * -1
                side = 'right'
            else:
                side = 'left'
        m_values = pd.concat([m_values, pd.Series(index=[i], data=[m_val])])
        d_values = pd.concat([d_values, pd.Series(index=[i], data=[d_val])])
        bar.update(i)
    point_layer['m_val'] = m_values
    point_layer['d_val'] = d_values

    return point_layer

# this function calculates the distance to the nearest vertex if no perpendicular projection
# is available for that point (i.e., convex side of an intersection)
def minAvgDist(line_coords, point):
    for i in range(len(line_coords) - 2):
        test_pt_1 = line_coords[i]
        test_pt_2 = line_coords[i + 1]
        temp_dist_1 = point.distance(test_pt_1)
        temp_dist_2 = point.distance(test_pt_2)
        temp_avg_dist = (temp_dist_1 + temp_dist_2) / 2
        if i == 0:
            avg_dist = temp_avg_dist
            dist = min(temp_dist_1, temp_dist_2)
            index = i
        if temp_avg_dist < avg_dist:
            avg_dist = temp_avg_dist
            dist = min(temp_dist_1, temp_dist_2)
            index = i
    return dist, index

if __name__ == "__main__":
    main()
