# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from geopy import distance
from geopy.point import Point as geo_point
import math
from rdflib.namespace import Namespace
import re
import sys
import stardog
import requests
import pprint
import shapely.wkt
from shapely.geometry import Point, Polygon
import json

from tqdm import tqdm
import warnings
from urllib3.exceptions import InsecureRequestWarning

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap


def compute_orientation(source_longitude, source_latitude, dest_longitude, dest_latitude):
    """
    Compute the yaw orientation between a source point and a destination point based on their longitude and latitude.

    Parameters:
    - source_longitude (float): Longitude of the source point.
    - source_latitude (float): Latitude of the source point.
    - dest_longitude (float): Longitude of the destination point.
    - dest_latitude (float): Latitude of the destination point.

    Returns:
    - yaw (float): The computed yaw orientation in radians.
    """
    delta_longitude = dest_longitude - source_longitude
    delta_latitude = dest_latitude - source_latitude

    yaw = math.atan2(delta_longitude, delta_latitude)

    return yaw

def compute_target_coordinates(source_longitude, source_latitude, yaw, distanceD, direction):
    """
    Compute the target coordinates given a source point, yaw orientation, distance, and direction.

    Parameters:
    - source_longitude (float): Longitude of the source point.
    - source_latitude (float): Latitude of the source point.
    - yaw (float): Yaw orientation in radians.
    - distanceD (float): Distance to the target point.
    - direction (str): Direction to move ('left', 'right', 'front', 'back').

    Returns:
    - (float, float): Tuple of longitude and latitude of the target point.
    """
    source_point = geo_point(source_latitude, source_longitude)
    if direction == 'left':
        target_point = distance.distance(meters=distanceD).destination(source_point, math.degrees(yaw)-90)
    if direction == 'right':
        target_point = distance.distance(meters=distanceD).destination(source_point, math.degrees(yaw)+90)
    if direction == 'front':
        target_point = distance.distance(meters=distanceD).destination(source_point, math.degrees(yaw))
    if direction == 'back':
        target_point = distance.distance(meters=distanceD).destination(source_point, math.degrees(yaw)+180)
        
    target_latitude = target_point.latitude
    target_longitude = target_point.longitude

    return target_longitude, target_latitude

def compute_target_polygon(source_longitude, source_latitude, yaw, distanceFront, distanceSide, distanceBack):
    """
    Compute the coordinates of a polygon representing an area around a source point, given distances and yaw orientation.

    Parameters:
    - source_longitude (float): Longitude of the source point.
    - source_latitude (float): Latitude of the source point.
    - yaw (float): Yaw orientation in radians.
    - distanceFront (float): Distance to extend the polygon in the front direction.
    - distanceSide (float): Distance to extend the polygon sideways.
    - distanceBack (float): Distance to extend the polygon in the backward direction.

    Returns:
    - list: Coordinates of the polygon vertices.
    """
    source_point = geo_point(source_latitude, source_longitude)
    target_point = distance.distance(meters=distanceFront).destination(source_point, math.degrees(yaw))
    target_point_left = distance.distance(meters=distanceSide).destination(target_point, math.degrees(yaw)-90)
    target_point_right = distance.distance(meters=distanceSide).destination(target_point, math.degrees(yaw)+90)
    target_latitude_left = target_point_left.latitude
    target_longitude_left = target_point_left.longitude
    target_latitude_right = target_point_right.latitude
    target_longitude_right = target_point_right.longitude
    target_point_back = distance.distance(meters=distanceBack).destination(source_point, math.degrees(yaw)+180)
    target_point_back_left = distance.distance(meters=distanceSide).destination(target_point_back, math.degrees(yaw)-90)
    target_point_back_right = distance.distance(meters=distanceSide).destination(target_point_back, math.degrees(yaw)+90)
    target_latitude_back_left =  target_point_back_left.latitude
    target_longitude_back_left =  target_point_back_left.longitude
    target_latitude_back_right =  target_point_back_right.latitude
    target_longitude_back_right = target_point_back_right.longitude
    
    return [target_longitude_left, str(target_latitude_left)+',', 
            target_longitude_right, str(target_latitude_right)+',', 
            target_longitude_back_right, str(target_latitude_back_right)+',',
            target_longitude_back_left, str(target_latitude_back_left)+',',
            target_longitude_left, target_latitude_left] 

def verify_end_scene(scene_ID, conn):
    """
    Verify and fetch the final scene details based on the provided scene ID.

    Parameters:
    - scene_ID (str): The ID of the scene to verify.
    - conn (stardog.Connection): Connection to the Stardog database.

    Returns:
    - list: Query results containing the final scene coordinates.
    """
    NS = Namespace('http://www.nuscenes.org/nuScenes/')
    sceneURI = NS['Scene_'+scene_ID]
    egoURI = NS['SceneParticipant_Ego_'+scene_ID]
    get_coordinates_query = f"""
        SELECT ?s1 ?WKT ?WKT1 ?seq
        WHERE{{
            BIND({sceneURI.n3()} as ?s)
            ?seq nus:hasScene ?s.
            FILTER NOT EXISTS {{
                ?s nus:hasNextScene ?s1.
            }}
            ?s nus:hasSceneParticipant ?ego.
            FILTER(CONTAINS(STR(?ego), "Ego_"))
            ?ego nus:hasPosition ?p.
            ?p geo:asWKT ?WKT.
            BIND(?WKT as ?WKT1)
            BIND(?s as ?s1)
        }}
    """
    result = conn.select(get_coordinates_query)
    # pprint.pprint(result)
    coords = result['results']['bindings']
    #pprint.pprint(coords)
    #conn.end
    return coords

def get_egos_coordinates(scene_ID, conn):
    """
    Retrieve the ego vehicle's coordinates for the given scene ID from the database.

    Parameters:
    - scene_ID (str): The scene ID to query.
    - conn (stardog.Connection): Connection to the Stardog database.

    Returns:
    - list: Query results containing ego vehicle's coordinates.
    """
    NS = Namespace('http://www.nuscenes.org/nuScenes/')
    sceneURI = NS['Scene_'+scene_ID]
    egoURI = NS['SceneParticipant_Ego_'+scene_ID]
    # conn.begin()
    get_ego_coordinates_query = f"""
        SELECT ?s1 ?WKT ?WKT1 ?WKT2 ?seq
        WHERE{{
                {{
                BIND({egoURI.n3()} as ?ego)
                BIND({sceneURI.n3()} as ?s)
                ?seq nus:hasScene ?s.
                ?s nus:hasNextScene ?s1.
                ?s1 nus:hasSceneParticipant ?ego1.
                FILTER(CONTAINS(STR(?ego1), "Ego_"))
                ?ego nus:hasPosition ?p.
                ?p geo:asWKT ?WKT.
                ?ego1 nus:hasPosition ?p1.
                ?p1 geo:asWKT ?WKT1.
            }}
            OPTIONAL{{
            ?s1 nus:hasNextScene ?s2.
            ?s2 nus:hasPosition ?p2.
            ?p2 geo:asWKT ?WKT2.
            }}
        }}
    """
    result = conn.select(get_ego_coordinates_query)
    # pprint.pprint(result)
    coords = result['results']['bindings']
    #pprint.pprint(coords)
    #conn.end
    return coords

def ego_coords_to_geometry(ego_coords):
    """
    Convert ego coordinates data to geometrical points.

    Parameters:
    - ego_coords (list): List containing the ego coordinates fetched from the database.

    Returns:
    - tuple: Source point, destination point, next scene ID, and sequence ID.
    """
    source_point = ego_coords[0]['WKT']['value']
    dest_point = ego_coords[0]['WKT1']['value']
    next_scene = ego_coords[0]['s1']['value'].split('http://www.nuscenes.org/nuScenes/Scene_')[1]
    sequence_ID = ego_coords[0]['seq']['value'].split('http://www.nuscenes.org/nuScenes/Sequence_')[1]
    return source_point, dest_point, next_scene, sequence_ID

def compute_distance(source_point, dest_point):
    """
    Compute the distance between two points represented as WKT strings.

    Parameters:
    - source_point (str): WKT string of the source point.
    - dest_point (str): WKT string of the destination point.

    Returns:
    - float: Distance in meters between the two points.
    """
    source_p = re.findall(r"[^A-Z()]", source_point)
    source_p = ''.join(source_p)
    source_p = re.split(" ", source_p)
    source_p.pop(0)
    source_p = [eval(i) for i in source_p]
    source_lon = source_p[0]
    source_lat = source_p[1]
    
    dest_p = re.findall(r"[^A-Z()]", dest_point)
    dest_p = ''.join(dest_p)
    dest_p = re.split(" ", dest_p)
    dest_p.pop(0)
    dest_p = [eval(i) for i in dest_p]
    dest_lon = dest_p[0]
    dest_lat = dest_p[1]
    
    source_coords = (source_lat, source_lon)
    dest_coords = (dest_lat, dest_lon)
    
    return distance.distance(source_coords, dest_coords).meters

def regex_query_process(source_point, dest_point, previous_yaw, distance):
    """
    Process the query based on regular expressions to extract and compute geometry data.

    Parameters:
    - source_point (str): WKT string of the source point.
    - dest_point (str): WKT string of the destination point.
    - previous_yaw (float): Previous yaw orientation in radians.
    - distance (float): Distance to be considered in the computation.

    Returns:
    - tuple: Contains longitude, latitude, new yaw, previous yaw, and computed polygon.
    """
    source_p = re.findall(r"[^A-Z()]", source_point)
    source_p = ''.join(source_p)
    source_p = re.split(" ", source_p)
    source_p.pop(0)
    source_p = [eval(i) for i in source_p]
    source_lon = source_p[0]
    source_lat = source_p[1]
    
    dest_p = re.findall(r"[^A-Z()]", dest_point)
    dest_p = ''.join(dest_p)
    dest_p = re.split(" ", dest_p)
    dest_p.pop(0)
    dest_p = [eval(i) for i in dest_p]
    dest_lon = dest_p[0]
    dest_lat = dest_p[1]
    
    if distance < 0.5:
        yaw = previous_yaw
    else:
        yaw = compute_orientation(source_lon, source_lat, dest_lon, dest_lat)
    polygon = compute_target_polygon(source_lon, source_lat, yaw, 150, 150, 150)
    #polygon = compute_target_polygon(source_lon, source_lat, yaw, 1000, 1000, 1000)
    polygon = " ".join(map(str,polygon))
    polygon = "\"POLYGON ((" + polygon + "))\""
    previous_yaw = yaw
    return source_lon, source_lat, yaw, previous_yaw, polygon

def visualize_polys(source_point, dest_point, previous_yaw, distance):
    """
    Generate polygons for visualization based on source and destination points, incorporating yaw and distance considerations.

    Parameters:
    - source_point (str): WKT string of the source point.
    - dest_point (str): WKT string of the destination point.
    - previous_yaw (float): Previous yaw orientation in radians.
    - distance (float): Distance to be considered in the computation.

    Returns:
    - tuple: Contains longitude, latitude, new yaw, previous yaw, and computed polygon for visualization.
    """
    source_p = re.findall(r"[^A-Z()]", source_point)
    source_p = ''.join(source_p)
    source_p = re.split(" ", source_p)
    source_p.pop(0)
    source_p = [eval(i) for i in source_p]
    source_lon = source_p[0]
    source_lat = source_p[1]
    
    dest_p = re.findall(r"[^A-Z()]", dest_point)
    dest_p = ''.join(dest_p)
    dest_p = re.split(" ", dest_p)
    dest_p.pop(0)
    dest_p = [eval(i) for i in dest_p]
    dest_lon = dest_p[0]
    dest_lat = dest_p[1]
    
    if distance < 0.5:
        yaw = previous_yaw
    else:
        yaw = compute_orientation(source_lon, source_lat, dest_lon, dest_lat)
    polygon = compute_target_polygon(source_lon, source_lat, yaw, 20, 0, 0)
    polygon = " ".join(map(str,polygon))
    polygon = "POLYGON ((" + polygon + "))"
    #polygon = [polygon]
    previous_yaw = yaw
    return source_lon, source_lat, yaw, previous_yaw, polygon

def get_all_dynamic_elements(scene_ID, conn):
    """
    Retrieve all dynamic elements for a given scene from the database.

    Parameters:
    - scene_ID (str): The identifier of the scene.
    - conn (stardog.Connection): Connection to the Stardog database.

    Returns:
    - list: A list of bindings representing the scene's dynamic elements, including their spatial information and types.
    """
    NS = Namespace('http://www.nuscenes.org/nuScenes/')
    sceneURI = NS['Scene_'+scene_ID]
    get_polygon_query = f"""  
        SELECT DISTINCT ?s ?geo ?type ?WKT
    WHERE{{
        BIND({sceneURI.n3()} as ?s)
        ?s nus:hasSceneParticipant ?participant.
        FILTER NOT EXISTS {{ 
            FILTER(CONTAINS(STR(?participant), "Ego_")). 
        }}
        
        ?participant nus:hasPosition ?geo.
        ?geo geo:asWKT ?WKT.
        ?participant nus:isSceneParticipantOf ?par.
        ?par a ?type.
    }}
    """
    result = conn.select(get_polygon_query)
    # pprint.pprint(result)
    elements = result['results']['bindings']
    return elements

def get_elements_in_polygon(scene_ID, polygon, conn):
    """
    Fetch map elements that are within a specified polygon from the database.

    Parameters:
    - scene_ID (str): The identifier of the scene.
    - polygon (str): The polygon (in WKT format) within which to find the map elements.
    - conn (stardog.Connection): Connection to the Stardog database.

    Returns:
    - list: A list of bindings with map elements and their spatial information within the specified polygon.
    """
    NS = Namespace('http://www.nuscenes.org/nuScenes/')
    sceneURI = NS['Scene_'+scene_ID]
    get_polygon_query = f"""  
        SELECT DISTINCT ?mapElement ?geo ?type ?WKT
    WHERE{{
        #{{
        #BIND({sceneURI.n3()} as ?s)
        #BIND({polygon}^^geo:wktLiteral as ?polygon)
        #?s nus:hasSceneParticipant ?participant.
        #?participant nus:hasPosition ?geo.
        #?geo geo:asWKT ?WKT.
        #?geo geof:within ?polygon.
        #?participant nus:isSceneParticipantOf ?par.
        #?par a ?type.


        #}}UNION{{
        BIND({polygon}^^geo:wktLiteral as ?polygon)
        ?hasShapeProperty rdfs:subPropertyOf* nus_map:hasShape.
        ?mapElement ?hasShapeProperty ?geo.
        ?geo geof:within ?polygon.
        ?geo geo:asWKT ?WKT.
        ?mapElement a ?type.
        #}}
        # {{
        #     SELECT ?s 
        #     WHERE{{
        #         ?s a nus:Scene
        #     }}
        #     LIMIT 1
        # }}
    }}
    """
    result = conn.select(get_polygon_query)
    # pprint.pprint(result)
    elements = result['results']['bindings']
    return elements

def dump_visualize_elements(elements, overwrite):
    """
    Write the geometrical data of elements to a file for visualization purposes.

    Parameters:
    - elements (list): The geometrical data of elements to be written.
    - overwrite (bool): If True, overwrite the existing file; append otherwise.

    Returns:
    - tuple: A tuple containing coordinates and types of the elements.
    """
    coords = [element['WKT']['value'] for element in elements]
    types = [element['type']['value'].replace('http://www.nuscenes.org/nuScenes/', '').replace('map/', '') for element in elements]

    mode = 'w' if overwrite else 'a'
    with open("elements.txt", mode) as file:
        for i, coord in enumerate(coords):
            coord_str = coord.strip("['']")
            if i < len(coords) - 1:
                coord_str += ","
            file.write(coord_str + "\n")

    return coords, types

def dump_visualize_debug_elements(elements, overwrite, i, type_to_count):
    """
    Write specific types of geometrical data of elements to a file for debugging.

    Parameters:
    - elements (list): The geometrical data of elements to be written.
    - overwrite (bool): If True, overwrite the existing file; append otherwise.
    - i (int): Index used in the filename to differentiate files.
    - type_to_count (str): The type of elements to be specifically written.

    Returns:
    - None
    """
    coords = [element['WKT']['value'] for element in elements]
    types = [element['type']['value'].replace('http://www.nuscenes.org/nuScenes/', '').replace('map/', '') for element in elements]

    mode = 'w' if overwrite else 'a'
    with open("elements"+str(i)+type_to_count+".txt", mode) as file:
        for coord, type_ in zip(coords, types):
            if type_ == type_to_count:
                coord_str = coord.strip("['']")
                if coord != coords[-1]:
                    coord_str += ","
                print(type_)
                print(coord_str)
                file.write(coord_str + "\n")

    return

def process_elements(elements, map_elements = False):
    """
    Process elements to extract coordinates, types, and optionally the IDs.

    Parameters:
    - elements (list): Elements containing geographical data.
    - map_elements (bool): Flag indicating whether to extract map element IDs.

    Returns:
    - tuple: Coordinates, types, and optionally the IDs of the elements.
    """
    coords = [element['WKT']['value'] for element in elements]
    types = [element['type']['value'].replace('http://www.nuscenes.org/nuScenes/', '').replace('map/', '') for element in elements]
    coords_ids = []
    if map_elements:
        coords_ids = [element['mapElement']['value'] for element in elements]
    return coords, types, coords_ids
    
def subpolygons(source_longitude, source_latitude, yaw, area_size, matrix_width, front_matrix_height, back_matrix_height):
    """
    Generate sub-polygons within a defined area using given parameters.

    Parameters:
    - source_longitude (float): Longitude of the reference point.
    - source_latitude (float): Latitude of the reference point.
    - yaw (float): Orientation angle.
    - area_size (float): The size of each subpolygon area.
    - matrix_width (int): The number of subpolygons along the width.
    - front_matrix_height (int): The number of subpolygons in the front direction.
    - back_matrix_height (int): The number of subpolygons in the back direction.

    Returns:
    - list: A list of subpolygons defined by their geometrical coordinates.
    """
    subpolygons = []
    
    source_lon, source_lat = compute_target_coordinates(source_longitude, source_latitude, yaw, back_matrix_height * area_size - 0.05, 'back')
    matrix_height = front_matrix_height + back_matrix_height
    for row in range(matrix_height):
        lonH, latH = compute_target_coordinates(source_lon, source_lat,
                                                  yaw, (matrix_height - (row+1)) * area_size, 'front')
        for col in range(matrix_width):       
            mid = math.floor(matrix_width/2)
            if col < mid:
                lon, lat = compute_target_coordinates(lonH, latH, yaw, (mid - col) * area_size , 'left')
            elif col > mid:
                lon, lat = compute_target_coordinates(lonH, latH, yaw, (col - mid) * area_size , 'right')
            else:
                lon, lat = lonH, latH
            subpolygon = compute_target_polygon(lon, lat, yaw, area_size, area_size / 2, 0)
            subpolygon = " ".join(map(str,subpolygon))
            subpolygon = "POLYGON ((" + subpolygon + "))"
            subpolygons.append(subpolygon)
    
    return subpolygons

def dump_visualize_polygons(polygons, overwrite, i):
    """
    Write the coordinates of polygons to a file for visualization.

    Parameters:
    - polygons (list): List of polygons to be written.
    - overwrite (bool): If True, overwrite the existing file; append otherwise.
    - i (int): Index used in the filename to differentiate files.

    Returns:
    - None
    """
    mode = 'w' if overwrite else 'a'
    with open("nuscene_comparison/polygons_"+str(i)+".txt", mode) as file:
        for i, poly in enumerate(polygons):
            polygon_str = poly.strip("['']")
            #if i < len(subpolygons) - 1:
            polygon_str += ","
            #print(polygon_str)
            file.write(polygon_str + "\n")
    return

def dump_visualize_polygon(polygon, overwrite, i):
    """
    Write a single polygon's coordinates to a file.

    Parameters:
    - polygon (str): The polygon to be written.
    - overwrite (bool): If True, overwrite the existing file; append otherwise.
    - i (int): Index used in the filename to differentiate files.

    Returns:
    - None
    """
    mode = 'w' if overwrite else 'a'
    with open("polygons_"+str(i)+".txt", mode) as file:
        polygon_str = polygon.strip("['']").replace('"','')
        file.write(polygon_str + "\n")
    return

def dump_visualize_subpolygons(subpolygons, overwrite, i):
    """
    Write the coordinates of subpolygons to a file for visualization.

    Parameters:
    - subpolygons (list): List of subpolygons to be written.
    - overwrite (bool): If True, overwrite the existing file; append otherwise.
    - i (int): Index used in the filename to differentiate files.

    Returns:
    - None
    """
    mode = 'w' if overwrite else 'a'
    with open("nuscene_comparison/subpolygons_"+str(i)+".txt", mode) as file:
        for i, poly in enumerate(subpolygons):
            polygon_str = poly.strip("['']")
            if i < len(subpolygons) - 1:
                polygon_str += ","
            #print(polygon_str)
            file.write(polygon_str + "\n")
    return

def verify_element_positions(coords, types, subpolygons, matrix_width, front_matrix_height, back_matrix_height, intersection_points):
    """
    Verify and assign map elements to their respective positions in a grid layout defined by subpolygons.

    Parameters:
    - coords (list): Coordinates of the elements.
    - types (list): Types of the elements.
    - subpolygons (list): Subpolygons defining the grid layout.
    - matrix_width (int): Width of the grid.
    - front_matrix_height (int): Height of the grid at the front.
    - back_matrix_height (int): Height of the grid at the back.
    - intersection_points (list): Points of intersections relevant to the elements.

    Returns:
    - tuple: Element positions within the grid and the coordinates included in the grid.
    """
    element_positions = [[] for _ in range(len(subpolygons))]
    in_coords = set()
    for i, element in enumerate(coords):
        shape = shapely.wkt.loads(element)
        for j, subpolygon in enumerate(subpolygons):
            subpolygon_shape = shapely.wkt.loads(subpolygon)
            if shape.intersects(subpolygon_shape):
                element_positions[j].append(types[i])
                in_coords.update([element])
    map_elements = ['Lane', 'CarparkArea', 'Walkway']
    for j, subpolygon in enumerate(subpolygons):
            if not any(element in map_elements for element in element_positions[j]):
                subpolygon_shape = shapely.wkt.loads(subpolygon)
                for point in intersection_points:
                    point_shape = shapely.wkt.loads(point)
                    if point_shape.intersects(subpolygon_shape):
                        element_positions[j].append('Intersection')
                        break
            
                
    element_positions = [element_positions[i:i+matrix_width] for i in range(0, len(element_positions), matrix_width)]
    element_positions = element_positions + [[]] * (front_matrix_height + back_matrix_height - len(element_positions))      

    return element_positions, in_coords

import json

def area_embeddings_to_json(element_positions_list, scene_ID_list, next_scene_ID_list, yaw_list, distance_list, country_list, sequence_ID, json_file_name):
    """
    Save the area embeddings along with additional scene information to a JSON file.

    Parameters:
    - element_positions_list (list): List of element positions for each scene.
    - scene_ID_list (list): List of scene IDs.
    - next_scene_ID_list (list): List of next scene IDs correlating to each scene.
    - yaw_list (list): List of yaw orientations for each scene.
    - distance_list (list): List of distances between consecutive scenes.
    - country_list (list): List of countries for each scene.
    - sequence_ID (str): Sequence identifier.
    - json_file_name (str): path to the json file storing the data

    Returns:
    - None
    """
    data = {}

    try:
        with open(json_file_name, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        pass
    
    for i, scene_ID in enumerate(scene_ID_list):
        if i < len(scene_ID_list) - 1:
            data[scene_ID] = {
                "area_embedding": element_positions_list[i],
                "next_scene_ID": next_scene_ID_list[i],
                "orientation_difference": round(math.degrees(yaw_list[i+1]) - math.degrees(yaw_list[i])),
                "distance_between_scenes": round(distance_list[i], 1),
                "country": country_list[i],
                "sequence_ID": sequence_ID
            }
        else:
            data[scene_ID] = {
                "area_embedding": element_positions_list[i],
                "next_scene_ID": next_scene_ID_list[i],
                "orientation_difference": 0,
                "distance_between_scenes": round(distance_list[i], 1),
                "country": country_list[i],
                "sequence_ID": sequence_ID
            }
            
    with open(json_file_name, "w") as file:
        json.dump(data, file)

    return

def get_consecutive_scenes(scene_ID, conn):
    """
    Retrieve consecutive scenes starting from a given scene ID.

    Parameters:
    - scene_ID (str): The starting scene ID.
    - conn (stardog.Connection): Connection to the Stardog database.

    Returns:
    - tuple: A tuple containing lists of scene IDs and their corresponding points.
    """
    NS = Namespace('http://www.nuscenes.org/nuScenes/')
    sceneURI = NS['Scene_'+scene_ID]
    egoURI = NS['SceneParticipant_Ego_'+scene_ID]
    # conn.begin()
    get_egos_pos_query = f"""  
        SELECT ?wkt ?s ?wkt1 ?s1
        WHERE {{
            BIND({sceneURI.n3()} as ?s)
            BIND({egoURI.n3()} as ?ego)
            ?s nus:hasNextScene+ ?s1.
            ?s1 nus:hasSceneParticipant ?ego1.
            FILTER(CONTAINS(STR(?ego1), "Ego"))
            ?ego1 nus:hasPosition ?p1.
            ?p1 geo:asWKT ?wkt1.
            ?ego nus:hasPosition ?p.
            ?p geo:asWKT ?wkt.
        }}"""
    
    result = conn.select(get_egos_pos_query)
    # pprint.pprint(result)
    rows = result['results']['bindings']
    #pprint.pprint(rows)
    #conn.end
    scene_ids = [item['s1']['value'].split('http://www.nuscenes.org/nuScenes/Scene_')[1] for item in rows]
    points = [item['wkt1']['value'] for item in rows]
    scene_ids.insert(0, scene_ID)
    points.insert(0, rows[0]['wkt']['value'])
    
    return scene_ids, points

def visualize_test_scene(test_scene_ID, conn, previous_yaw=0):
    """
    Visualize test scene by querying and processing its related data, then writing it for visualization.

    Parameters:
    - test_scene_ID (str): The ID of the test scene.
    - conn (stardog.Connection): Connection to the Stardog database.
    - previous_yaw (float): The previous yaw orientation, used for continuity in visualization.

    Returns:
    - None
    """
    ego_coords = get_egos_coordinates(test_scene_ID, conn)
    if ego_coords:
        source_point, dest_point, next_scene_ID = ego_coords_to_geometry(ego_coords)
    else:
        end_scene = verify_end_scene(test_scene_ID, conn)
        source_point, dest_point, _ = ego_coords_to_geometry(end_scene)
        next_scene_ID = ''

    source_lon, source_lat, yaw, previous_yaw, polygon = regex_query_process(source_point, dest_point, previous_yaw)
    elements = get_elements_in_polygon(test_scene_ID, polygon, conn)
    dump_visualize_elements(elements, overwrite=True)
    coords, types = process_elements(elements)
    subpolys = subpolygons(source_lon, source_lat, yaw, 5, 5, 8)
    dump_visualize_subpolygons(subpolys,  overwrite=True)
    with open("elements.txt", 'a') as file:
        file.write("," + source_point + "\n")
    #element_positions = verify_element_positions(coords, types, subpolys)
    #area_embeddings_to_json(element_positions, scene_ID, next_scene_ID)
    return

from shapely.geometry import Point
from shapely.geometry import mapping
from shapely.geometry import shape
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.wkt import loads, dumps

def compute_centerline(lane_id, conn, small_poly):
    """
    Computes the centerline of a lane within a specified polygon area, returning geometrical points representing the centerline.

    Parameters:
    - lane_id (str): The identifier of the lane for which to compute the centerline.
    - conn (stardog.Connection): Connection to the Stardog database.
    - small_poly (str): A WKT representation of the polygon within which the centerline is computed.

    Returns:
    - list: A list of WKT strings representing points or geometries that constitute the computed centerline within the specified polygon.
    """
    small_poly_shape = loads(small_poly)
    
    get_centerline_query = f"""  
        SELECT ?connector ?WKT
        WHERE {{
            {{
                BIND(nus_map:{lane_id} as ?lane)
                ?incomingConnectors nus_map:connectsIncomingLane ?lane.
                ?incomingConnectors nus_map:connectorHasPose ?pose.
                ?pose nus_map:poseHasPosition ?point.
                ?point geo:asWKT ?WKT.
                BIND(?incomingConnectors as ?connector).
            }}
            UNION
            {{
                BIND(nus_map:{lane_id} as ?lane)
                ?outgoingConnectors nus_map:connectsOutgoingLane ?lane.
                ?outgoingConnectors nus_map:connectorHasPose ?pose.
                ?pose nus_map:poseHasPosition ?point.
                ?point geo:asWKT ?WKT.
                BIND(?outgoingConnectors as ?connector).
            }}
        }}
        ORDER BY ?connector ?WKT
    """
    result = conn.select(get_centerline_query)
    elements = result['results']['bindings']
    res = []

    for element in elements:
        wkt_point = element['WKT']['value']
        point = loads(wkt_point)
        
        # Check if the point is within the small polygon
        if point.within(small_poly_shape):
            # Create a circle centered at the point with a 2-meter radius
            circle = point.buffer(0.00001)  # 2-meter radius
            
            # Add the circle to the result
            res.append(dumps(circle))

    return res

    
def check_country(longitude, latitude):
    """
    Determines the country based on the given longitude and latitude by checking against predefined polygonal boundaries.

    Parameters:
    - longitude (float): The longitude of the point to check.
    - latitude (float): The latitude of the point to check.

    Returns:
    - str: The country code ('US' for the United States, 'Singapore' for Singapore) if the point is within the predefined boundaries, otherwise None.
    """
    BOSTON_WKT = "POLYGON((-71.206730 42.469805, -70.891226 42.469805, -70.805107 42.230687, -71.260655 42.246180))"
    SINGAPORE_WKT = "POLYGON((103.569901 1.485775, 104.108661 1.476395, 104.157141 1.158229, 103.576939 1.105849))"
    boston_polygon = Polygon([point for point in [tuple(map(float, coord.split())) for coord in BOSTON_WKT.split("POLYGON((")[1].split(")")[0].split(",")]])
    singapore_polygon = Polygon([point for point in [tuple(map(float, coord.split())) for coord in SINGAPORE_WKT.split("POLYGON((")[1].split(")")[0].split(",")]])

    point = Point(longitude, latitude)
    if boston_polygon.contains(point):
        return "US"
    elif singapore_polygon.contains(point):
        return "Singapore"
    else:
        print("None country error!")
        return None

def get_first_scenes(conn):
    """
    Retrieves the first scenes of sequences in the dataset, which are those without any previous scenes linked to them.

    Parameters:
    - conn (stardog.Connection): Connection to the Stardog database.

    Returns:
    - list: A list of identifiers for the first scenes in the dataset.
    """
    get_first_scenes_query = f"""
        SELECT DISTINCT ?s
        WHERE {{
           ?s a nus:Scene.
           ?s nus:hasSceneParticipant ?sp
           FILTER NOT EXISTS {{
              ?s nus:hasPreviousScene ?s1.
           }}
        }}
        
        #LIMIT 30
    """
    result = conn.select(get_first_scenes_query)
    #pprint.pprint(result)
    first_scenes = result['results']['bindings']
    #pprint.pprint(first_scenes)
    #conn.end
    first_scene_IDs = [item['s']['value'].split('http://www.nuscenes.org/nuScenes/Scene_')[1] for item in first_scenes]
    return first_scene_IDs
  
def consecutive_scenes_area_embeddings_to_json(scene_IDs, conn, area_size, matrix_width, front_matrix_height, back_matrix_height, json_file_name):
    """
    Generates and saves JSON data representing area embeddings for a series of consecutive scenes, incorporating various scene-related parameters and spatial configurations determined by the area size and matrix dimensions.

    Parameters:
    - scene_IDs (list): A list of scene IDs representing consecutive scenes.
    - conn (stardog.Connection): Connection to the Stardog database.
    - area_size (float): The size of one single square area, defining the spatial resolution of the embeddings.
    - matrix_width (int): The total width of the matrix in the number of areas, with the ego vehicle centered. This value should be odd.
    - front_matrix_height (int): The number of areas ahead of the ego vehicle, determining the forward extent of the generated embeddings.
    - back_matrix_height (int): The number of areas behind the ego vehicle, determining the backward extent of the generated embeddings.

    This function processes the data for each scene, computes various parameters like element positions, orientation differences, distances, and country, and then saves this data to a JSON file. The method ensures continuity across the scenes in terms of their spatial and orientation data, handling the sequential logic efficiently.

    Returns:
    - None: The function saves the output directly to a file and does not return any value.
    """

    element_positions_list = []
    scene_ID_list = []
    next_scene_ID_list = []
    yaw_list = []
    distance_list = []
    country_list = []
    previous_yaw=0
    
    
    for i, scene_ID in enumerate(scene_IDs):
        ego_coords = get_egos_coordinates(scene_ID, conn)
        if ego_coords:
            source_point, dest_point, next_scene_ID, sequence_ID = ego_coords_to_geometry(ego_coords)
        else:
            end_scene = verify_end_scene(scene_ID, conn)
            source_point, dest_point, _, sequence_ID = ego_coords_to_geometry(end_scene)
            next_scene_ID = ''
        distance = compute_distance(source_point, dest_point)
        if i == 0 and distance < 0.5:
            temp_next_scene_ID = next_scene_ID
            temp_distance = distance
            while temp_distance < 0.5:
                temp_ego_coords = get_egos_coordinates(temp_next_scene_ID, conn)
                temp_source_point, _, temp_next_scene_ID, _ = ego_coords_to_geometry(temp_ego_coords)
                temp_distance = compute_distance(source_point, temp_source_point)
                if temp_next_scene_ID == '':
                    return
            _, _, _, previous_yaw, _ = regex_query_process(source_point, temp_source_point, previous_yaw, temp_distance)

        
        

        scene_ID_list.append(scene_ID)
        next_scene_ID_list.append(next_scene_ID)
        distance_list.append(distance)
        source_lon, source_lat, yaw, previous_yaw, polygon = regex_query_process(source_point, dest_point, previous_yaw, distance)
        country = check_country(source_lon, source_lat)
        country_list.append(country)
        if distance >= 0.5:
            yaw_list.append(yaw)
        else:
            yaw_list.append(previous_yaw)
        map_elements = get_elements_in_polygon(scene_ID, polygon, conn)
        elements = get_all_dynamic_elements(scene_ID, conn)
        coords, types, _ = process_elements(elements)
        coords_map, types_map, coords_ids_map = process_elements(map_elements, map_elements=True)
        coords = coords+coords_map
        coords.append(source_point)
        types = types+types_map
        types.append('EgoCar')
        subpolys = subpolygons(source_lon, source_lat, yaw, area_size, matrix_width, front_matrix_height, back_matrix_height)
        small_poly = compute_target_polygon(source_lon, source_lat, yaw, area_size*front_matrix_height, area_size*matrix_width, area_size*back_matrix_height)
        small_poly = " ".join(map(str,small_poly))
        small_poly = "POLYGON ((" + small_poly + "))"
        large_poly = compute_target_polygon(source_lon, source_lat, yaw, 2*area_size*front_matrix_height, 2*area_size*matrix_width, 3*area_size*back_matrix_height)
        large_poly = " ".join(map(str,large_poly))
        large_poly = "POLYGON ((" + large_poly + "))"
        lanes = [coords_map[i] for i, element in enumerate(types_map) if element == 'Lane']
        lanes_ids = [coords_ids_map[i].split('http://www.nuscenes.org/nuScenes/map/')[1] for i, element in enumerate(types_map) if element == 'Lane']
        lanes_within_poly = []
        large_poly_shape = shapely.wkt.loads(large_poly)
        lanes_ids_within_poly = []
        for i, lane in enumerate(lanes):
            lane_shape = shapely.wkt.loads(lane)
            if lane_shape.intersects(large_poly_shape):
                lanes_within_poly.append(lane)
                lanes_ids_within_poly.append(lanes_ids[i])
        intersection_points = []
        for lane_id in lanes_ids_within_poly:
            point = compute_centerline(lane_id, conn, small_poly)
            if point:
                intersection_points.append(point)
        intersection_points = [element for sublist in intersection_points for element in sublist]
        element_positions, in_coords = verify_element_positions(coords, types, subpolys, 11, 15, 5, intersection_points)
        element_positions_list.append(element_positions)
    
    
    # print(scene_IDs[0])
    # print(distance_list)
    # print(len(distance_list))
    # print(yaw_list)
    area_embeddings_to_json(element_positions_list, scene_ID_list, next_scene_ID_list, yaw_list, distance_list, country_list, sequence_ID, json_file_name)
    return

import argparse
import yaml
from tqdm import tqdm
import warnings
from urllib3.exceptions import InsecureRequestWarning

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description='Process the config file.')
parser.add_argument('config_path', type=str, help='Path to the config.yaml file')
parser.add_argument('credentials_path', type=str, help='Path to the credentials.yaml file')
args = parser.parse_args()

# Function to read config values
def read_config(config_path, credentials_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    with open(credentials_path, 'r') as file:
        credentials = yaml.safe_load(file)
    return {**config, **credentials}  # Merge the configurations

# Load the configuration from the provided paths
config = read_config(args.config_path, args.credentials_path)

# Ignore the InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


if __name__ == "__main__":
    
    # Setup a new session
    session = requests.Session()
    # Add proxy details
    # Disable SSL verification
    session.verify = False
    conn_details = {
        'endpoint': config['endpoint'],
        'username': config['username'],
        'password': config['password'],
        'database': config['database'],
        'session': session
    }

    conn = stardog.Connection(**conn_details)
    
    first_scene_IDs = get_first_scenes(conn)
    for first_scene_ID in tqdm(first_scene_IDs, desc="Processing scenes", dynamic_ncols=True):
        scene_IDs, ego_points = get_consecutive_scenes(first_scene_ID, conn)
        parked_ego = False
        for i in range(0, len(ego_points), 5):
            if i + 4 < len(ego_points):
                parked_ego = (compute_distance(ego_points[i], ego_points[i + 4]) <= 1)
                if not parked_ego:
                    break
        if not parked_ego:
            area_config = {k: config[k] for k in ['area_size', 'matrix_width', 'front_matrix_height', 'back_matrix_height', 'json_file_name']}
            consecutive_scenes_area_embeddings_to_json(scene_IDs, conn, **area_config)