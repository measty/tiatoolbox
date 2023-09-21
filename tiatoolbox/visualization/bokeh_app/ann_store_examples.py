import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.utils.misc import add_from_dat, store_from_dat

"""set of example code snippets for creating an AnnotationStore from model outputs
in a variety of scenarios, and for manipulating Annotation stores for example to add
additional properties or shift store so that coordinates are relative to
some reference point.
"""


"""Scenario: you have patch level predictions for a model. The top left corner
of each patch, and the patch score are in a .csv file. Patch size is 512.
"""

results_path = Path("path/to/results.csv")
SQ = SQLiteStore()
patch_df = pd.read_csv(results_path)
annotations = []
for i, row in patch_df.iterrows():
    x = row["x"]
    y = row["y"]
    score = row["score"]
    annotations.append(
        Annotation(Polygon.from_bounds(x, y, x + 512, y + 512), {"score": score})
    )
SQ.append_many(annotations)
SQ.dump("path/to/annotations.db")


"""Scenario: you have some contours and associated properties in a geojson
or hovernet-style .dat format
"""

# example of hovernet-style .dat file data structure
sample_dict = {
    "nuc_id": {
        "box": List,
        "centroid": List,
        "contour": List[List],
        "prob": float,
        "type": int,
        "prop1": float,
        # can add as many additional properties as we want...
    },
    "next id": {},  # other instances
}


# example of a geojson FeatureCollection data structure:
{
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[21741, 49174.09], [21737.84, 49175.12], ...]],
            },
            "properties": {"type": "cell", "Circularity": 0.951},
        },
        {"type": "Feature", "geometry": "etc..."},
    ],
}


# use class methods of SQLiteStore to create a new AnnotationStore
# from the geojson or hovernet-style .dat file
geojson_path = Path("path/to/annotations.geojson")
SQ1 = SQLiteStore.from_geojson(geojson_path)
SQ1.dump("path/to/annotations.db")

dat_path = Path("path/to/annotations.dat")
# use scale factor to rescale annotations to baseline res
# if annotations have been saved at some other resolution
SQ2 = store_from_dat(dat_path, scale_factor=2)
SQ2.dump("path/to/annotations.db")


"""Scenario: You have a collection of raw centroids or detection contours
with corresponding properties/scores.
"""

centroid_list = [[1, 4], [3, 2]]
# if its contours each element is a list of points instead
properties_list = [
    {"score": "some_score", "class": "some_class"},
    {"score": "other _score", "class": "other_class"},
]

annotations = []

for annotation, properties in zip(centroid_list, properties_list):
    props = {"score": properties["score"], "type": properties["class"]}
    annotations.append(
        Annotation(Point(annotation), props)
    )  # use Polygon() instead if its a contour
SQ.append_many(annotations)
SQ.create_index("area", '"area"')  # create index on area for faster querying
SQ.dump("path/to/annotations.db")


"""Scenario: you have a graph defined by nodes and edges,
and associated node properties
"""

# create a dictionary in the following format:
graph_dict = {
    "edge_index": "2 x n_edges array of indices of pairs of connected nodes",
    "coordinates": "n x 2 array of x,y coordinates for each graph node",
    "score": "n x 1 array of scores for each graph node. Nodes will be coloured by this",
    "feats": "n x n_feats array of properties for each graph node",
    "feat_names": "list of names for each feature in feats array",
    # other instances
}
# will be able to colour by feats in the feats array

# save it as a pickle file:
with open("path/to/graph.pkl", "wb") as f:
    pickle.dump(graph_dict, f)


"""Scenario: you have an existing annotation store and want to add/change
properties of annotations. (can also do similarly for geometry)
"""

# lets assume you have calculated a score in some way, that you want to add to
# the annotations in a store
scores = [0.9, 0.5]

SQ = SQLiteStore("path/to/annotations.db")
# use the SQLiteStore.patch_many method to replace the properties dict
# for each annotation.
new_props = {}
for i, (key, annotation) in enumerate(SQ.items()):
    new_props[key] = annotation.properties  # get existing props
    new_props[key]["score"] = scores[i]  # add the new score

SQ.patch_many(
    SQ.keys(), properties_iter=new_props
)  # replace the properties dict for each annotation


"""The interface will only open one annotation store at a time. If you have annotations
belonging to the same slide in different stores that you want to display
at the same time, jsut put them all in the same store as follows:
"""

SQ1 = SQLiteStore("path/to/annotations1.db")
SQ2 = SQLiteStore("path/to/annotations2.db")
anns = list(SQ1.items())
SQ2.append_many(anns)  # SQ2 .db file now contains all annotations from SQ1 too


"""shifting coordinates. Lets say you want to get all the annotations in a tile,
and put them in a new store with the annotations relative to the top left of tile.
"""
top_left = [2048, 1024]  # top left of tile
tile_size = 1024  # tile size
SQ1 = SQLiteStore("path/to/annotations.db")
query_geom = Polygon.from_bounds(
    top_left[0], top_left[1], top_left[0] + tile_size, top_left[1] + tile_size
)
SQ2 = SQLiteStore()
tile_anns = SQ1.query(query_geom)
SQ2.append_many(tile_anns.values(), tile_anns.keys())


def translate_geom(geom):
    return geom.translate(-top_left[0], -top_left[1])


SQ2.transform(translate_geom)  # translate so coordinates relative to top left of tile
SQ2.dump("path/to/tile_annotations.db")
