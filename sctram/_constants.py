#!/usr/bin/env python3

from collections.abc import Iterable
from typing import Dict, List, Union

# Custom Typings

InputGraphDictWithoutEdge = Dict[str, Iterable[tuple[str, str]]]
# Represents a graph where each key maps to an iterable of edge tuples without attributes.

InputGraphDictNoneEdge = Dict[str, Iterable[tuple[str, str, None]]]
# Represents a graph where each key maps to an iterable of edge tuples with a `None` attribute placeholder.

InputGraphDictWithEdge = Dict[str, List[tuple[str, str, dict]]]
# Represents a graph where each key maps to a list of edge tuples, each containing start node,
# end node, and a dictionary of edge attributes.

InputGraphPossibleTypes = Union[InputGraphDictWithoutEdge, InputGraphDictNoneEdge, InputGraphDictWithEdge]
# Defines the union of all possible input graph data types accepted by the graph processing functions.

# Constants

key_trajectories = "trajectories"
# The key used to identify and store trajectory information within node and edge attributes.

edge_reserved_keys = [key_trajectories]
# A list of reserved keys that are not permitted to be used as edge attribute keys to prevent
# conflicts with predefined graph properties.

node_reserved_keys = [key_trajectories]
# A list of reserved keys that are not permitted to be used as node attribute keys to prevent
# conflicts with predefined graph properties.

# AnnData Keys

labels_key = "labels"
# Key used to store label information in AnnData structures.

connectivities_key = "connectivities"
# Key used to store connectivities information in AnnData structures after neighbor calculation by scanpy.

distances_key = "distances"
# Key used to store distance information in AnnData structures after neighbor calculation by scanpy

x_diffmap_key = "X_diffmap"
# Key used to store diffusion map coordinates in AnnData structures.

iroot_key = "iroot"
# Key used to store the root identifier in AnnData structures.

neighbors_key = "neighbors"

x_pca_key = "X_pca"

x_umap_key = "X_umap"
