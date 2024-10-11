#!/usr/bin/env python3

from collections.abc import Iterable
from typing import Dict, List, Union

# Custom Typings
InputGraphDictWithoutEdge = Dict[str, Iterable[tuple[str, str]]]
InputGraphDictNoneEdge = Dict[str, Iterable[tuple[str, str, None]]]
InputGraphDictWithEdge = Dict[str, List[tuple[str, str, dict]]]
InputGraphPossibleTypes = Union[InputGraphDictWithoutEdge, InputGraphDictNoneEdge, InputGraphDictWithEdge]

# Constants
key_trajectories = "trajectories"
edge_reserved_keys = [key_trajectories]
node_reserved_keys = [key_trajectories]

# AnnData Keys
labels_key = "labels"
x_diffmap_key = "X_diffmap"
iroot_key = "iroot"
