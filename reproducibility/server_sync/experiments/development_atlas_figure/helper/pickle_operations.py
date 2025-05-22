import pickle
from pathlib import Path
from typing import Any, Tuple

def save_pickle_bundle(file_path: str | Path, objects: Tuple[Any, ...]) -> None:
    """
    Save an ordered bundle of objects to a single pickle file.

    Parameters
    ----------
    file_path : str | Path
        Absolute path **including the filename**, e.g. '/abs/path/hdca_bundle.pkl'.
    objects : tuple
        The objects to be pickled (order-preserving).
    """
    file_path = Path(file_path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)   # ensure directory exists

    with file_path.open("wb") as f:
        pickle.dump(objects, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_bundle(file_path: str | Path) -> Tuple[Any, ...]:
    """
    Load and return the tuple of objects previously saved with `save_pickle_bundle`.
    """
    file_path = Path(file_path).expanduser().resolve()

    with file_path.open("rb") as f:
        return pickle.load(f)
