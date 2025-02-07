# Things to improve or fix

## Libraries 
there are missing libraries declared in `pyproject.toml`

to add them
```bash
poetry add ipykernel -dev
poetry add loguru
```

## Absolute paths
in a `example_operations.ipynb`
```python
dataset_dir = "/Users/kemalinecik/git_nosync/sctram/__temp__/data"
```
switch to `pathlib` library for modularity

example
```python
from pathlib import Path

user_path = Path().home().expanduser()
dataset_dir = user_path / "git_nosync" / "sctram" / "__temp__" / "data"
```
and it is platform independent.

## Too many formatters
the project has 
- black
- isort
- flake8
- prettier

`ruff` can do it all and more at a much faster rate.

## Typehints

Optional and Union are outdated.
```python
embedding: Optional[Union[np.ndarray, DataFrame]] = None,
```
Also, Union is outdated.
Use
```python
from __future__ import annotations
...
embedding: np.ndarray | DataFrame | None = None,
```
latter is also suggested by a PEP and it is more concise.


