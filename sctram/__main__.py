#!/usr/bin/env python
"""Command-line interface."""
import click
from rich import traceback


@click.command()
@click.version_option(version="0.0.1", message=click.style("sctram Version: 0.0.1"))
def main() -> None:
    """sctram."""


if __name__ == "__main__":
    traceback.install()
    main(prog_name="sctram")  # pragma: no cover
