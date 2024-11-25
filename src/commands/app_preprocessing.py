import os
from pathlib import Path
from typing import Annotated

import typer
from omegaconf import OmegaConf

from src.console import console

app = typer.Typer(pretty_exceptions_show_locals=False, rich_markup_mode="rich")


@app.callback()
def explain():
    """
    The preprocessing command is used to preprocess the data.

    The command can be used with the following subcommands:

    * preprocess: to preprocess the data according to the configuration file.

    ---

    To get help on a specific subcommand, use the subcommand name followed by --help.
    """


@app.command()
def preprocess(config: Annotated[Path, typer.Argument()] = None):
    """Preprocess the data.

    This command preprocesses the data according to the configuration file.

    Parameters
    ----------
    config : Path
        the path to the configuration file.

    Raises
    ------
    typer.Abort
        _description_
    typer.Abort
        _description_
    typer.Abort
        _description_
    """
    if config is None:
        print("No config file")
        raise typer.Abort()
    if config.is_file():
        conf = OmegaConf.load(config)
        console.log(dict(conf))
    elif config.is_dir():
        print("Config is a directory, will use all its config files")
        raise typer.Abort()
    elif not config.exists():
        print("The config doesn't exist")
        raise typer.Abort()

    # Add the preprocessing steps here


if __name__ == "__main__":
    app()
