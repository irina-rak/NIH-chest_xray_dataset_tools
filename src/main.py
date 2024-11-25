import typer
from trogon import Trogon
from typer.main import get_group

import src.commands.app_preprocessing as preprocessing

app = typer.Typer(
    name="nih_processor_app",
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)
app.add_typer(preprocessing.app, name="preprocessing")


@app.command()
def tui(ctx: typer.Context):
    Trogon(get_group(app), click_context=ctx).run()


@app.callback()
def explain():
    """
    The app is made of the following commands:

    * preprocessing: to launch the preprocessing pipeline to clean and prepare the data for the training.

    ---

    To get help on a specific command, use the command name followed by --help.

    Example: `pybiscus preprocessing --help`
    """


if __name__ == "__main__":
    app()
