from rich.console import Console, RenderGroup
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
import numpy as np

class TrainerConsole:
    """
    Class to output information to console.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def update(self, current_model=None):
        """Make a new table."""
        if current_model is None:
            return Text("Training ...", justify="center")
        
        table = Table(expand=True)
        table.add_column("Classifier", ratio=25)
        table.add_column("Score", ratio=10, justify="right",  no_wrap=True)
        table.add_column("Params", ratio=25,  no_wrap=True)
        table.add_column("Model ID",ratio=40,  no_wrap=True)

        for name, stuff in self.trainer.performance.items():
            score, params, hash_id = stuff
            style = "bold green" if name == self.trainer.best_classifier__name else ""
            best_one = " ***" if name == self.trainer.best_classifier__name else ""
    
            table.add_row(
                str(name),
                str(np.round(score, 2)), 
                str(params), 
                f"{str(hash_id)}{best_one}",
                style=style)

        last_ = Text("*** = Best Model  ", style="bold green", justify="right") \
        if current_model == list(self.trainer.grid.keys())[-1] \
        else Text("Training ...", justify="center")

        return RenderGroup( table, last_ )