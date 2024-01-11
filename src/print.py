from rich import inspect
from rich.markdown import Markdown
from rich.console import Console

def print(*args, **kwargs):
    console = Console(width=85)
    console.print(*args, **kwargs)
