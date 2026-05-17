import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

class MemoryFactory:
    """
    Creates the persistent LangGraph checkpointer.

    The checkpointer saves graph state into a SQLite database so conversations
    can continue across turns and across program restarts.
    """
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.connection: sqlite3.Connection | None = None

    def create_checkpointer(self) -> SqliteSaver:
        """
        Create a SQLite checkpointer for LangGraph.

        The SQLite connection is kept open on the factory object because the
        checkpointer needs an active database connection while the graph runs.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path),
                                          check_same_thread=False)
        return SqliteSaver(self.connection)