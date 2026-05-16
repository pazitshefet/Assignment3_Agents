import json
import pandas as pd
from typing import Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DatasetColumns:
    instruction: str
    response: str
    category: str
    intent: str


class BitextDataset:
    """
    Loads and queries the Bitext Customer Service dataset.

    The original dataset versions may use slightly different column names.
    This class normalizes the most important columns:
    instruction/query, response, category, intent.
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.csv_path}. "
                f"Put the Bitext CSV under data/bitext_customer_service.csv "
                f"or update DATASET_PATH in .env."
            )

        self.df = pd.read_csv(self.csv_path)
        self.columns = self._detect_columns(self.df)
        self.df = self.df.copy()
        self.df["_row_id"] = range(len(self.df))
        self.df["_category_norm"] = self.df[self.columns.category].astype(str).str.strip().str.lower()
        self.df["_intent_norm"] = self.df[self.columns.intent].astype(str).str.strip().str.lower()
        self.df["_instruction_norm"] = self.df[self.columns.instruction].astype(str).str.lower()
        self.df["_response_norm"] = self.df[self.columns.response].astype(str).str.lower()

    def _detect_columns(self, df: pd.DataFrame) -> DatasetColumns:
        """
        Load the Bitext CSV file and prepare normalized helper columns.

        The helper columns make filtering by category, intent, customer query, and
        agent response easier and more consistent.
        """
        normalized = {c.lower().strip(): c for c in df.columns}

        def find(possible_names: list[str]) -> str:
            for name in possible_names:
                if name in normalized:
                    return normalized[name]
            raise ValueError(
                f"Could not find one of columns {possible_names}. "
                f"Available columns: {list(df.columns)}"
            )

        instruction = find(["instruction",
                            "query",
                            "utterance",
                            "customer_query",
                            "text"])
        response = find(["response",
                         "agent_response",
                         "answer"])
        category = find(["category",
                         "label_category"])
        intent = find(["intent",
                       "label_intent"])

        return DatasetColumns(instruction=instruction,
                              response=response,
                              category=category,
                              intent=intent)

    def categories(self) -> list[str]:
        """
        Return all unique categories in the dataset.

        The result is sorted and cleaned so the agent can use it when answering category
        listing questions.
        """
        values = (self.df[self.columns.category].dropna()
                  .astype(str).str.strip().sort_values().unique().tolist())
        return values

    def intents(self, category: str | None = None) -> list[str]:
        """
        Return all unique intents in the dataset.

        If a category is provided, only intents from that category are returned.
        """
        df = self._filter(category=category)
        values = (df[self.columns.intent].dropna()
                  .astype(str).str.strip().sort_values().unique().tolist())
        return values

    def count(self, category: str | None = None,
              intent: str | None = None, text_search: str | None = None) -> int:
        """
        Count rows in the dataset after applying optional filters.

        The count can be filtered by category, intent, and keyword search.
        """
        return len(self._filter(category=category, intent=intent, text_search=text_search))

    def intent_distribution(self, category: str | None = None) -> list[dict[str, Any]]:
        """
        Return the distribution of intents.

        If a category is provided, the distribution is calculated only for rows in that
        category.
        """
        df = self._filter(category=category)
        result = (df[self.columns.intent].value_counts(dropna=False).reset_index())
        result.columns = ["intent", "count"]
        return result.to_dict(orient="records")

    def examples(self, category: str | None = None,
                 intent: str | None = None, text_search: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """
        Return example rows from the dataset.

        The examples can be filtered by category, intent, or keyword search and are
        limited to the requested number of rows.
        """
        df = self._filter(category=category, intent=intent, text_search=text_search)
        rows = df.head(limit)
        return [{
                "row_id": int(row["_row_id"]),
                "category": str(row[self.columns.category]),
                "intent": str(row[self.columns.intent]),
                "customer_query": str(row[self.columns.instruction]),
                "agent_response": str(row[self.columns.response])
            } for _, row in rows.iterrows()]

    def rows_for_summary(self, category: str | None = None,
                         intent: str | None = None, text_search: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        """
        Return representative rows for summarization.

        This method reuses the example selection logic but is intended for open-ended summary questions.
        """
        return self.examples(category=category,
                             intent=intent,
                             text_search=text_search,
                             limit=limit)

    def _filter(self, category: str | None = None,
                intent: str | None = None, text_search: str | None = None) -> pd.DataFrame:
        """
        Filter the dataset using optional category, intent, and keyword conditions.

        This is the shared filtering logic used by the counting, examples, and summary
        methods.
        """
        df = self.df

        if category:
            category_norm = category.strip().lower()
            df = df[df["_category_norm"] == category_norm]
        if intent:
            intent_norm = intent.strip().lower()
            df = df[df["_intent_norm"] == intent_norm]
        if text_search:
            mask = self._text_search_mask(df, text_search)
            df = df[mask]

        return df

    def _text_search_mask(self, df: pd.DataFrame, text_search: str) -> pd.Series:
        """
        Create a boolean mask for simple keyword search.

        The search checks both customer queries and agent responses and supports simple keyword search
        OR expressions such as 'refund OR money back OR reimbursement'.
        """
        raw = text_search.lower().strip()

        if " or " in raw:
            terms = [t.strip() for t in raw.split(" or ") if t.strip()]
            mode = "any"
        else:
            terms = [raw]
            mode = "any"

        combined_text = df["_instruction_norm"] + " " + df["_response_norm"]

        if mode == "any":
            mask = pd.Series(False, index=df.index)
            for term in terms:
                mask = mask | combined_text.str.contains(term, regex=False, na=False)
            return mask

        return combined_text.str.contains(raw, regex=False, na=False)

    @staticmethod
    def to_json(data: Any) -> str:
        """
        Convert Python data structures to readable JSON text.
        """
        return json.dumps(data, ensure_ascii=False, indent=2)
