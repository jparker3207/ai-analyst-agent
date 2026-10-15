import math
import pandas as pd


def _clean(value):
    """Recursively replaces NaN with None so results can always be safely sent as JSON.
    Standard JSON has no way to represent NaN -- pandas uses NaN for missing data, so
    without this, any row with a missing value (like a blank 'note') would crash the
    request to the Gemini API with an invalid-JSON error."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def load_data(filepath):
    """Loads a tabular data file (CSV or Excel) into a pandas DataFrame.

    Args:
        filepath: Path to the file. Supports .csv, .xlsx, and .xls files.

    Returns:
        A pandas DataFrame with the file's contents.
    """
    try:
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
    except Exception as e:
        raise ValueError(f"Could not load file '{filepath}': {e}")


def list_columns(df):
    """Lists the column names available in the dataset, along with their data types.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A dictionary mapping each column name to its data type as a string.
    """
    return _clean({col: str(dtype) for col, dtype in df.dtypes.items()})


def describe_column(df, column_name: str):
    """Returns summary statistics for a single column.

    For numeric columns, returns count, mean, min, max, and standard deviation.
    For non-numeric columns, returns count, number of unique values, and the most common value.

    Args:
        df: The DataFrame to analyze.
        column_name: The name of the column to describe.

    Returns:
        A dictionary of summary statistics, or an error dictionary if the column doesn't exist.
    """
    if column_name not in df.columns:
        return {"error": f"Column '{column_name}' not found. Available columns: {list(df.columns)}"}

    column = df[column_name]

    if pd.api.types.is_numeric_dtype(column):
        result = {
            "count": int(column.count()),
            "mean": float(column.mean()),
            "min": float(column.min()),
            "max": float(column.max()),
            "std": float(column.std()),
        }
    else:
        result = {
            "count": int(column.count()),
            "unique_values": int(column.nunique()),
            "most_common": column.mode().iloc[0] if not column.mode().empty else None,
        }
    return _clean(result)


def group_and_aggregate(df, group_by_column: str, value_column: str, aggregation: str = "mean"):
    """Groups the data by one column and aggregates another column within each group.

    Args:
        df: The DataFrame to analyze.
        group_by_column: The column to group rows by (e.g. "day_of_week").
        value_column: The numeric column to aggregate within each group (e.g. "take_home").
        aggregation: The aggregation to apply: "mean", "sum", "count", "min", or "max". Defaults to "mean".

    Returns:
        A dictionary mapping each group to its aggregated value, or an error dictionary if a column is missing.
    """
    if group_by_column not in df.columns:
        return {"error": f"Column '{group_by_column}' not found. Available columns: {list(df.columns)}"}
    if value_column not in df.columns:
        return {"error": f"Column '{value_column}' not found. Available columns: {list(df.columns)}"}
    if aggregation not in ["mean", "sum", "count", "min", "max"]:
        return {"error": f"Unsupported aggregation '{aggregation}'. Use one of: mean, sum, count, min, max."}

    try:
        result = df.groupby(group_by_column)[value_column].agg(aggregation).sort_values(ascending=False)
        return _clean(result.to_dict())
    except Exception as e:
        return {"error": f"Could not aggregate: {e}"}


def top_n(df, column_name: str, n: int = 5, ascending: bool = False):
    """Returns the top (or bottom) n rows of the dataset, ranked by a given column.

    Args:
        df: The DataFrame to analyze.
        column_name: The column to rank rows by.
        n: How many rows to return. Defaults to 5.
        ascending: If True, returns the smallest values instead of the largest. Defaults to False.

    Returns:
        A list of row dictionaries, or an error dictionary if the column doesn't exist.
    """
    if column_name not in df.columns:
        return {"error": f"Column '{column_name}' not found. Available columns: {list(df.columns)}"}

    try:
        if ascending:
            result = df.nsmallest(n, column_name)
        else:
            result = df.nlargest(n, column_name)
        return _clean(result.to_dict(orient="records"))
    except Exception as e:
        return {"error": f"Could not rank rows: {e}"}


if __name__ == "__main__":
    df = load_data("serverlife_shifts.csv")

    print("--- Columns ---")
    print(list_columns(df))

    print("\n--- Describe 'take_home' ---")
    print(describe_column(df, "take_home"))

    print("\n--- Group by day_of_week, average take_home_per_hour ---")
    print(group_and_aggregate(df, "day_of_week", "take_home_per_hour"))

    print("\n--- Top 3 shifts by take_home (includes rows with missing notes) ---")
    print(top_n(df, "take_home", n=3))

    print("\n--- Error handling test: nonexistent column ---")
    print(describe_column(df, "this_column_does_not_exist"))
