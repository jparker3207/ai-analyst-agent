import pandas as pd

def average_by_day_of_week(df):
    """Returns the average take-home pay per hour, grouped by day of week."""
    result = df.groupby("day_of_week")["take_home_per_hour"].mean().sort_values(ascending=False)
    return result.to_dict()

def best_and_worst_shifts(df, n=3):
    """Returns the n highest-earning and n lowest-earning shifts by total take-home pay."""
    best = df.nlargest(n, "take_home")[["date", "day_of_week", "take_home"]].to_dict(orient="records")
    worst = df.nsmallest(n, "take_home")[["date", "day_of_week", "take_home"]].to_dict(orient="records")
    return {"best": best, "worst": worst}

def overall_summary(df):
    """Returns high-level stats: total shifts, total earned, average per shift, average per hour."""
    return {
        "total_shifts": len(df),
        "total_take_home": float(df["take_home"].sum()),
        "avg_take_home_per_shift": float(df["take_home"].mean()),
        "avg_take_home_per_hour": float(df["take_home_per_hour"].mean()),
    }


if __name__ == "__main__":
    df = pd.read_csv("serverlife_shifts.csv")

    print("--- Average by day of week ---")
    print(average_by_day_of_week(df))

    print("\n--- Best and worst shifts ---")
    print(best_and_worst_shifts(df))

    print("\n--- Overall summary ---")
    print(overall_summary(df))
    