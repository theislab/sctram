import pandas as pd
from pandas.api.types import is_string_dtype, is_object_dtype

def harmonize_union_with_report(
    df_new: pd.DataFrame, 
    df_big: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    df_new : pd.DataFrame
        The authoritative (usually smaller) DataFrame.
    df_big : pd.DataFrame
        The larger DataFrame that should already contain all indexes of df_new.

    Returns
    -------
    out_df      : pd.DataFrame
        Union of the two data frames (df_new has priority on shared indexes).
    mismatch_df : pd.DataFrame
        Report of rows / columns where the two inputs disagree.
        • Index is the row label.
        • Columns are '<col>_new' and '<col>_big' for each mismatching column.
        • Empty (shape == (0, 0)) if no mismatches are detected.
    """
    cols = list(df_new.columns)

    missing_idx = df_new.index.difference(df_big.index)
    missing_df = df_new.loc[missing_idx].copy()
    # if not missing_idx.empty:
    #     raise ValueError(
    #         f"df_big is missing {len(missing_idx)} index label(s) "
    #         f"that appear in df_new: {missing_idx.tolist()}"
    #     )
    
    # ------------------------------------------------------------------
    # 1. Build mismatch report for shared rows
    # ------------------------------------------------------------------
    shared_idx   = df_new.index.intersection(df_big.index)
    mismatch_buf = {}

    if not shared_idx.empty:
        common_cols = [c for c in cols if c in df_big.columns]

        for c in common_cols:
            s_new = df_new.loc[shared_idx, c]
            s_big = df_big.loc[shared_idx, c].reindex_like(s_new)

            unequal_mask = ~(s_new.eq(s_big) | (s_new.isna() & s_big.isna()))
            if unequal_mask.any():
                # store only the differing values
                mismatch_buf[f"{c}_new"] = s_new.where(unequal_mask)
                mismatch_buf[f"{c}_big"] = s_big.where(unequal_mask)

    mismatch_df = (
        pd.DataFrame(mismatch_buf)
        .dropna(how="all")
        .astype("category")
        if mismatch_buf
        else pd.DataFrame(index=[], columns=[])
    )

    # ------------------------------------------------------------------
    # 2. Rows that exist only in df_big
    # ------------------------------------------------------------------
    df_big_extra = df_big.loc[~df_big.index.isin(df_new.index)].copy()

    for col in cols:
        if col not in df_big_extra.columns:
            filler = "NAN" if is_string_dtype(df_new[col]) or is_object_dtype(df_new[col]) else pd.NA
            df_big_extra[col] = filler
    df_big_extra = df_big_extra[cols]

    # ------------------------------------------------------------------
    # 3. Concatenate & cast every column to 'category'
    # ------------------------------------------------------------------

    out_df = pd.concat([df_new, df_big_extra], axis=0)
    # keep only shared ones
    if len(missing_df) > 0:
        out_df = out_df[~out_df.index.isin(missing_df.index)]
    
    for col in cols:
        out_df[col] = out_df[col].astype("category")
        out_df[col] = out_df[col].cat.remove_unused_categories()
    
    return out_df, mismatch_df, missing_df

def analyze_column_consistency(dataframes: dict):
    all_column_sets = {k: set(v.columns) for k, v in dataframes.items()}
    
    # Find intersection of all column sets
    intersecting_columns = set.intersection(*all_column_sets.values())

    # Find non-shared (extra or missing) columns for each DataFrame
    differing_columns = {
        k: cols.symmetric_difference(intersecting_columns)
        for k, cols in all_column_sets.items()
        if cols != intersecting_columns
    }

    return intersecting_columns, differing_columns

def combine_dataframes_on_columns(
    dataframes: dict,
    columns_to_keep: list,
    verify_integrity: bool = False,  # raise if the final index has duplicates
):
    """
    Vertically stack DataFrames from `dataframes` while

    * retaining their existing index values,
    * keeping only `columns_to_keep`,
    * adding a new column `csv_key` with the dict key,
    * optionally raising if any index labels collide.

    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        Mapping of name → DataFrame.
    columns_to_keep : list[str]
        Columns that must be present in every DataFrame.
    verify_integrity : bool, default False
        If True, `pd.concat(..., verify_integrity=True)` is used so the call
        fails when duplicate index labels occur.

    Returns
    -------
    pd.DataFrame
    """
    processed = []
    for key, df in dataframes.items():
        missing = set(columns_to_keep) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in '{key}': {missing}")

        df_subset = df[columns_to_keep].copy()  # keeps original index
        df_subset["csv_key"] = key
        processed.append(df_subset)

    return pd.concat(
        processed,
        axis=0,
        verify_integrity=verify_integrity  # keeps (and optionally checks) index
    )
