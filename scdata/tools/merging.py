import pandas as pd
import panel as pn
import hvplot.pandas
import holoviews as hv

pn.extension()
hv.extension("bokeh")

hv.opts.defaults(
    hv.opts.Curve(width=900, height=300, line_width=2),
    hv.opts.Scatter(size=6),
    hv.opts.Overlay(legend_position="top_left")
)

def dataframe_row_diff(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    cutoff=None,
    ignore_columns=None
):
    """
    Returns:
    - rows_only_in_df1
    - rows_only_in_df2

    Params:
    - cutoff: only compare rows >= cutoff
    - ignore_columns: list of columns to ignore
    """

    df1 = df1.copy()
    df2 = df2.copy()

    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff)

        if df1.index.tz is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(df1.index.tz)
        elif df1.index.tz is None and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(None)

        df1 = df1[df1.index >= cutoff]
        df2 = df2[df2.index >= cutoff]

    if ignore_columns:
        df1 = df1.drop(columns=[c for c in ignore_columns if c in df1.columns], errors="ignore")
        df2 = df2.drop(columns=[c for c in ignore_columns if c in df2.columns], errors="ignore")

    df1 = df1.sort_index()
    df2 = df2.sort_index()

    only_in_df1_idx = df1.index.difference(df2.index)
    only_in_df2_idx = df2.index.difference(df1.index)

    rows_only_in_df1 = df1.loc[only_in_df1_idx]
    rows_only_in_df2 = df2.loc[only_in_df2_idx]

    return rows_only_in_df1, rows_only_in_df2

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    report = {}

    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    report["extra_columns_in_df1"] = sorted(cols1 - cols2)
    report["extra_columns_in_df2"] = sorted(cols2 - cols1)
    report["common_columns"] = sorted(cols1 & cols2)

    idx1 = set(df1.index)
    idx2 = set(df2.index)

    report["extra_rows_in_df1"] = len(idx1 - idx2)
    report["extra_rows_in_df2"] = len(idx2 - idx1)

    report["extra_index_values_df1"] = sorted(idx1 - idx2)[:10]  # sample
    report["extra_index_values_df2"] = sorted(idx2 - idx1)[:10]

    common_idx = df1.index.intersection(df2.index)
    common_cols = list(cols1 & cols2)

    df1_common = df1.loc[common_idx, common_cols].sort_index()
    df2_common = df2.loc[common_idx, common_cols].sort_index()

    diff_mask = (df1_common != df2_common) & ~(df1_common.isna() & df2_common.isna())

    report["num_value_differences"] = int(diff_mask.sum().sum())

    report["value_differences_per_column"] = diff_mask.sum().to_dict()

    diffs = diff_mask.stack()
    diffs = diffs[diffs]

    if not diffs.empty:
        sample_idx = diffs.index[:10]
        report["sample_differences"] = [
            {
                "index": idx,
                "column": col,
                "df1": df1_common.loc[idx, col],
                "df2": df2_common.loc[idx, col],
            }
            for idx, col in sample_idx
        ]
    else:
        report["sample_differences"] = []

    report["are_equal"] = (
        not report["extra_columns_in_df1"]
        and not report["extra_columns_in_df2"]
        and report["extra_rows_in_df1"] == 0
        and report["extra_rows_in_df2"] == 0
        and report["num_value_differences"] == 0
    )

    return report

class MergeEngine:

    def __init__(self, df_base, df_new):
        self.df_base, self.df_new = df_base.align(df_new, join="outer")

    def merge_column(self, col, priority="new"):
        if priority == "new":
            return self.df_new[col].combine_first(self.df_base[col])
        return self.df_base[col].combine_first(self.df_new[col])

    def get_conflicts(self, col):
        b = self.df_base[col]
        n = self.df_new[col]
        return b.notna() & n.notna() & (b != n)

    def get_presence_mismatch(self, col):
        b = self.df_base[col]
        n = self.df_new[col]

        only_base = b.notna() & n.isna()
        only_new = n.notna() & b.isna()

        return only_base, only_new

class MergePlotBuilder:

    def __init__(self, engine: MergeEngine):
        self.engine = engine

    def build_plot(self, col, priority, show_conflicts=True, show_presence=True):

        base = self.engine.df_base[col]
        new = self.engine.df_new[col]
        merged = self.engine.merge_column(col, priority)

        overlay = (
            base.hvplot(label="base", alpha=1) *
            new.hvplot(label="new", alpha=0.4, line_width=1) *
            merged.hvplot(label="merged", line_width=1)
        )

        if show_conflicts:
            mask = self.engine.get_conflicts(col)
            if mask.any():
                overlay *= base[mask].hvplot.scatter(label="conflicts")

        if show_presence:
            only_base, only_new = self.engine.get_presence_mismatch(col)

            if only_base.any():
                overlay *= base[only_base].hvplot.scatter(
                    label="only base",
                    color="red",
                    marker="circle",
                    size=60
                )

            if only_new.any():
                overlay *= new[only_new].hvplot.scatter(
                    label="only new",
                    color="purple",
                    marker="triangle",
                    size=60
                )

        return overlay.opts(title=col)

class MergeTool:

    def __init__(self, df_base, df_new):

        self.engine = MergeEngine(df_base, df_new)
        self.plot_builder = MergePlotBuilder(self.engine)

        self.columns = list(self.engine.df_base.columns)

        # widgets
        self.col_select = pn.widgets.MultiChoice(
            name="Columns",
            options=self.columns,
            value=self.columns[:1]
        )

        self.priority = {
            col: pn.widgets.Select(
                name=col,
                options=["base", "new"],
                value="new",
                width=120
            )
            for col in self.columns
        }

        self.show_conflicts = pn.widgets.Checkbox(value=True, name="Show conflicts")
        self.show_presence = pn.widgets.Checkbox(
            value=True,
            name="Show presence mismatch"
        )

        self.merge_button = pn.widgets.Button(
            name="Merge ALL",
            button_type="primary"
        )

        # panes
        self.plot_pane = pn.pane.HoloViews()
        self.preview_pane = pn.pane.DataFrame(height=250)
        self.output_pane = pn.pane.DataFrame(height=300)

        # events
        self.col_select.param.watch(self.update, "value")
        self.show_conflicts.param.watch(self.update, "value")
        self.show_presence.param.watch(self.update, "value")

        for w in self.priority.values():
            w.param.watch(self.update, "value")

        self.merge_button.on_click(self.merge_all)

        self.update()

    def update(self, *events):

        cols = self.col_select.value
        if not cols:
            return

        plots = []
        preview_blocks = []

        for col in cols:
            priority = self.priority[col].value

            # plot (HV object)
            plot = self.plot_builder.build_plot(
                col,
                priority,
                show_conflicts=self.show_conflicts.value,
                show_presence=self.show_presence.value
            )
            plots.append(plot)

            # preview
            base = self.engine.df_base[col]
            new = self.engine.df_new[col]
            merged = self.engine.merge_column(col, priority)

            tmp = pd.DataFrame({
                f"{col}_base": base,
                f"{col}_new": new,
                f"{col}_merged": merged
            })

            preview_blocks.append(tmp)

        self.plot_pane.object = hv.Layout(plots).cols(1)

        preview = pd.concat(preview_blocks, axis=1)
        self.preview_pane.object = preview.dropna(how="all").tail(200)

    def merge_all(self, event=None):

        result = pd.DataFrame(
            index=self.engine.df_base.index.union(self.engine.df_new.index)
        )

        for col in self.columns:
            priority = self.priority[col].value
            result[col] = self.engine.merge_column(col, priority)

        self.result = result
        self.output_pane.object = result.tail(500)

    def view(self):

        controls = pn.Column(
            "## Merge Tool",
            self.col_select,
            "### Priority per column",
            pn.GridBox(*self.priority.values(), ncols=3),
            self.show_conflicts,
            self.merge_button
        )

        return pn.Row(
            controls,
            pn.Column(
                self.plot_pane,
                self.preview_pane,
                "### Result",
                self.output_pane
            )
        )