import pandas as pd
import numpy as np
import panel as pn

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span, Div, CustomJS, Range1d, WheelZoomTool, LinearAxis, PanTool, TextInput, RangeTool
from bokeh.layouts import column
from itertools import cycle

pn.extension()

class Logger:
    def __init__(self):
        self.pane = pn.pane.Markdown("")

    def log(self, msg):
        self.pane.object += f"\n{msg}"

class TimeSeriesPanel:

    def __init__(self, series_dict, max_plots=3, height=400, width=800, debug=False):
        self.series_dict = series_dict
        self.max_plots = max_plots
        self.height = height
        self.width = width
        self.debug = debug
        self.logger = Logger()

        # --- device mapping ---
        self.device_map = {}
        for name in series_dict:
            dev, col = name.split(":", 1)
            self.device_map.setdefault(dev, []).append(name)

        # --- widgets ---
        self.device_select = pn.widgets.MultiChoice(
            name="Devices",
            options=list(self.device_map.keys()),
            value=list(self.device_map.keys())[:]
        )

        self.series_select = pn.widgets.MultiChoice(
            name="Series",
            options=[],
            value=[]
        )

        self.target_plot = pn.widgets.IntInput(
            name="Target subplot",
            value=0,
            start=0,
            end=max_plots-1
        )

        # widget to select right/left axis for selected series
        self.axis_select = pn.widgets.RadioButtonGroup(name="Axis", options=["left","right"], value="left")
        self.series_axis = {name: "left" for name in series_dict.keys()}

        self.add_button = pn.widgets.Button(name="Update subplot", button_type="primary")
        self.new_plot_button = pn.widgets.Button(name="New subplot", button_type="success")
        self.remove_plot_button = pn.widgets.Button(name="Remove subplot", button_type="danger")
        # self.toggle_legend_button = pn.widgets.Button(name="Toggle legend")
        self.legend_visible = True

        # --- plot grouping ---
        self.plot_groups = {0: []}
        self.num_plots = 1

        # --- color map ---
        palette = [
            "#1f77b4","#ff7f0e","#2ca02c",
            "#d62728","#9467bd","#8c564b",
            "#e377c2","#7f7f7f","#bcbd22","#17becf"
        ]
        color_cycle = cycle(palette)
        self.series_colors = {name: next(color_cycle) for name in series_dict.keys()}

        # --- axis labels ---
        self.text_input_left = TextInput(value="Left axis name", title="Left axis label:")
        self.text_input_right = TextInput(value="Right axis name", title="Right axis label:")
        self.axis_labels = {0: [self.text_input_left.value, self.text_input_right.value]}

        # --- data sources ---
        self.sources = {}
        for name, s in series_dict.items():
            ts = s.index.view("int64") // 10**6  # ms
            self.sources[name] = ColumnDataSource(
                data=dict(x=ts, y=s.values)
            )

        # --- tooltip ---
        self.tooltip = Div(width=300, styles={'background-color': 'aliceblue', 'padding': '10px'})

        # --- wiring ---
        self.device_select.param.watch(self._update_series_options, 'value')
        self.axis_select.param.watch(lambda e: self._axis_select(), 'value')
        self.target_plot.param.watch(lambda e: self._axis_select(), 'value')
        self.add_button.on_click(self._add_to_plot)
        self.new_plot_button.on_click(self._add_new_plot)
        self.remove_plot_button.on_click(self._remove_subplot)
        # self.toggle_legend_button.on_click(self._toggle_legend)

        self._update_series_options()

        # --- reactive plot container ---
        self.plot_pane = pn.Column(self._build_plots())

    # --------------------------
    def _update_series_options(self, *_):
        opts = []
        for dev in self.device_select.value:
            opts.extend(self.device_map.get(dev, []))
        self.series_select.options = opts
        self.series_select.value = []

    # --------------------------
    def _axis_select(self, *_):
        idx = self.target_plot.value
        new_series = []
        for s in self.plot_groups[idx]:
            if self.series_axis[s] == self.axis_select.value:
                new_series.append(s)
        self.series_select.value = new_series

    # --------------------------
    def _add_to_plot(self, *_):
        idx = self.target_plot.value
        if idx not in self.plot_groups:
            self.plot_groups[idx] = []
        for s in self.series_select.value:
            if s not in self.plot_groups[idx]:
                self.plot_groups[idx].append(s)
            self.series_axis[s] = self.axis_select.value
        to_remove = []
        for item in self.plot_groups[idx]:
            self.logger.log(item)
            if item not in self.series_select.value:
                if self.series_axis[item] == self.axis_select.value:
                    self.logger.log(f'Removing {item}')
                    to_remove.append(item)
        if to_remove:
            for item in to_remove:
                self.plot_groups[idx].remove(item)
        self.axis_labels[idx] = [self.text_input_left.value, self.text_input_right.value]
        self._refresh_plots()

    # --------------------------
    def _add_new_plot(self, *_):
        if self.num_plots >= self.max_plots:
            return
        self.plot_groups[self.num_plots] = []
        self.axis_labels[self.num_plots] = []
        self.num_plots += 1
        self.target_plot.end = self.num_plots - 1
        self.target_plot.value = self.target_plot.end
        self.series_select.value = []
        self.text_input_left.value = "Left axis name"
        self.text_input_right.value = "Right axis name"
        self._refresh_plots()

    # --------------------------
    def _refresh_plots(self):
        """Refresh the Bokeh plots in the column pane"""
        self.plot_pane.objects = [self._build_plots()]

    def _remove_subplot(self, *_):
        idx = self.target_plot.value
        if idx in self.plot_groups:
            # remove the subplot
            del self.plot_groups[idx]
            del self.axis_labels[idx]

            # shift higher-index subplots down
            new_plot_groups = {}
            new_axis_labels = {}
            for i, k in enumerate(sorted(self.plot_groups.keys())):
                new_plot_groups[i] = self.plot_groups[k]
            for i, k in enumerate(sorted(self.axis_labels.keys())):
                new_axis_labels[i] = self.axis_labels[k]
            self.plot_groups = new_plot_groups
            self.axis_labels = new_axis_labels
            self.num_plots = len(self.plot_groups)
            self.target_plot.end = max(0, self.num_plots - 1)
        if len(self.plot_groups.keys()) == 0:
            self._add_new_plot()
        self._refresh_plots()

    # def _toggle_legend(self, *_):
    #     self.legend_visible = not self.legend_visible
    #     for fig in getattr(self, "_last_figs", []):
    #         fig.legend.visible = self.legend_visible
    #     self._refresh_plots()

    # --------------------------
    def _build_plots(self):
        figs = []
        spans = []
        shared_x_range = None
        all_series = []

        for i in range(self.num_plots):
            series_list = self.plot_groups.get(i, [])
            axis_name = self.axis_labels.get(i, [])
            if not series_list:
                continue

            if shared_x_range is None:
                p = figure(
                    height=self.height,
                    width=self.width,
                    x_axis_type="datetime",
                    tools="xpan,box_zoom,reset,save",
                    active_scroll=None
                )
                shared_x_range = p.x_range
            else:
                p = figure(
                    height=self.height,
                    width=self.width,
                    x_axis_type="datetime",
                    x_range=shared_x_range,
                    tools="xpan,box_zoom,reset,save",
                    active_scroll=None
                )
            p.yaxis[0].axis_label = axis_name[0]

            # Create separate y-ranges for right axis series if needed
            right_series = [s for s in series_list if self.series_axis[s] == "right"]
            if right_series:
                min_y = []
                max_y = []
                for s in right_series:
                    y = self.sources[s].data['y']
                    y = y[~np.isnan(y)]
                    min_y.append(y.min())
                    max_y.append(y.max())
                p.extra_y_ranges = {"right": Range1d(start=min(min_y),
                                                     end=max(max_y))}
                p.add_layout(LinearAxis(y_range_name="right"), "right")
                p.yaxis[1].axis_label = axis_name[1]

            # plot series
            for name in series_list:
                src = self.sources[name]
                axis_name = self.series_axis[name]
                color = self.series_colors[name]
                if axis_name == 'left':
                    p.line(
                        "x", "y",
                        source=src,
                        line_width=1,
                        color=color,
                        legend_label=name
                    )
                elif axis_name == "right":
                    p.line(
                        "x", "y",
                        source=src,
                        line_width=1,
                        color=color,
                        legend_label=name,
                        y_range_name=axis_name
                    )
                all_series.append(name)

            # Add independent vertical zoom
            wheel_zoom = WheelZoomTool()
            wheel_zoom.zoom_together = "none"

            # Add pan tool
            pan_tool = PanTool()

            p.add_tools(wheel_zoom, pan_tool)
            p.toolbar.active_scroll = wheel_zoom

            # Vline
            vline = Span(location=0, dimension="height", line_color="red", line_width=1)
            p.add_layout(vline)
            spans.append(vline)

            # RangeTool
            select = figure(title="Range select",
                            height=120, width=self.width,
                            x_axis_type="datetime", y_axis_type=None,
                            tools="", toolbar_location=None, background_fill_color="#efefef")
            select.x_range.range_padding = 0
            select.x_range.bounds = "auto"
            range_tool = RangeTool(x_range=p.x_range, start_gesture="pan")
            range_tool.overlay.fill_color = "navy"
            range_tool.overlay.fill_alpha = 0.2

            select.line('x', 'y', source=src)
            select.ygrid.grid_line_color = None
            select.add_tools(range_tool)

            # Legend
            p.legend.visible = self.legend_visible
            p.legend.click_policy="hide"
            p.add_layout(p.legend[0], 'right')
            figs.append(p)
            figs.append(select)

        self._last_figs = figs  # store for toggle legend

        data_sources = {name: self.sources[name] for name in set(all_series)}

        if figs:
            callback = CustomJS(
                args=dict(spans=spans, tooltip=self.tooltip, sources=data_sources),
                code="""
                    const x = cb_obj.x;
                    for (let i=0; i<spans.length; i++) { spans[i].location = x; }
                    let text = `<b>${new Date(x).toISOString()}</b><br/>`;
                    function findClosest(xs, x) {
                        let lo = 0, hi = xs.length - 1;
                        while (hi - lo > 1) {
                            let mid = Math.floor((lo + hi)/2);
                            if (xs[mid] < x) lo = mid; else hi = mid;
                        }
                        return (Math.abs(xs[lo] - x) < Math.abs(xs[hi] - x)) ? lo : hi;
                    }
                    for (const key in sources) {
                        const data = sources[key].data;
                        const xs = data.x; const ys = data.y;
                        if (xs.length===0) continue;
                        const idx = findClosest(xs, x);
                        const val = ys[idx];
                        text += `${key}: ${val.toFixed(3)}<br/>`;
                    }
                    tooltip.text = text;
                """
            )
            for fig in figs:
                fig.js_on_event("mousemove", callback)

        return pn.pane.Bokeh(column(*figs, sizing_mode="stretch_width"), sizing_mode="stretch_width")

    # --------------------------
    def view(self):
        controls = pn.Column(
            "### Add series",
            self.device_select,
            self.series_select,
            self.target_plot,
            pn.Row(self.axis_select, self.add_button),
            # self.toggle_legend_button,
            self.text_input_left,
            self.text_input_right,
            "### Edit subplots",
            pn.Row(self.new_plot_button, self.remove_plot_button),
            '### Tooltip',
            self.tooltip,
            width=320
        )
        if self.debug:
            return pn.Row(
                controls,
                pn.Column(self.plot_pane, pn.Row(self.logger.pane)),
                sizing_mode="stretch_both"
            )
        else:
            return pn.Row(
                controls,
                pn.Column(self.plot_pane),
                sizing_mode="stretch_both"
            )