import matplotlib
import matplotlib.colors as pltcol
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from . import dims
from . import utils


OUTPUT_FOLDER = 'figures'



# Point markers in matplotlib:
# "." 	point
# "," 	pixel
# "o" 	circle
# "v" 	triangle_down
# "^" 	triangle_up
# "<" 	triangle_left
# ">" 	triangle_right
# "1" 	tri_down
# "2" 	tri_up
# "3" 	tri_left
# "4" 	tri_right
# "8" 	octagon
# "s" 	square
# "p" 	pentagon
# "P" 	plus (filled)
# "*" 	star
# "h" 	hexagon1
# "H" 	hexagon2
# "+" 	plus
# "x" 	x
# "X" 	x (filled)
# "D" 	diamond
# "d" 	thin_diamond
# "|" 	vline
# "_" 	hline




# ------------------------------------------------------------------
#                        CREATING FIGURES
# ------------------------------------------------------------------


def set_latex(b=True):
    """Enables/disables LaTeX interpretation in matplotlib."""
    from matplotlib import rc
    if b:
        matplotlib.rcParams['text.latex.unicode'] = True
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        ## for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)
    else:
        rc('text', usetex=False)


def get_averages_line(series, max_len=None):
    """Computes averages on the ith positions of the list. Ignores missing values. Returns both averages and standard deviations."""
    res_avg = []
    res_std = []
    if max_len is None:
        max_len = max([len(s) for s in series])
    for i in range(0, max_len):
        values = [s[i] for s in series if len(s) > i]
        res_avg.append(np.average(values))
        res_std.append(np.std(values))
    return res_avg, res_std



def plot_single_run(prop, keys, legend=None, title=None, ylabel='Value'):
    """Plots a single run by reading data under specified keys from the properties and plotting them against generation number.

    X axis: generation no.
    Y axis: data series for each key.
   
    TODO: this is work in progress.
    """
    if title is None:
        title = "generation no. -> " + str(keys)
    if legend is None:
        legend = {k:k for k in keys}

    assert len(keys) > 0
    assert type(keys) == list
    assert type(legend) == dict

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('Generation')
    ax1.set_title(title)

    for k in keys:
        series = utils.str2list(prop[k])
        assert type(series) == list

        ax1.plot(series, label=legend[k])

    plt.legend(loc='upper left')
    plt.show()



def plot_fitness_single_run(prop, keys, legend = None, title = None):
    """Plots a single run by reading data under specified keys from the properties and plotting them against generation number.

    X axis: generation no.
    Y axis: data series for each key.

    TODO: this is work in progress.
    """
    if title is None:
        title = "generation no. -> " + str(keys)
    if legend is None:
        legend = {k:k for k in keys}

    assert len(keys) > 0
    assert type(keys) == list
    assert type(legend) == dict

    fig = plt.figure()
    plt.subplot()
    plt.margins(0.01)  # Pad margins so that markers don't get clipped by the axes
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Passed tests')
    ax1.set_xlabel('Generation')
    ax1.set_title(title)
    ax1.grid('on')

    num_generations = int(prop["result.lastGeneration"])
    x_axis = range(0,(num_generations+1))

    for k in keys:
        series = utils.str2list(prop[k])
        assert type(series) == list
        if k == "result.stats.avgFitness":
            errors = utils.str2list(prop["result.stats.stdFitness"])
            ax1.errorbar(x_axis, series, yerr=errors)
        ax1.plot(series, label=legend[k])

    lgd = ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    fig.savefig('samplefigure.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()






def plot_value_progression_grid_simple(props, dim_rows, dim_cols, key_values, legend_labels=None, legend_colors=None, title=None,
                                       unify_xaxis=True, unify_xlim=None, plot_individual_runs=False,
                                       key_lastGen=None, markevery=2, errorevery=2, savepath=None, opacity=0.5, show_plot=False,
                                       plot_grid=True, shared_x_axis_scale=False):
    """For each element of the row dimension (usually benchmark) creates a series of subplots for each
    configuration of the column dimension. On subplots plotted are changes of some value through time.
    If plot_individual_runs is set to True, each evolution run will be plotted as a distinct line.

    :param key_values: (list[str]) values used for plotting. They are assumed to be in the format: 'X1,X2,X3,...' where Xi is a number.
    :param key_lastGen: (str) dict's key used to compute the maximum generation so that all x axes are synchronized.
    :param shared_x_axis_scale: (bool) if True, then every subplot will have the same range on the X axis.
    """
    assert type(props) == list
    assert isinstance(dim_rows, dims.Dim)
    assert isinstance(dim_cols, dims.Dim)
    assert isinstance(key_values, list)
    if legend_labels is None:
        legend_labels = ["{0}".format(kv) for kv in key_values]
    assert isinstance(legend_labels, list) and len(legend_labels) == len(key_values)
    if legend_colors is None:
        legend_colors = [(0, 0, 0.5), (0, 0.5, 0), (0.5, 0, 0), (0, 0, 0)]

    nrow = len(dim_rows)
    ncol = len(dim_cols)
    figsize = (int(4.25*ncol), int(2.75*nrow))
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize, facecolor='white')

    # fig.tight_layout()
    if title is not None:
        fig.suptitle(title, fontsize=16)


    # legend_info stores information necessary to create a legend and color the plots.
    legend_info = dict()
    for key, label, color in zip(key_values, legend_labels, legend_colors):
        legend_info[key] = (label, pltcol.ColorConverter().to_rgba(color, opacity))

    # Remove irrelevant props (data, which will not be plotted).
    props = (dim_rows * dim_cols).filter_out_outsiders(props)

    xlim_row = None
    if unify_xaxis:
        if unify_xlim is not None:
            xlim_row = unify_xlim
        else:
            # Compute maximum x-axis value for all props which are relevant.
            if key_lastGen is not None:
                max_generations = max([int(p[key_lastGen]) for p in props])
            else:
                max_generations = max([ max([int(p[k].count(",")) for p in props]) for k in key_values ])
            xlim_row = [0, max_generations]


    i = 0
    for config_row in dim_rows:
        props_row = config_row.filter_props(props)
        maxValueY = None
        minValueY = None
        j = 0
        for config_col in dim_cols:
            axes[i, j].margins(0.01)  # Pad margins so that markers don't get clipped by the axes
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].set_axis_bgcolor('white')
            if plot_grid:
                axes[i, j].grid("on")
            else:
                axes[i, j].grid("off")

            props_rc = config_col.filter_props(props_row)
            # If props_rc is empty, then there is no point in plotting anything
            if len(props_rc) == 0:
                continue

            d_series_to_plot = {k: [utils.str2list(p[k]) for p in props_rc] for k in legend_info}

            if len(d_series_to_plot) == 0:
                print("len(d_series_to_plot) == 0")

            max_y = max([ max([max(s) for s in d_series_to_plot[k] ])  for k in d_series_to_plot ])
            min_y = min([min([min(s) for s in d_series_to_plot[k]]) for k in d_series_to_plot])
            if maxValueY is None or maxValueY < max_y:
                maxValueY = max_y
            if minValueY is None or minValueY < min_y:
                minValueY = min_y

            if plot_individual_runs:
                _progression_grid_subplot_individualRuns(fig, axes, i, j, legend_info, d_series_to_plot,
                                                         row_header=config_row.get_caption(),
                                                         col_header=config_col.get_caption(),
                                                         markevery=markevery)
            else:
                _progression_grid_subplot_avgs(fig, axes, i, j, legend_info, d_series_to_plot,
                                               row_header=config_row.get_caption(),
                                               col_header=config_col.get_caption(),
                                               markevery=markevery, errorevery=errorevery)

            # Update Y axis value range
            ylim_row = [minValueY, maxValueY]
            for jj in range(len(dim_cols)):
                axes[i, jj].set_ylim(ylim_row)
            j += 1


        # Update X axis value range
        if shared_x_axis_scale:
            for ii in range(len(dim_rows)):
                for jj in range(len(dim_cols)):
                    if xlim_row is not None:
                        axes[ii, jj].set_xlim(xlim_row)

        maxValueY = None
        i += 1

    # Set legend
    legend_patches = []
    for k in sorted(legend_info.keys()):
        name, color = legend_info[k]
        patch = mpatches.Patch(color=color, label=name)
        legend_patches.append(patch)
    legend_patches = list(reversed(legend_patches))
    plt.legend(handles=legend_patches, loc='center right', bbox_to_anchor = (0,0,0.975,1), bbox_transform = plt.gcf().transFigure)


    if savepath is not None:
        fig.savefig(savepath, facecolor=fig.get_facecolor(), edgecolor="white")
    if show_plot:
        plt.show()




def _progression_grid_subplot_individualRuns(fig, axes, i, j, legend_info, d_series_to_plot, row_header, col_header, markevery=2):
    """Draws a subplot containing fitness progression through iterations on an intersection
    of a single benchmark and single variant.

    :param d_series_to_plot: (dict[str,list]) A dictionary containing for each key from legend_info a list of series
     of values to plot.
    """
    assert isinstance(d_series_to_plot, dict)
    assert all([k in d_series_to_plot for k in legend_info.keys()])

    if i == 0:
        axes[i,j].set_title(col_header, fontsize=16)
    if j == 0:
        axes[i,j].set_ylabel(row_header, fontsize=16)


    for k in legend_info:
        name, color = legend_info[k]
        # Plot individual series for each
        for series in d_series_to_plot[k]:
            x = range(0, len(series))
            axes[i,j].plot(x, series, color=color, linewidth=1.0, markevery=markevery)



def _progression_grid_subplot_avgs(fig, axes, i, j, legend_info, d_series_to_plot, row_header, col_header, markevery=2, errorevery=2):
    """Draws a subplot containing fitness progression through iterations on an intersection
    of a single benchmark and single variant.

    :param d_series_to_plot: (dict[str,list]) A dictionary containing for each key from legend_info a list of series
     of values to plot.
    """
    assert isinstance(d_series_to_plot, dict)
    assert all([k in d_series_to_plot for k in legend_info.keys()])

    if i == 0:
        axes[i,j].set_title(col_header, fontsize=16)
    if j == 0:
        axes[i,j].set_ylabel(row_header, fontsize=16)

    for k in legend_info:
        name, color = legend_info[k]
        averages, errors = get_averages_line(d_series_to_plot[k])
        x = range(0, len(averages))
        axes[i, j].errorbar(x, averages, color=color, yerr=errors, linewidth=0.5, errorevery=errorevery, markevery=markevery)





def compare_avg_data_series_d(props, dim_series, key_x, key_y, point_markers=None,
                            savepath=None, show_plot=0, **kwargs):
    assert isinstance(key_y, str)
    fun_x = lambda p: p[key_y]
    compare_avg_data_series(props, dim_series, key_x, fun_x, 0, point_markers, savepath,
                              show_plot, **kwargs)



def compare_avg_data_series(props, dim_series, key_x, getter_y, is_aggr=0, point_markers=None,
                            savepath=None, show_plot=0, **kwargs):
    """Creates a simple 2D plot on which data series are compared on a certain value.

    :param props: (list[dict[str, str]]) dictionaries to be processed.
    :param dim_series: (Dim) a dimension used to generate different data series.
    :param key_x: (str) a key from which should be taken values on x axis (treated
    as ordinal variables sorted lexicographically).
    :param getter_y: (lambda) a function from prop to a value. If is_aggr=1, then the
    function is applied to all properties meeting the x point predicate.
    """
    assert isinstance(props, list)
    assert isinstance(dim_series, dims.Dim)
    assert isinstance(key_x, str)
    assert not isinstance(getter_y, str)
    point_markers = ["x", "+", "o", "^", "s", "8"] if point_markers is None else point_markers

    kwargs["title"] =  kwargs.get("ylabel", "")
    kwargs["ylabel"] =  kwargs.get("ylabel", "Y")
    kwargs["xlabel"] =  kwargs.get("xlabel", "X")
    kwargs["legend_loc"] =  kwargs.get("legend_loc", "lower right")

    dim_x = dims.Dim.from_dict(props, key_x).sort()
    x_capts = dim_x.get_captions()
    xs = list(range(len(x_capts)))

    fig = plt.figure() # figsize=(12, 7)
    plt.xticks(xs, x_capts)
    plt.margins(0.01)  # Pad margins so that markers don't get clipped by the axes
    ax1 = fig.add_subplot(111)
    ax1.set_title(kwargs["title"])
    ax1.set_ylabel(kwargs["ylabel"])
    ax1.set_xlabel(kwargs["xlabel"])


    def compute_series(f_props):
        """Produces xs and ys for a single data series. ys here is an average."""
        ys = []
        for x_conf in dim_x:
            x_props = x_conf.filter_props(f_props)
            if is_aggr:
                avg = getter_y(x_props)
            else:
                avg  = np.mean([float(getter_y(p)) for p in x_props])
            ys.append(avg)
        return xs, ys

    series = []
    for s in dim_series:
        f_props = s.filter_props(props)
        if len(f_props) > 0:
            data = compute_series(f_props)
            series.append((s.get_caption(), data))

    for i, s in enumerate(series):
        pm = point_markers[i] if i < len(point_markers) else point_markers[-1]
        name, data = s
        ax1.plot(data[0], data[1], pm+"-", label=name)

    lgd = ax1.legend(loc=kwargs["legend_loc"], bbox_transform=plt.gcf().transFigure)
    if savepath is not None:
        print("Saving figure: " + savepath)
        fig.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    if show_plot:
        plt.show()




def compare_fitness_on_benchmarks(props, dim_benchmarks, dim_variants, key="result.best.eval",
                                  print_quantities=False, use_latex=False, savepath=None, show_plot=1):
    """For each benchmark creates a box-and-wshiskers plot on which boxes are drawn for
    each configuration of the experiment.

    :param props: (list[dict[str, str]]) dictionaries to be processed.
    :param dim_variants: (Dim) a dimension used to generate different data series.
    """
    assert isinstance(dim_benchmarks, dims.Dim)
    assert isinstance(dim_variants, dims.Dim)
    nrow = len(dim_benchmarks)
    fig, axes = plt.subplots(nrows=nrow, ncols=1, figsize=(10, 11))
    fig.tight_layout(h_pad=2.5)
    # fig.set_title("Comparison of Fitness Across Benchmarks")

    i = 0
    for db in dim_benchmarks:
        props_db = db.filter_props(props)
        _compare_fitness_on_benchmarks_subplot(fig, axes, i, db, props_db, dim_variants, key_fit=key, print_quantities=print_quantities, use_latex=use_latex)
        i += 1

    if savepath is not None:
        print("Saving figure: " + savepath)
        fig.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    if show_plot:
        plt.show()





def plot_ratio_meeting_predicate(props, getter, predicate, condition=None, series_dim=None, xs=None,
                                 savepath=None, xlabel=None, xticks=None, title=None, show_plot=1,
                                 xlogscale=False, ylogscale=False):
    """Plots a ratio of solutions meeting the predicate. Ratios are presented on the Y axis. On the X axis
    presented are values of a certain attribute on which progression is made (e.g. time to solve). Those
    values (obtained by getter) are then compared with the dict's value (using the predicate).
    
    As 100% treated is the total number of elements in props.
    
    :param props: (list[dict[str, str]]) Dictionaries to be processed.
    :param getter: (lambda) Function used to obtain from the dict a property plotted on the X axis.
    :param predicate: (lambda) Predicate used to count number of dicts which should be counted to the ratio.
     Predicate takes two parameters: A, B, where A is always value of a property assigned to a dict, and
     B is value of the same property iterated by plotter.
     
     Assume we have d={"atr1":12}, and we are checking atr1 values from 0 to 100 with step 25. In the first
     step A=12, B=0, and predicate (assume simple <=) checks, if A<=B, which translates in this case to
     12 <= 0. Predicate is not met, so dict does not count for this point. Next is B=25, and this time
     dict will be taken into the ratio.
    :param condition: (lambda) Condition on a dictionary, which determines, if it counted 'positively' into
     the ratio. Using condition usually will mean that the ratio of 1.0 will not be met for any point on
     X axis.
    :param series_dim: (Dim) a dimension used to generate series.
    """
    fig = plt.figure(figsize=(12, 7))
    plt.subplot()
    plt.margins(0.01)  # Pad margins so that markers don't get clipped by the axes
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Ratio')
    if xlabel is not None:
        ax1.set_xlabel(xlabel)
    if title is not None:
        ax1.set_title(title)
    ax1.grid('on')
    ax1.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax1.set_ylim((0.0, 1.0))

    getter_values = [getter(p) for p in props]
    if xs is None:
        N = 10  # number of points per plot line
        # compute range based on values present in the dicts
        r = (min(getter_values), max(getter_values))
        xs = np.linspace(r[0], r[1], N)
    if xticks is not None:
        ax1.set_xticks(xticks)
    else:
        ax1.set_xticks(xs)

    def compute_series(getter_values, f_props):
        ys = []
        for x in xs:
            count = 0
            for i, value in enumerate(getter_values):
                if value is None:
                    continue
                if predicate(value, x) and (condition is None or condition(f_props[i])):
                    count += 1
            ys.append(float(count) / len(getter_values))
        return xs, ys


    series = []
    if series_dim is None: # no dimensions
        series.append(("All" + "  (total: {0})".format(len(props)), compute_series(getter_values, props)))
    else:
        for config in series_dim:
            f_props = config.filter_props(props)
            if len(f_props) > 0:
                gv = [getter(p) for p in f_props]
                data = compute_series(gv, f_props)
                series.append((config.get_caption() + "  (total: {0})".format(len(f_props)), data))


    for s in series:
        name, data = s
        if xlogscale:
            ax1.set_xscale("log", nonposx='clip')
        if ylogscale:
            ax1.set_yscale("log", nonposy='clip')
        ax1.plot(data[0], data[1], ".-", label=name)  # label=label

    lgd = ax1.legend(loc='lower right', bbox_transform=plt.gcf().transFigure)
    if savepath is not None:
        print("Saving figure: " + savepath)
        fig.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    if show_plot:
        plt.show()





def _compare_fitness_on_benchmarks_subplot(fig, axes, i, benchmark, props, dim_variants, key_fit,
        print_quantities=False, use_latex=False):
    """Draws a subplot containing box-and-wshiskers plot for all variants on a single benchmark."""
    assert isinstance(benchmark, dims.ConfigList) and len(benchmark) == 1 # only one filter function for benchmark.
    all_data = []
    labels = []
    for config in dim_variants:
        variant_data = [float(p[key_fit]) for p in props if config.filter(p)]
        all_data.append(variant_data)
        label = r"{0}".format(str(config.get_caption()))
        if print_quantities:
            label += r" {0}".format(str(len(variant_data)))
        labels.append(label)

    axes[i].margins(0.01)  # Pad margins so that markers don't get clipped by the axes
    axes[i].boxplot(all_data, labels=labels, showmeans=True)

    bench_caption = benchmark.get_caption()
    bench_optFit = benchmark.filters[0][2]
    if use_latex:
        title = r"${0}$".format(bench_caption) + r" {(optimal fitness=" + str(bench_optFit) + r")"
    else:
        title = r"{0} (optimal fitness={1})".format(bench_caption, str(bench_optFit))
    axes[i].set_title(title)
    ylabel = "Fitness"
    if use_latex:
        ylabel = r"$\textit{" + ylabel + r"}$"
    axes[i].set_ylabel(ylabel)
    axes[i].set_ylim([0, bench_optFit])
    axes[i].yaxis.grid(True)
