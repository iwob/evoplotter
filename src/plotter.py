import matplotlib
import matplotlib.colors as pltcol
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from . import dims
from . import utils


OUTPUT_FOLDER = 'figures'



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







def plot_fitness_progression_on_benchmarks(props, dim_benchmarks, dim_variants, key_avgFits="result.stats.avgFitness", key_maxFits=None, plot_max=True, unify_xaxis=True, unify_xlim=None, plot_individual_runs=False, key_lastGen="result.lastGeneration", markevery=2, errorevry=2, savepath=None):
	"""For each benchmark creates a series of subplots for each configuration of the experiment. On subplots plotted are changes of fitness through time. If plot_individual_runs is set to True, each evolution run will be plotted as a distinct line.
	"""
	assert type(props) == list
	assert isinstance(dim_benchmarks, dims.Dim)
	assert isinstance(dim_variants, dims.Dim)
	if plot_max and key_maxFits is None:
		key_maxFits = "result.stats.maxFitness" # Use default keys from smtgp.

	nrow = len(dim_benchmarks)
	ncol = len(dim_variants)
	figsize = (int(3.5*ncol), int(2.75*nrow))
	fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize, facecolor='white')

	# fig.tight_layout()
	fig.suptitle("Fitness Progression Across Benchmarks", fontsize=16)

	gray = pltcol.ColorConverter().to_rgba((0, 0, 0), 0.2)
	green = pltcol.ColorConverter().to_rgba((0, 0.5, 0), 0.2)
	keys_info = {key_avgFits: ("avg", gray)}
	if key_maxFits is not None:
		keys_info[key_maxFits] = ("max", green)

	# Remove irrelevant props (data, which will not be plotted).
	props = (dim_benchmarks * dim_variants).filter_out_outsiders(props)

	xlim_db = None
	if unify_xaxis:
		if unify_xlim is not None:
			xlim_db = unify_xlim
		else:
			# Compute maximum x-axis value for all props which are relevant.
			max_generations = max([int(p[key_lastGen]) for p in props])
			xlim_db = [0, max_generations]

	i = 0
	for db in dim_benchmarks:
		print("Plotting row {0}/{1}: {2}".format(i, len(dim_benchmarks)-1, db.filters[0][0]))
		props_db = db.filter_props(props)
		bench_maxFit = db.filters[0][2]  # Getting max fitness of the benchmark.
		ylim_db = [0, bench_maxFit]
		j = 0
		for dv in dim_variants:
			props_dbdv = dv.filter_props(props_db)
			_plot_fitness_progression_on_benchmarks_subplot(fig, axes, i, j, props_dbdv, db, dv, keys_info, xlim_db, ylim_db, plot_individual_runs=plot_individual_runs, key_lastGen=key_lastGen, key_avgFits=key_avgFits, markevery=markevery, errorevry=errorevry)
			j += 1
		i += 1

	if plot_individual_runs:
		legend_patches = []
		for k in sorted(keys_info.keys()):
			name, color = keys_info[k]
			patch = mpatches.Patch(color=color, label=name)
			legend_patches.append(patch)
		legend_patches = list(reversed(legend_patches))
		plt.legend(handles=legend_patches, loc='center right', bbox_to_anchor = (0,0,0.975,1), bbox_transform = plt.gcf().transFigure)


	if savepath is None:
		name = "fit_progression_runs.pdf" if plot_individual_runs else "fit_progression_avgs.pdf"
		savepath = OUTPUT_FOLDER + "/" + name
	fig.savefig(savepath, facecolor=fig.get_facecolor(), edgecolor="white")
	plt.show()




def _plot_fitness_progression_on_benchmarks_subplot(fig, axes, i, j, props, benchmark, variant, keys_info, xlim_db=None, ylim_db=None, plot_individual_runs=False, key_lastGen="result.lastGeneration", key_avgFits="result.stats.avgFitness", markevery=2, errorevry=2):
	"""Draws a subplot containing fitness progression through iterations on an intersection of a single benchmark and single variant."""
	axes[i,j].margins(0.01)  # Pad margins so that markers don't get clipped by the axes
	axes[i,j].spines['top'].set_visible(False)
	axes[i,j].spines['right'].set_visible(False)
	axes[i,j].set_axis_bgcolor('white')
	axes[i,j].grid("on")

	if i == 0:
		axes[i,j].set_title(variant.get_caption(), fontsize=16)
	if j == 0:
		axes[i,j].set_ylabel(benchmark.filters[0][0], fontsize=16)

	if len(props) > 0:
		if plot_individual_runs:
			for k in keys_info:
				name, color = keys_info[k]
				series_to_plot = [utils.str2list(p[k]) for p in props]
				for series in series_to_plot:
					x = range(0, len(series))
					axes[i,j].plot(x, series, color=color, linewidth=1.0, markevery=markevery)

		else:
			max_generations = max([int(p[key_lastGen]) for p in props])
			if xlim_db is not None and xlim_db[1] > max_generations:
				max_generations = xlim_db[1]

			series_to_plot = [utils.str2list(p[key_avgFits]) for p in props]
			# We must pad fitness with optimal values after the run has ended.
			for series in series_to_plot:
				series.extend([benchmark.head()[2]] * (max_generations - len(series)))

			averages, errors = get_averages_line(series_to_plot)
			x = range(0, len(averages))
			axes[i, j].errorbar(x, averages, color="black", yerr=errors, linewidth=0.5, errorevery=errorevry, markevery=markevery)
			# axes[i, j].plot(x, averages, color="blue", linewidth=1.0)

	if xlim_db is not None:
		axes[i,j].set_xlim(xlim_db)
	if ylim_db is not None:
		axes[i,j].set_ylim(ylim_db)







def compare_fitness_on_benchmarks(props, dim_benchmarks, dim_variants, key_fit="result.best.eval", print_quantities=False, use_latex=False, savepath=None):
	"""For each benchmark creates a box-and-wshiskers plot on which boxes are drawn for each configuration of the experiment."""
	assert isinstance(dim_benchmarks, dims.Dim)
	assert isinstance(dim_variants, dims.Dim)
	nrow = len(dim_benchmarks)
	fig, axes = plt.subplots(nrows=nrow, ncols=1, figsize=(10, 11))
	fig.tight_layout(h_pad=2.5)
	# fig.set_title("Comparison of Fitness Across Benchmarks")

	i = 0
	for db in dim_benchmarks:
		props_db = db.filter_props(props)
		_compare_fitness_on_benchmarks_subplot(fig, axes, i, db, props_db, dim_variants, key_fit=key_fit, print_quantities=print_quantities, use_latex=use_latex)
		i += 1

	if savepath is None:
		savepath = OUTPUT_FOLDER + "/box.pdf"
	fig.savefig(savepath)
	plt.show()



def _compare_fitness_on_benchmarks_subplot(fig, axes, i, benchmark, props, dim_variants, key_fit, print_quantities=False, use_latex=False):
	"""Draws a subplot containing box-and-wshiskers plot for all variants on a single benchmark."""
	assert isinstance(benchmark, dims.Config) and len(benchmark) == 1 # only one filter function for benchmark.
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
