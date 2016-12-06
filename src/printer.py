from . import dims



def text_listing(props, dim, fun, is_fun_single_prop=False, d_configs="\n\n\n", fun_config_header=None):
	"""Returns a text listing of values computed for the specified configs. By default follows a format similar to the presented below:

	(*) CONFIG: c1
	vals_c1

	(*) CONFIG: c2
	vals_c2

	.....

	:param props: (dict) all properties files containing experiment's data.
	:param dim: (Dim) dimension along which listing will be created.
	:param fun: (list[dict] => str) function returning text containing processed information (e.g. average) for a config-filtered props.
	:param is_fun_single_prop: (bool) if true, then to fun will be passed seperately every config-filtered prop (e.g. useful for printing best solutions per config). If false, then to fun will be passed whole set of config-filtered props (e.g. useful for printing statistics).
	:param d_configs: (str) delimiter between configs in the listing.
	:param fun_config_header: (Config => str) Function which returns text of the header describing a configuration.
	:return: (str) Text of a listing.
	"""
	if fun_config_header is None:
		fun_config_header = lambda c: "(*) CONFIG: {0}\n".format(c.get_caption())

	text = ""
	for config in dim:
		text += fun_config_header(config)
		filtered = config.filter_props(props)
		if is_fun_single_prop:
			for p in filtered:
				text += fun(p)
		else:
			text += str(fun(filtered))
		text += d_configs
	return text



def save_to_file(text, path):
	"""Just saves a text to the file. This is a convenience function."""
	with open(path, 'w') as file_:
		file_.write(text)



def text_table_row(props_row, config_row, dim_cols, fun, d_cols="\t", d_rows="\n"):
	"""Constructs a single row of a table.

	:param props_row: (dict) all props which applies to the given row.
	:param config_row: (Config) a concrete configuration defined as a list of filters.
	:param dim_cols: (Dim) a dimension, defined as a list of configurations.
	:param fun: (list[dict] => str) function returning a cell's content given a list of dicts representing relevant data.
	:param d_cols: (str) delimiter separating columns.
	:param d_rows: (str) delimiter separating rows.
	:return: (str) text of the table's row.
	"""
	assert isinstance(config_row, dims.Config)
	assert isinstance(dim_cols, dims.Dim)

	text = config_row.get_caption() + d_cols # print name of the row
	for c in dim_cols:
		filtered = c.filter_props(props_row)
		cell_content = fun(filtered)
		if cell_content is not None:
			text += str(cell_content)
		text += d_cols
	return text[:-len(d_cols)] + d_rows



def text_table(props, dim_rows, dim_cols, fun, title=None, d_cols="\t", d_rows="\n"):
	"""Returns text of the table containing in the cells values from the intersection of configs in row and column. By manipulating delimiters LaTeX table may be produced.

	:param props: (dict) all props gathered in the experiment.
	:param dim_rows: (Dim) a dimension of rows.
	:param dim_cols: (Dim) a dimension of columns.
	:param fun: (list[dict] => str) a function returning a cell's content given a list of props "in" the cell.
	:param title: (str) a title to be placed before the table. By default there is no title.
	:param d_cols: (str) delimiter separating columns.
	:param d_rows: (str) delimiter separating rows.
	:return: (str) text of the table.
	"""
	assert isinstance(dim_rows, dims.Dim)
	assert isinstance(dim_cols, dims.Dim)

	text = ""
	if title is not None:
		text += title + "\n"

	# Printing header.
	text += d_cols
	values = [c.get_caption() for c in dim_cols]
	text += d_cols.join(values) + d_rows

	# Printing table's rows.
	for r in dim_rows:
		filtered_r = r.filter_props(props)
		text += text_table_row(filtered_r, r, dim_cols, fun, d_cols, d_rows)
	return text



def latex_table(props, dim_rows, dim_cols, fun, title=None):
	return text_table(props, dim_rows, dim_cols, fun, title, d_cols=" & ", d_rows="\\\\\n")
