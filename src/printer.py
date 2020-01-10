from . import dims
from . import utils


class CellRenderer(object):
    def __init__(self, condition, editor):
        self.condition = condition
        self.editor = editor

    def __call__(self, *args, **kwargs):
        value = args[0]
        body = args[1]
        if self.condition(value, body):
            return self.editor(value, body)
        else:
            return body


class LatexCommand(CellRenderer):
    def __init__(self, cmdOpen, cmdClose, condition):
        assert isinstance(cmdOpen, str) and isinstance(cmdClose, str)
        editor = lambda value, body: cmdOpen + str(body) + cmdClose
        CellRenderer.__init__(self, condition, editor)


class LatexTextbf(LatexCommand):
    def __init__(self, condition):
        LatexCommand.__init__(self, r"\textbf{", "}", condition)


class LatexTextit(LatexCommand):
    def __init__(self, condition):
        LatexCommand.__init__(self, r"\textit{", "}", condition)


class CellShading(CellRenderer):
    def __init__(self, MinNumber, MidNumber, MaxNumber, MinColor="colorLow", MidColor="colorMedium",
                 MaxColor="colorHigh"):
        """
        :param MinNumber: (float) below or equal to this value everything will be colored fully with MinColor.
          Higher values will be gradiented towards MidColor.
        :param MidNumber: (float) middle point. Values above go towards MaxColor, and values below towards MinColor.
        :param MaxNumber: (float) above or equal to this value everything will be colored fully with MaxColor.
          Lower values will be gradiented towards MidColor.
        :param MinColor: (str) name of the LaTeX color representing the lowest value.
        :param MidColor: (str) name of the LaTeX color representing the middle value. This color is also used for
          gradient, that is closer a given cell value is to the MidNumber, more MidColor'ed it becomes.
        :param MaxColor: (str) name of the LaTeX color representing the highest value."""
        def color_cell(v, body):
            if v == "-" or v == "-" or v.strip().lower() == "nan" or not utils.isfloat(v.strip()):
                return v
            else:
                # Computing color gradient.
                val = float(v.strip())
                if val > MidNumber:
                    PercentColor = max(min(100.0 * (val - MidNumber) / (MaxNumber - MidNumber), 100.0), 0.0)
                    color = "{0}!{1:.1f}!{2}".format(MaxColor, PercentColor, MidColor)
                else:
                    PercentColor = max(min(100.0 * (MidNumber - val) / (MidNumber - MinNumber), 100.0), 0.0)
                    color = "{0}!{1:.1f}!{2}".format(MinColor, PercentColor, MidColor)
                return "\cellcolor{" + color + "}" + str(body)
        condition = lambda v, b: True
        editor = lambda v, b: color_cell(v, b)
        CellRenderer.__init__(self, condition, editor)



class TableHeaderInterface(object):
    def __init__(self):
        self.cells = []

    def removeCell(self, index):
        pass

    def addCell(self, index, cell):
        pass



class TableHeader(TableHeaderInterface):
    def __init__(self, dimCols, layeredHeadline=True, verticalBorder=0, horizontal_border=1, useBooktabs=False):
        TableHeaderInterface.__init__(self)
        assert isinstance(dimCols, dims.Dim)
        self.dimCols = dims.Dim(dimCols.configs[:]) # copy of the original dimCols

        # Each cell of the header is a list of captions of a given Config.
        # Cells of header are associated with the respective columns of the table.
        # Removing a column means that its header cell also will be removed.
        # Adding a column means adding its header cell.
        self.cells = dimCols.get_captions_list()

        self.layeredHeadline = layeredHeadline
        self.verticalBorder = verticalBorder
        self.horizontal_border = horizontal_border
        self.useBooktabs = useBooktabs

    def removeCell(self, index):
        assert isinstance(index, int)
        del self.cells[index]
        # Removing corresponding column dimension
        del self.dimCols[index]

    def addCell(self, index, cell):
        assert isinstance(cell, list)
        self.cells.insert(index, cell)

    def render(self):
        return latex_table_header_cells(self.cells, layered_headline=self.layeredHeadline, vertical_border=self.verticalBorder,
                                        horizontal_border=self.horizontal_border, tabFirstCol=False, useBooktabs=self.useBooktabs)



class EmptyTableHeader(TableHeaderInterface):
    def __init__(self):
        TableHeaderInterface.__init__(self)

    def removeCell(self, index):
        pass

    def addCell(self, index, cell):
        pass

    def render(self):
        return ""




class Table(object):
    """
    Rule: in hierarchical header merged are cells with the same caption on the same level.
    """
    def __init__(self, tableBody, dimCols=None, cellRenderers=None, layeredHeadline=True, verticalBorder=0,
                 horizontalBorder=1, useBooktabs=False):
        if cellRenderers is None:
            cellRenderers = []
        assert isinstance(tableBody, str)
        assert isinstance(cellRenderers, list)
        tableBody = tableBody.strip() # Remove leading and trailing whitespaces
        self.rows = []
        # self.dimCols = dimCols
        self.dimCols = dims.Dim(dimCols.configs[:])  # copy of the original dimCols
        if dimCols is None:
            self.header = EmptyTableHeader()
        else:
            self.header = TableHeader(dimCols, layeredHeadline=layeredHeadline, verticalBorder=verticalBorder,
                                      horizontal_border=horizontalBorder, useBooktabs=useBooktabs)
        self.cellRenderers = cellRenderers
        self.layeredHeadline = layeredHeadline
        self.verticalBorder = verticalBorder
        self.horizontalBorder = horizontalBorder
        self.useBooktabs = useBooktabs
        # Extracting rows
        for line in tableBody.split("\n"):
            cols = [c.strip() for c in line.split("&")]
            if cols[-1].endswith(r"\\"):
                cols[-1] = cols[-1][:-2]
            elif cols[-1].endswith(r"\\\hline"):
                cols[-1] = cols[-1][:-8]
            self.rows.append(cols)

    def removeColumn(self, index):
        assert isinstance(index, int)
        self.header.removeCell(index)
        for row in self.rows:
            del row[index]
        # Removing corresponding column dimension
        del self.dimCols[index]

    def removeColumns(self, indexes):
        assert isinstance(indexes, list)
        for i in sorted(indexes, reverse=True):
            self.removeColumn(i)

    def leaveColumns(self, indexes):
        """Removes all the columns but those indicated by indexes."""
        assert isinstance(indexes, list)
        for i in range(len(self.rows[0])-1, -1, -1):
            if i not in indexes:
                self.removeColumn(i)

    def removeRow(self, index):
        assert isinstance(index, int)
        del self.rows[index]

    def addRow(self, row):
        assert isinstance(row, list)
        assert len(row) == len(self.header.cells), "Number of columns in the row does not match the table header!"
        self.rows.append(row)

    def getText(self, opts=None):
        """Part of the interface of reporting module."""
        return self.render()

    def applyRenderers(self, value):
        text = str(value)
        for rend in self.cellRenderers:
            text = rend(value, text)
        return text

    def getHeaderCells(self):
        return self.header.cells

    def renderTableHeader(self):
        return self.header.render()

    def renderTableBody(self):
        text = ""
        for i, row in enumerate(self.rows):
            rowRendered = [self.applyRenderers(cell) for cell in row]
            text += " & ".join(rowRendered) + r"\\"
            if self.horizontalBorder >= 2 and i < len(self.rows) - 1:
                if self.useBooktabs:
                    text += r"\midrule"
                else:
                    text += r"\hline"
            text += "\n"
        return text

    def render(self, latexizeUnderscores=True, firstColAlign="l", middle_col_align="c"):
        text  = self.renderTableHeader()
        text += self.renderTableBody()
        dimCols = self.dimCols if self.dimCols is not None else dims.Dim.generic_labels(len(self.rows[0]))
        return latex_table_wrapper(text, dimCols, latexize_underscores=latexizeUnderscores, vertical_border=self.verticalBorder,
                                   horizontal_border=self.horizontalBorder, first_col_align=firstColAlign,
                                   middle_col_align=middle_col_align, tabFirstCol=False, useBooktabs=self.useBooktabs)

    def __str__(self):
        return self.render()





def text_listing(props, dim, fun, is_fun_single_prop=False, d_configs="\n\n\n", fun_config_header=None):
    """Returns a text listing of values computed for the specified configs. By default follows a format
    similar to the presented below:

    (*) CONFIG: c1
    vals_c1

    (*) CONFIG: c2
    vals_c2

    .....

    :param props: (dict) all properties files containing experiment's data.
    :param dim: (Dim) dimension along which listing will be created.
    :param fun: (list[dict] => str) function returning text containing processed information (e.g. average)
     for a config-filtered props.
    :param is_fun_single_prop: (bool) if true, then to fun will be passed separately every config-filtered
     prop (e.g. useful for printing best solutions per config). If false, then to fun will be passed whole
     set of config-filtered props (e.g. useful for printing statistics).
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
    :param fun: (list[dict] => str) function returning a cell's content given a list
     of dicts representing relevant data.
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


def text_table_header(dim_cols, d_cols="\t", d_rows="\n"):
    text = ""
    text += d_cols
    values = [c.get_caption() for c in dim_cols]
    text += d_cols.join(values) + d_rows
    return text


def text_table_body(props, dim_rows, dim_cols, fun, d_cols="\t", d_rows="\n"):
    text = ""
    for r in dim_rows:
        filtered_r = r.filter_props(props)
        text += text_table_row(filtered_r, r, dim_cols, fun, d_cols, d_rows)
    return text


def text_table(props, dim_rows, dim_cols, fun, title=None, d_cols="\t", d_rows="\n"):
    """Returns text of the table containing in the cells values from the intersection of configs
    in row and column. By manipulating delimiters LaTeX table may be produced.

    :param props: (dict) all props gathered in the experiment.
    :param dim_rows: (Dim) a dimension for rows.
    :param dim_cols: (Dim) a dimension for columns.
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
    # text += d_cols
    # values = [c.get_caption() for c in dim_cols]
    # text += d_cols.join(values) + d_rows
    #
    # # Printing table's rows.
    # for r in dim_rows:
    # 	filtered_r = r.filter_props(props)
    # 	text += text_table_row(filtered_r, r, dim_cols, fun, d_cols, d_rows)
    text += text_table_header(dim_cols, d_cols=d_cols, d_rows=d_rows)
    text += text_table_body(props, dim_rows, dim_cols, fun, d_cols=d_cols, d_rows=d_rows)
    return text


def latex_table_wrapper(tableBody, dim_cols, latexize_underscores=True, vertical_border=1, horizontal_border=1,
                        first_col_align="l", middle_col_align="c", tabFirstCol=True, useBooktabs=False):
    r"""Responsible for invoking and closing \tabular environment."""
    assert isinstance(tableBody, str)
    assert isinstance(dim_cols, dims.Dim)
    # Tabular prefix
    if tabFirstCol: # TODO: remove tabFirstCol option entirely
        numCols = len(dim_cols.configs) + 1
    else:
        numCols = len(dim_cols.configs)
    if vertical_border >= 2:
        alignments = "|{0}|".format(first_col_align) + "|".join(middle_col_align * (numCols - 1)) + "|"  # and not layered_headline
    elif vertical_border == 1:
        alignments = "|{0}|".format(first_col_align) + (middle_col_align * (numCols - 1)) + "|"
    else:
        alignments = first_col_align + (middle_col_align * (numCols - 1))
    text = r"\begin{tabular}{" + alignments + "}\n"
    if horizontal_border >= 1:
        text += r"\hline" + "\n" if not useBooktabs else r"\toprule" + "\n"
    else:
        text += "\n"

    # Tabular body
    text += tableBody

    # Tabular suffix
    if horizontal_border >= 1:
        text += r"\hline" + "\n" if not useBooktabs else r"\bottomrule" + "\n"
    else:
        text += "\n"
    text += r"\end{tabular}" + "\n"
    if latexize_underscores:
        text = text.replace("_", r"\_")
    return text


def latex_table(props, dim_rows, dim_cols, fun, latexize_underscores=True, layered_headline=False,
                vertical_border=1, first_col_align="l", middle_col_align="c", headerRowNames=None):
    """Returns code of a LaTeX table (tabular environment) created from the given dimensions.

    :param props: (dict) all props gathered in the experiment.
    :param dim_rows: (Dim) a dimension for rows.
    :param dim_cols: (Dim) a dimension for columns.
    :param fun: (list[dict] => str) a function returning a cell's content given a list of
     props "in" the cell.
    :param latexize_underscores: (bool) if set to to true, every underscore ("_") will be
     turned into version acceptable by LaTeX ("\_"). This, however, may be undesired if
     some elements are in math mode and use subscripts.
    :param layered_headline: (bool) if set to to true, headline will be organized into layers
     depending on the configuration.
    :param vertical_border: (int) mode of the vertical borders in the table. Bigger the number,
     more dense the vertical borders. Range of values: 0 - 2.
    :param first_col_align: (str) alignment of the first column. Used as \tabular arguments.
    :param middle_col_align: (str) alignment of the middle columns (all beside the first one).
     Used as \tabular arguments.
    :param headerRowNames: (list[str]) a list of names of the rows of the header.
    :return: (str) code of the LaTeX table.
    """
    assert isinstance(dim_rows, dims.Dim)
    assert isinstance(dim_cols, dims.Dim)

    body = latex_table_header(dim_cols, layered_headline=layered_headline, d_cols=" & ", d_rows="\\\\\n",
                              vertical_border=vertical_border, headerRowNames=headerRowNames)
    body += text_table_body(props, dim_rows, dim_cols, fun, d_cols=" & ", d_rows="\\\\\n")

    text = latex_table_wrapper(body, dim_cols, latexize_underscores=latexize_underscores, vertical_border=vertical_border,
                               first_col_align=first_col_align, middle_col_align=middle_col_align)
    return text


def latex_table_header_cells(cells, layered_headline=False, d_cols=" & ", d_rows="\\\\\n",
                             vertical_border=0, horizontal_border=1, tabFirstCol=True, useBooktabs=False):
    """Produces header for a LaTeX table. In the case of generating layered headline, columns
    dimension is assumed to contain Configs with the same number of filters and corresponding
    configs placed on the same positions (this will always be correct, if '*' was used to
    combine dimensions).
    """
    if layered_headline:
        return latex_table_header_multilayered_cells(cells, d_cols=d_cols, d_rows=d_rows,
                                            vertical_border=vertical_border, horizontal_border=horizontal_border,
                                            tabFirstCol=tabFirstCol, useBooktabs=useBooktabs)
    else:
        return latex_table_header_one_layer_cells(cells, d_cols=d_cols, d_rows=d_rows,
                                                  vertical_border=vertical_border, horizontal_border=horizontal_border,
                                                  tabFirstCol=tabFirstCol)


def latex_table_header(dim_cols, layered_headline=False, d_cols=" & ", d_rows="\\\\\n",
                       vertical_border=0, useBooktabs=False, headerRowNames=None):
    """Produces header for a LaTeX table. In the case of generating layered headline, columns
    dimension is assumed to contain Configs with the same number of filters and corresponding
    configs placed on the same positions (this will always be correct, if '*' was used to
    combine dimensions).
    """
    if layered_headline:
        return latex_table_header_multilayered(dim_cols, d_cols=d_cols, d_rows=d_rows,
                                               vertical_border=vertical_border, useBooktabs=useBooktabs,
                                               headerRowNames=headerRowNames)
    else:
        return latex_table_header_one_layer(dim_cols, d_cols=d_cols, d_rows=d_rows,
                                            vertical_border=vertical_border)


def latex_table_header_one_layer_cells(cells_cols, d_cols=" & ", d_rows="\\\\\n", vertical_border=0, horizontal_border=1, sep="_", tabFirstCol=True):
    chead = [r"\multicolumn{1}{c}{" + sep.join(d) + "}" for d in cells_cols]
    text  = d_cols if tabFirstCol else ""
    text += d_cols.join(chead) + d_rows
    if horizontal_border >= 1:
        text += r"\hline"
    text += "\n"
    return text


def latex_table_header_one_layer(dim_cols, d_cols=" & ", d_rows="\\\\\n", vertical_border=0, horizontal_border=1, tabFirstCol=True):
    chead = [r"\multicolumn{1}{c}{" + d.get_caption() + "}" for d in dim_cols]
    text  = d_cols if tabFirstCol else ""
    text += d_cols.join(chead) + d_rows
    if horizontal_border >= 1:
        text += r"\hline"
    text += "\n"
    return text


def latex_table_header_multilayered_cells(cells_cols, d_cols=" & ", d_rows="\\\\\n", vertical_border=0, horizontal_border=1,
                                          tabFirstCol=True, useBooktabs=False):
    """cells_cols = a list of captions for dimensions"""
    dim_cols = dims.Dim([])
    for cell in cells_cols:
        filters = []
        for c in cell:
            filters.append((c, None))
        dim_cols = dim_cols + dims.Dim(dims.Config(filters))
    return latex_table_header_multilayered(dim_cols, d_cols=d_cols, d_rows=d_rows, vertical_border=vertical_border,
                                           horizontal_border=horizontal_border, tabFirstCol=tabFirstCol, useBooktabs=useBooktabs)


def latex_table_header_multilayered(dim_cols, d_cols=" & ", d_rows="\\\\\n", vertical_border=0, horizontal_border=1,
                                    tabFirstCol=True, useBooktabs=False, headerRowNames=None):
    """Produces a multi-layered header of the LaTeX table. Multi-layered means that there is some
     hierarchy to dimensions of the experiment and the subdimensions will be presented under their
     parent dimension.

    :param dim_cols: (Dim) dimensions of the columns.
    :param d_cols: (str) separator for columns.
    :param d_rows: (str) separator for rows.
    :param vertical_border: (int) takes values from 0 to 2. Bigger the number, the more dense vertical
     border would be.
    :param horizontal_border: (int) takes values from 0 to 2. Bigger the number, the more dense horizontal
     border would be.
    :param tabFirstCol: (bool) if true, then column headers will start from the second column instead of the
     the first. Scenarios in which tabFirstCol should be false are rather rare.
    :param useBooktabs: (bool) if true, then instead of \hline in the middle \midrule will be used.
    :param headerRowNames: (list[str]) a list of names of the rows of the header.
    :return: (str) header of the table in LaTeX.
    """

    assert isinstance(dim_cols, dims.Dim)
    num_layers = max([len(c.filters) for c in dim_cols.configs])  # num of layers in the example filter
    if headerRowNames is None:
        headerRowNames = [""] * num_layers
    assert isinstance(headerRowNames, list)
    assert len(headerRowNames) >= num_layers, "headerRowNames has {0} entries, but it should have as many entries as layers to be created ({1})".format(len(headerRowNames), num_layers)

    # Going from the highest layer to the lowest.
    def produce_lines(dimens, layer_no):
        if layer_no == num_layers - 1: #len(dimens[0]) == 1 or ...
            # Only a single row, use a simplified routine.

            headerCells = []
            for i, d in enumerate(dimens):
                if vertical_border == 0:
                    align = "c"
                else:
                    align = "c|"
                    # Necessary, because apparently multicolumn{c|} overrides {|l|}
                    # on the border of the table
                    if i == 0:
                        align = "|" + align
                headerCells.append(r"\multicolumn{1}{" + align + "}{" + d.get_caption() + "}")

            firstColSep = headerRowNames[layer_no] + d_cols if tabFirstCol else ""
            if horizontal_border >= 1:
                ender = r"\hline" + "\n" if not useBooktabs else r"\midrule" + "\n"
            else:
                ender = "\n"
            return firstColSep + d_cols.join(headerCells) + d_rows + ender
        text = ""
        top_filters_list = [] # stores tuples (filter, numContiguous)
        last = None
        for conf in dimens:
            if last is None or conf.filters[0] != last:
                last = conf.filters[0]
                top_filters_list.append((conf.filters[0], 1))
            elif conf.filters[0] == last:
                filt, numCont = top_filters_list[-1]
                top_filters_list[-1] = (filt, numCont + 1)

        # Producing top-level header.
        buffer = []
        for i in range(len(top_filters_list)):
            f, foccurs = top_filters_list[i]
            fname = str(f[0])  # name of the filter
            align = "c"
            if vertical_border >= 1:
                align += "|"
                # if i != 0:
                #     align += "|"
                # Necessary, because apparently multicolumn{c|} overrides {|l|}
                # on the border of the table
                if i == 0:
                    align = "|" + align
            ftext = r"\multicolumn{" + str(foccurs) + "}{" + align + "}{" + fname + "}" # \multicolumn{6}{c}{$EPS$}
            buffer.append(ftext)

        # We need to add subconfigs to the queue. Removing first filter from every config.
        subconfigs_queue = []
        for conf in dimens:
            new_filters = conf.filters[1:]
            if len(new_filters) > 0:
                subconfigs_queue.append(dims.Config(new_filters))
            else:
                # Add dummy config; multiline header needs to know that column's border needs to continue
                subconfigs_queue.append(dims.Config("", lambda p: False))

        if tabFirstCol:
            text += headerRowNames[layer_no] + d_cols
        text += d_cols.join(buffer) + d_rows
        text += produce_lines(dims.Dim(subconfigs_queue), layer_no + 1)
        return text

    return produce_lines(dim_cols, 0)



def decorate_table(table_text, convert_fun, d_cols=" & ", d_rows="\\\\\n"):
    """Transforms text of the table by applying converter function to each element of this table.

    :param table_text: (str) text of the table.
    :param convert_fun: (str => str) a function to be applied to each element of the table.
    :param d_cols: (str) delimiter between columns.
    :param d_rows: (str) delimiter between rows.
    :return: (str) text of the converted table.
    """
    def process_cell(s):
        return str(convert_fun(s))
    if d_cols not in table_text:
        return table_text # delimiter was not present

    splitted = table_text.split(d_cols)
    new_text = ""
    for i in range(0, len(splitted)):
        s = splitted[i]
        last_in_row = d_rows in s
        if last_in_row:
            two_elems = s.split(d_rows)
            decorated = process_cell(two_elems[0]) + d_rows
            if len(two_elems) > 1 and two_elems[1] != '':
                decorated += process_cell(two_elems[1])
        else:
            decorated = convert_fun(s)

        new_text += decorated
        if i < len(splitted)-1:
            new_text += d_cols
    return new_text



def table_color_map(text, MinNumber, MidNumber, MaxNumber, MinColor="colorLow", MidColor="colorMedium",
                    MaxColor="colorHigh", funValueExtractor=None):
    """Creates a table with cells colored depending on their values ("color map"). Colored will be only
    cells containing numbers.

    :param text: (str) text of the table.
    :param MinNumber: (float) below or equal to this value everything will be colored fully with MinColor.
     Higher values will be gradiented towards MidColor.
    :param MidNumber: (float) middle point. Values above go towards MaxColor, and values below towards MinColor.
    :param MaxNumber: (float) above or equal to this value everything will be colored fully with MaxColor.
     Lower values will be gradiented towards MidColor.
    :param MinColor: (str) name of the LaTeX color representing the lowest value.
    :param MidColor: (str) name of the LaTeX color representing the middle value. This color is also used for
     gradient, that is closer a given cell value is to the MidNumber, more MidColor'ed it becomes.
    :param MaxColor: (str) name of the LaTeX color representing the highest value.
    :param funValueExtractor: (lambda) a function applied to cell's text returning the value based on which
     color will be computed. Content of a cell will remain unchanged.
    :return: (str) text of the table with added \cellcolor commands with appropriate colors as arguments.
    """
    if funValueExtractor is None:
        funValueExtractor = lambda x: x
    def color_cell(s):
        extracted = str(funValueExtractor(s)).strip()
        if s == "-" or s.strip().lower() == "nan" or not utils.isfloat(extracted):
            return s
        else:
            # Computing color gradient.
            val = float(extracted)
            if val > MidNumber:
                PercentColor = max(min(100.0 * (val - MidNumber) / (MaxNumber-MidNumber), 100.0), 0.0)
                color = "{0}!{1:.1f}!{2}".format(MaxColor, PercentColor, MidColor)
            else:
                PercentColor = max(min(100.0 * (MidNumber - val) / (MidNumber - MinNumber), 100.0), 0.0)
                color = "{0}!{1:.1f}!{2}".format(MinColor, PercentColor, MidColor)
            return "\cellcolor{" + color + "}" + s


    return decorate_table(text, color_cell, d_cols=" & ")