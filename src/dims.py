from . import utils


class Dim(object):
    """Stores a list of configurations used during the experiment.

    For example, lets say that we want to compare variants of GP with different
    probability of mutation and also at the same time with different tournament sizes.
    The first dimension would be the probability of mutation. We will define this
    dimension with a list of possible configurations, e.g.:
        Config('mut0.2', pred_mut02), Config('mut0.5', pred_mut05), ...
    The other dimension would be a list of tuples describing possible tournament sizes, e.g.:
        Config('k4', pred_k4), Config('k7', pred_k7), ...

    Dimensions may be combined with each other to generate concrete "real" configurations
    of the experiment. This is done by means of a Cartesian product between them. This
    is realized with * operator, and in this case it would lead to the following
    configurations of a newly created "aggregate" dimension:

        Config([('mut0.2', pred_mut02), ('k4', pred_k4)]),
        .....,
        Config([('mut0.5', pred_mut05), ('k7', pred_k7)])


    A simpler example:
        dim1 = [A1, A2, A3], dim2 = [B1, B2].

    By combining those two dimensions (Dim1 * Dim2) we obtain a single dimension defined as:
        dim3 = [(A1,B1), (A1,B2), ..., (A3,B1), (A3,B2)].

    In such a case experiment logs will be filtered by successive applications of predicates.

    Operator + may be used to add a configuration to already existing dimension. This
    may be useful for instance for the control random algorithm, which is a single
    complete configuration (not dependent on any considered parameters).
    """

    def __init__(self, configs):
        if isinstance(configs, list):
            self.configs = configs
        elif isinstance(configs, Config):
            self.configs = [configs]
        else:
            raise Exception("Incorrect arguments passed to Dimension!")

    def __len__(self):
        return len(self.configs)

    def __iter__(self):
        for c in self.configs:
            yield c

    def __getitem__(self, item):
        return self.configs[item]

    def __mul__(self, other):
        assert isinstance(other, Dim), "Dimension may be merged only with other Dimension."
        if len(other) == 0:
            return self
        else:
            return Dim(generate_configs([self, other]))

    def __add__(self, other):
        if isinstance(other, Config):
            configs = self.configs[:]
            configs.append(other)
            return Dim(configs)
        elif isinstance(other, Dim):
            configs = self.configs[:]
            configs.extend(other.configs[:])
            return Dim(configs)
        else:
            raise Exception("To the Dimension may be added only a Config or other Dimension.")
    __rmul__ = __mul__

    def get_captions(self):
        """Returns a list of captions of all configs in this dimension."""
        return [c.get_caption() for c in self.configs]

    def filter_out_outsiders(self, props):
        """Returns properties in which contained are only elements belonging to one of
        configs in this dimension. Note that dimension values (configs) do not have to
        cover the whole possible space or to be disjoint. This functions allows to remove
        unnecessary configs and thus may reduce computation time.
        """
        return [p for p in props if any([c.filter(p) for c in self.configs])]

    def sort(self):
        """Sorts this dimension alphabetically on the names of Configs within it."""
        self.configs.sort()
        return self

    @classmethod
    def from_data(cls, props, extr):
        """Creates a Dim object by collecting all unique values in the data.
        Extractor (extr) is a function used to get values."""
        s = utils.get_unique_values(props, extr)
        configs = [Config(el, lambda p, extr=extr, el=el: extr(p) == el) for el in s]
        return Dim(configs)

    @classmethod
    def from_dict(cls, props, key):
        """Creates a Dim object by collecting all unique values under the specified
        key in the dictionaries."""
        s = utils.get_unique_values(props, lambda p: p[key])
        configs = []
        for el in s:
            kwargs = {key: el}
            configs.append(Config(el, lambda p, key=key, el=el: p[key] == el, **kwargs))
        return Dim(configs)




def generate_configs(dims_list):
    """Returns a list of configurations for a dimension."""
    final_filters = []
    final_values = []
    _generate_filters_helper(dims_list, [], {}, final_filters, final_values)
    return [Config(flist, **values) for flist, values in zip(final_filters, final_values)]

def _generate_filters_helper(cur_dims, cur_filters, cur_values, final_filters, final_values):
    assert isinstance(cur_filters, list)
    assert isinstance(cur_values, dict)
    if len(cur_dims) == 0:
        final_filters.append(cur_filters)
        final_values.append(cur_values)
    else:
        for config in cur_dims[0]:
            new_filters = cur_filters[:]
            new_filters.extend(config.filters)
            new_values = cur_values.copy()
            new_values.update(config.stored_values)
            _generate_filters_helper(cur_dims[1:], new_filters, new_values, final_filters, final_values)




class Config(object):
    """Defines a single configuration of the experiment. Config may be partial or
    complete. Complete config fully describes a variant of the experiment. Partial
    config may leave some of its aspects unspecified (note: partial configs are
    useful for aggregating results).

    Config is defined as a list of filters. Filter is a tuple containing a name
    and a predicate. Name describes a filter's function/role and is used during
    drawing of plots, and predicate is used to leave only properties files which
    were generated in a run under this configuration. If more than one filter is
    defined, conjunction of all the predicates is considered.
    """
    def __init__(self, *filters, **kwargs):
        self.stored_values = kwargs
        if len(filters) == 1:
            if isinstance(filters[0], list):
                self.filters = filters[0]
            else:
                self.filters = [filters[0]]
        else:
            # Create Config with a filter being a tuple of arbitrary length.
            self.filters = [tuple(filters)]
        assert len(self.filters) > 0, "Trying to create Config with empty filters list!"

    def __len__(self):
        return len(self.filters)

    def __iter__(self):
        for c in self.filters:
            yield c

    def __lt__(self, other):
        n_other = "_".join(x[0] for x in other.filters)
        n_self = "_".join([x[0] for x in self.filters])
        return n_self < n_other

    def __str__(self):
        return "Config({0})".format(self.get_caption())

    def head(self):
        """Returns the first filter defined in this config. Convenience function."""
        return self.filters[0]

    def get_caption(self, sep="_"):
        """Returns a merged name of this Config. This name is generated by merging
        names of filters which constitute it."""
        return sep.join([str(f[0]) for f in self.filters])

    def contains_filter(self, otherFilt):
        """Checks, if this Config contains specified filter."""
        return any([f == otherFilt for f in self.filters])

    def contains_filter_name(self, otherFiltName):
        """Checks, if this Config contains a filter with the specified name."""
        return any([f[0] == otherFiltName for f in self.filters])

    def filter_props(self, props):
        """Returns all properties files satisfied by a conjunction of all filters in this Config."""
        return [p for p in props if self.filter(p)]

    def filter(self, p):
        """Checks, if properties file p is satisfied by a conjunction of all filters in this Config."""
        for f in self.filters:
            if not f[1](p):
                return False
        return True