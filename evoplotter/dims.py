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
            self.configs = []
            for c in configs:
                if isinstance(c, ConfigList):
                    self.configs.append(c)
                elif isinstance(c, tuple):
                    self.configs.append(Config(c[0], c[1]))
                elif isinstance(c, str):  # creating only labels
                    self.configs.append(Config(c, None))
                else:
                    raise Exception("Incorrect arguments passed to Dimension: {0}".format(c))
        elif isinstance(configs, ConfigList):
            self.configs = [configs]
        else:
            raise Exception("Incorrect arguments passed to Dimension!")

    def __len__(self):
        return len(self.configs)

    def __iter__(self):
        for c in self.configs:
            yield c

    def __getitem__(self, item):
        if isinstance(item, tuple) or isinstance(item, list):
            configs = []
            for i in item:
                configs.append(self.configs[i])
            return configs
        else:
            return self.configs[item]

    def __delitem__(self, key):
        del self.configs[key]

    def __mul__(self, other):
        assert isinstance(other, Dim) or isinstance(other, ConfigList), "Dimension may be merged only with other Dimension or ConfigList."
        if isinstance(other, Dim):
            if len(other) == 0:
                return self
            else:
                return Dim(generate_configs([self, other]))
        else:
            return self * Dim(other)

    def __add__(self, other):
        if isinstance(other, ConfigList):
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

    def __reversed__(self):
        return Dim(list(reversed(self.configs)))

    def __str__(self):
        return str([c.get_caption() for c in self.configs])

    def insert(self, index, config):
        """Inserts config at the particular position on the list of dimension's configs."""
        if isinstance(config, Dim):
            config = config[0]
        self.configs.insert(index, config)

    def get_captions_list(self):
        """Returns a list of captions of all configs in this dimension."""
        return [c.get_captions_list() for c in self.configs]

    def get_captions(self):
        """Returns a list of captions of all configs in this dimension."""
        return [c.get_caption() for c in self.configs]

    def get_predicates(self):
        """Returns a list of predicates of all configs in this dimension."""
        all_lam = []
        for c in self.configs:
            lam = lambda p: all([f[1](p) for f in c.filters])
            all_lam.append(lam)
        return all_lam

    def filter_out_outsiders(self, props):
        """Returns properties in which contained are only elements belonging to one of
        the configs in this dimension. Note that dimension values (configs) do not have to
        cover the whole possible space or to be disjoint. This functions allows to remove
        unnecessary configs and thus may reduce computation time.
        """
        return [p for p in props if any([c.filter(p) for c in self.configs])]

    def filter_props(self, props):
        """Returns all properties files satisfied by at least one config."""
        assert isinstance(props, list), "filter_props expects a list of property files"
        return [p for p in props if self.filter(p)]

    def filter(self, p):
        """Returns True, if a given property file is covered by at least one config."""
        return any(c.filter(p) for c in self.configs)

    def sort(self, key=None):
        """Sorts this dimension alphabetically on the names of Configs within it."""
        if key is None:
            key = lambda x: x
        self.configs.sort(key=key)
        return self

    def copy(self):
        """Creates a copy of this dimension."""
        return Dim(self.configs[:])

    def dim_true_within(self, name="ALL"):
        """Returns a new dimension accepting any configuration accepted by this 'parent' dimension."""
        return Dim(ConfigOr(name, self.configs))

    @classmethod
    def dim_true(cls, name="ALL"):
        """Returns a new dimension accepting all configurations."""
        return Dim(Config(name, lambda p: True, method=None))

    @classmethod
    def generic_labels(cls, num, prefix="A"):
        """Creates a Dim object containing num generic names."""
        configs = [prefix + str(x) for x in range(num)]
        return Dim(configs)

    @classmethod
    def from_names(cls, names):
        """Creates a dummy Dim object with config names but no lambdas."""
        configs = [Config(name, None) for name in names]
        return Dim(configs)

    @classmethod
    def from_data(cls, props, extr):
        """Creates a Dim object by collecting all unique values in the data.
        Extractor (extr) is a function used to get values."""
        s = utils.get_unique_values(props, extr)
        configs = [Config(el, lambda p, extr=extr, el=el: extr(p) == el) for el in s]
        return Dim(configs)

    @classmethod
    def from_dict(cls, props, key, nameFun=None):
        """Creates a Dim object by collecting all unique values under the specified
        key in the dictionaries."""
        s = utils.get_unique_values(props, lambda p: p[key])
        configs = []
        for el in s:
            kwargs = {key: el}
            name = el if nameFun is None else nameFun(el)
            configs.append(Config(name, lambda p, key=key, el=el: p[key] == el, **kwargs))
        return Dim(configs)

    @classmethod
    def from_dict_postprocess(cls, props, key, fun):
        """Creates a Dim object by collecting all unique values under the specified
        key in the dictionaries after the retrieved values are processed in a specified way."""
        values = utils.get_unique_values(props, lambda p: p[key])
        values = {fun(x) for x in values}
        configs = []
        for v in values:
            kwargs = {key: v}
            configs.append(Config(v, lambda p, key=key, v=v, fun=fun: fun(p[key]) == v, **kwargs))
        return Dim(configs)

    @classmethod
    def from_dict_value_match(cls, key, values):
        """Creates a dimension based on a check if the dictionary contains a given value under the
        provided key. Dimension spawns all the provided values.

        :param key: (str) a key used to obtain an actual value from the dictionary.
        :param values: (list[any]) values, which will be matched to the value obtained from the dict."""
        assert isinstance(key, str)
        assert isinstance(values, list) and len(values) > 0
        configs = []
        for v in values:
            kwargs = {key: v}
            configs.append(Config(v, lambda p, key=key, v=v: p[key] == v, **kwargs))
        return Dim(configs)



def generate_configs(dims_list):
    """Returns a list of configurations for a dimension."""
    final_filters = []
    final_values = []
    _generate_filters_helper(dims_list, [], {}, final_filters, final_values)
    return [ConfigList(flist, **values) for flist, values in zip(final_filters, final_values)]

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




class ConfigList(object):
    """ConfigList is defined as a list of filters. A 'filter' is a tuple containing a name
    and a predicate. Names of the filters are used during generation of plots/tables, and predicate
    is used to leave only properties dicts which were generated in a run matching this configuration.
    If more than one filter is defined, conjunction of all predicates is considered."""
    def __init__(self, *filters, **kwargs):
        self.stored_values = kwargs
        if len(filters) == 1:
            if isinstance(filters[0], list):
                # filters: [(name, lambda)]
                self.filters = filters[0]
            else:
                # filters: (name, lambda)
                self.filters = [filters[0]]
        else:
            # filters: name, lambda
            self.filters = [tuple(filters)]

    def __len__(self):
        return len(self.filters)

    def __iter__(self):
        for c in self.filters:
            yield c

    def __getitem__(self, item):
        return self.filters[item]

    def __delitem__(self, key):
        del self.filters[key]

    def __lt__(self, other):
        n_other = "_".join(x[0] for x in other.filters)
        n_self = "_".join([x[0] for x in self.filters])
        return n_self < n_other

    def __str__(self):
        return "ConfigList({0})".format(self.get_caption())

    def __call__(self, *args, **kwargs):
        return self.filter(args[0])

    def head(self):
        """Returns the first filter defined in this config. Convenience function."""
        return self.filters[0]

    def get_caption(self, sep="/"):
        """Returns a merged name of this Config. This name is generated by merging
        names of filters which constitute it."""
        return sep.join([str(f[0]) for f in self.filters])

    def get_captions_list(self):
        """Returns a list containing names of all filters in this Config."""
        return [str(f[0]) for f in self.filters]

    def contains_filter(self, otherFilt):
        """Checks, if this Config contains specified filter."""
        return any([f == otherFilt for f in self.filters])

    def contains_filter_name(self, otherFiltName):
        """Checks, if this Config contains a filter with the specified name."""
        return any([f[0] == otherFiltName for f in self.filters])

    def filter_props(self, props):
        """Returns all properties files satisfied by a conjunction of all filters in this Config."""
        assert isinstance(props, list), "filter_props expects a list of property files"
        return [p for p in props if self.filter(p)]

    def filter(self, p):
        """Checks, if properties file p is satisfied by a *conjunction* of all filters in this Config."""
        assert isinstance(p, dict), "filter expects property file"
        for f in self.filters:
            if not f[1](p):
                return False
        return True




class Config(ConfigList):
    """Defines a single configuration of the experiment. Is equivalent to the tuple (name, filter)."""
    def __init__(self, name, filter, **kwargs):
        assert not isinstance(filter, list), "Config associates the name and a particular filter. " \
                                             "To use multiple filters, please use ConfigOr or ConfigAnd."
        assert filter is None or callable(filter), "Filter must be a callable object or None."
        self.name = name
        self.filter = filter
        ConfigList.__init__(self, (name, self), **kwargs)

    def __getitem__(self, item):
        return self.name if item == 0 else self

    def __call__(self, *args, **kwargs):
        return self.filter(args[0])

    def __str__(self):
        return "Config({0})".format(self.get_caption())

    def filter(self, p):
        return self.filter(p)




class ConfigAnd(ConfigList):
    """A list of configs composed by conjunction. E.g.: accept solutions both with p=1
    and p=2 (in this case, impossibility)."""
    def __init__(self, name, configs, **kwargs):
        assert isinstance(configs, list), "Configs should be provided as a list."
        assert len(configs) > 0, "Trying to create ConfigAnd with empty configs list."
        self.name = name
        self.configs = configs
        ConfigList.__init__(self, (name, self), **kwargs)

    def __call__(self, *args, **kwargs):
        return self.filter(args[0])

    def __str__(self):
        return "ConfigAnd({0})".format(self.get_caption())

    def filter(self, p):
        """Checks, if properties file p is satisfied by a *disjunction* of all filters in this Config."""
        assert isinstance(p, dict), "filter expects property file"
        for f in self.configs:
            if not f(p):
                return False
        return True



class ConfigOr(ConfigList):
    """A list of configs composed by disjunction. E.g.: accept solutions either with p=1 or p=2."""
    def __init__(self, name, configs, **kwargs):
        assert isinstance(configs, list), "Configs should be provided as a list."
        assert len(configs) > 0, "Trying to create ConfigOr with empty configs list."
        self.name = name
        self.configs = configs
        ConfigList.__init__(self, (name, self), **kwargs)

    def __call__(self, *args, **kwargs):
        return self.filter(args[0])

    def __str__(self):
        return "ConfigOr({0})".format(self.get_caption())

    def filter(self, p):
        """Checks, if properties file p is satisfied by a *disjunction* of all filters in this Config."""
        assert isinstance(p, dict), "filter expects property file"
        for f in self.configs:
            if f(p):
                return True
        return False




dim_all = Dim([Config("", lambda p: True)])