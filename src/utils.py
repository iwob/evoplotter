import os
import shutil
import subprocess
from . import dims



def get_unique_values(data, extr):
    """Collects unique values in the data.

    :param data: (list[dict[str]]) list storing all the data.
    :param extr: (lambda) function for extracting value from a dictionary.
    :return: (set[str]) a set of unique values.
    """
    assert isinstance(data, list), "Data should be a list!"
    s = set()
    for p in data:
        s.add(extr(p))
    return s


def read_lines(path):
    """Reads all lines from the specified file.

    :param path: (str) path to a file.
    :return: (list[str]) list of lines in the file.
    """
    f = open(path, 'r')
    return f.readlines()


def file_ends_with_extension(f, exts):
    assert isinstance(exts, list), "Extensions must be provided in the form of a list!"
    for e in exts:
        if f.endswith(e):
            return True
    return False


def load_properties(lines, sep="=", comment_marker="#"):
    """Creates a dictionary for properties provided as a list of lines. Split on the first found sep is conducted.

    :param lines: (list[str]) lines of the file.
    :param sep: (str) separator between property key and its value.
    :param comment_marker: (str) marker signaling that this line is a comment. Must be the first character in the row excluding whitespaces.
    :return: (dict[str,str]) dictionary representing a properties file.
    """
    res = {}
    for r in lines:
        if sep not in r or r.strip()[0] == comment_marker:
            continue
        i = r.index(sep)
        key = r[:i].strip()
        content = "".join(r[i+1:]).strip()
        res[key] = content
    return res


def save_properties_file(p, path_file):
    """Saves a dictionary in a given file using the Java properties format.

    :param path_file: (str) path to a file.
    :param p: (dict[str,any]) properties dictionary.
    :return: (dict[str,str]) dictionary with property names as keys.
    """
    f = open(path_file, "w")
    for k in sorted(p.keys()):
        f.write("{0} = {1}\n".format(k, p[k]))
    f.close()


def load_properties_file(path_file, add_file_path=False):
    """Creates a dictionary for properties loaded from a given file.

    :param path_file: (str) path to a file.
    :param add_file_path: (bool) specifies if path to a file on disk should be stored in dictionary.
     The path will be stored under 'evoplotter.file' key.
    :return: (dict[str,str]) dictionary with property names as keys.
    """
    lines = read_lines(path_file)
    p = load_properties(lines)
    if add_file_path:
        p["evoplotter.file"] = path_file
    return p


def load_properties_dir(path_dir, exts=None, ignoreExts=None, add_file_path=False, predicate=None):
    """Creates a dictionary for properties loaded from files in the given directory. All subdirectories will be recursively traversed.

    :param path_dir: (str) path to a directory from which paths will be read.
    :param exts: (list[str]) list of accepted extensions. None means that all extensions are to be accepted.
    :param ignoreExts: (list[str]) list of ignored extensions. None means that no extensions are ignored.
    :param add_file_path: (bool) specifies if path to a file on disk should be stored in dictionary.
     The path will be stored under 'evoplotter.file' key.
    :param predicate: (lambda[dict,bool]) a predicate determining, if a dictionary loaded from file will be added to the list.
    :return: (list(dict[str,str])) list of dictionaries created for each file in the folder.
    """
    res = []
    for root, subFolders, files in os.walk(path_dir):
        for f in files:
            if (exts is None or file_ends_with_extension(f, exts)) and \
               (ignoreExts is None or not file_ends_with_extension(f, ignoreExts)):
                full_name = os.path.join(root, f)
                p = load_properties_file(full_name, add_file_path=add_file_path)
                if predicate is None or predicate(p):
                    res.append(p)
    return res


def load_properties_dirs(dirs, exts=None, ignoreExts=None, add_file_path=False, predicate=None):
    """Loads properties files from the specified directories.  All subdirectories will be recursively traversed.

    :param dirs: (list[str]) list of paths to directories.
    :param exts: (list[str]) list of accepted extensions. None means that all extensions are to be accepted.
    :param ignoreExts: (list[str]) list of ignored extensions. None means that no extensions are ignored.
    :param add_file_path: (bool) specifies if path to a file on disk should be stored in dictionary.
     The path will be stored under 'evoplotter.file' key.
    :param predicate: (lambda[dict,bool]) a predicate determining, if a dictionary loaded from file will be added to the list.
    :return: (list[dict[str,str]]) list of dictionaries created for each file in the specified folders.
    """
    res = []
    for d in dirs:
        res.extend(load_properties_dir(d, exts, ignoreExts, add_file_path=add_file_path, predicate=predicate))
    return res


def str2list(s, sep=","):
    """Converts a string representing a list with floats to a Python list object filled with floats."""
    return [float(x.strip()) for x in s.split(sep)]


def isfloat(value):
    """Checks, if the given number is a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def ensure_dir(file_path):
    assert file_path[-1] == "/", "directory path must end with '/'"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_clear_dir(file_path):
    assert file_path[-1] == "/", "directory path must end with '/'"
    directory = os.path.dirname(file_path)
    if os.path.exists(directory):
        shutil.rmtree(directory, ignore_errors=False)
    os.makedirs(directory)


def save_to_file(file_path, content):
    file = open(file_path, "w")
    file.write(content)
    file.close()


def compile_graphviz_file_to_pdf(path):
    cmd = "dot -Tpdf {0} -o {1}".format(path, path[:path.rfind(".")] + ".pdf")
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()


def normalize_name(name):
    pairs = {"$":"", "/":"", "\\":"", " ":"", "\t":"", "}":"", "{":""}
    for k, v in pairs.items():
        name = name.replace(k, pairs[k])
    return name


def reorganizeExperimentFiles(props, dim, target_dir, maxRuns, key_file="evoplotter.file"):
    """Takes a list of dictionaries storing experiment's results and the dimension, and creates
     a new folder with experiments where results are stored hierarchically as defined by
     dimensions of the experiment.

    :param props: (list[dict]) list of dictionaries to be processed.
    :param dim: (Dim) dimension which will determine the structure of the target directory.
    :param target_dir: (str) path to a directory which will be created in order to store the
     cleaned results.
    :param maxRuns: (int) maximum number of property files which would land in a single created
     directory in the cleaned results.
    :param key_file: (str) a key under which is stored the location of the original file of the
     properties dict. That file will be copied to appropriate target_dir subfolder.
    """
    assert isinstance(props, list)
    assert isinstance(dim, dims.Dim)
    assert target_dir[-1] == "/", "directory path must end with '/'"
    ensure_clear_dir(target_dir)

    for config in dim.configs:
        accum_path = target_dir
        for name, filter in config:
            accum_path += normalize_name(name) + "/"
            ensure_dir(accum_path)

        # at this point accum_path is the path to the folder for a given config
        data = config.filter_props(props)[:maxRuns]  # take only first maxRuns dicts
        for p in data:
            assert key_file in p, "Key containing the path of the file was not present in the dictionary!"
            p_full_path = p[key_file]
            p_name = p_full_path[p_full_path.rfind("/")+1:]
            shutil.copy(p_full_path, accum_path + p_name)



def deleteFilesByPredicate(props, predicate, simulate=False, verbose=True, key_file="evoplotter.file"):
    """Deletes all physical files meeting a predicate for a list of dictionaries storing
     experiment's results.

    :param props: (list[dict]) list of dictionaries to be processed.
    :param predicate: (lambda) a boolean function specifying which files are to be removed.
    :param simulate: (bool) if true, then files won't be removed from disk.
    :param verbose: (bool) if true, then names of the deleted files will be printed on screen.
    :param key_file: (str) a key under which is stored the location on disk of the original file of the
     properties dict.
    """
    assert isinstance(props, list)

    for p in props:
        if key_file in p and predicate(p):
            path = p[key_file]
            if not simulate:
                os.remove(path)
            if verbose:
                print("File removed: {0}".format(path))