import os


def read_lines(path):
	"""Reads all lines from the specified file.

	:param path: (str) path to a file.
	:return: (list[str]) list of lines in the file.
	"""
	f = open(path, 'r')
	return f.readlines()


def file_extension_ok(f, exts):
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


def load_properties_file(path_file):
	"""Creates a dictionary for properties loaded from a given file.

	:param path_file: (str) path to a file.
	:return: (dict[str,str]) dictionary with property names as keys.
	"""
	lines = read_lines(path_file)
	return load_properties(lines)


def load_properties_dir(path_dir, exts=None):
	"""Creates a dictionary for properties loaded from files in the given directory. All subdirectories will be recursively traversed.

	:param path_dir: (str) path to a directory from which paths will be read.
	:param exts: (list[str]) list of accepted extensions. None means that all extensions are to be accepted.
	:return: (list(dict[str,str])) list of dictionaries created for each file in the folder.
	"""
	res = []
	for root, subFolders, files in os.walk(path_dir):
		for f in files:
			if exts is None or file_extension_ok(f, exts):
				full_name = os.path.join(root, f)
				res.append(load_properties_file(full_name))
	return res


def load_properties_dirs(dirs, exts=None):
	"""Loads properties files from the specified directories.  All subdirectories will be recursively traversed.

	:param dirs: (list[str]) list of paths to directories.
	:param exts: (list[str]) list of accepted extensions. None means that all extensions are to be accepted.
	:return: (list[dict[str,str]]) list of dictionaries created for each file in the specified folders.
	"""
	res = []
	for d in dirs:
		res.extend(load_properties_dir(d, exts))
	return res


def str2list(s, sep=", "):
	"""Converts a string representing a list with floats to a Python list object filled with floats."""
	return [float(x) for x in s.split(sep)]



def isfloat(value):
	"""Checks, if the given number is a float."""
	try:
		float(value)
		return True
	except ValueError:
		return False