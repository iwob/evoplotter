from subprocess import call

class ReportPDF(object):
	"""PDF generated from an automatically generated LaTeX source containing results of experiments.
	Content of the PDF is defined by a template defined by a user.
	"""
	geometry_params = "[paperwidth=65cm, paperheight=40cm, margin=0.3cm]"
	packages = ["[utf8]{inputenc}",
	            geometry_params + "{geometry}",
	            "[table]{xcolor}"]

	def __init__(self, blocks = None):
		if blocks is None:
			blocks = []
		self.blocks = blocks
		self.blocks.append(self.get_preamble())
		self.root = BlockEnvironment("document", [])
		self.blocks.append(self.root)


	def add(self, block):
		"""Adds a block inside a document environment."""
		self.root.append(block)

	def apply(self):
		"""Creates report for the data and returns its LaTeX code."""
		text = ""
		for b in self.blocks:
			text += b.get_text()
		return text

	def save(self, filename):
		"""Saves LaTeX source file under the given name."""
		file_ = open(filename, 'w')
		file_.write(self.apply())
		file_.close()

	def save_and_compile(self, filename):
		"""Saves LaTeX source file under the given name and compiles it using pdflatex."""
		self.save(filename)
		call(["pdflatex", filename])
		noext = filename[:filename.rfind('.')]
		call(["rm", noext+".aux", noext+".log"])

	def get_preamble(self):
		text = r"\documentclass[12pt]{article}" + "\n\n"
		for p in self.packages:
			text += r"\usepackage" + p + "\n"
		text += "\n"
		text += r"\DeclareUnicodeCharacter{00A0}{~} % replacing non-breaking spaces" + "\n"
		text += r"\setlength{\tabcolsep}{10pt}" + "\n"
		text += "\n\n"
		return BlockLatex(text)



class BlockBundle(object):
	"""Simply stores several blocks in a collection."""
	def __init__(self, contents):
		self.contents = contents
	def get_text(self):
		text = ""
		for b in self.contents:
			text += b.get_text()
		return text
	def add(self, b):
		self.contents.append(b)


class BlockLatex(object):
	"""Simply stores as a single string blob several LaTeX instructions or whole text paragraphs."""
	def __init__(self, text):
		self.text = text
	def get_text(self):
		return self.text


class BlockEnvironment(object):
	def __init__(self, name, contents):
		assert isinstance(contents, list)
		self.name = name
		self.contents = contents
	def get_text(self):
		text = r"\begin{" + self.name + "}\n\n"
		for b in self.contents:
			text += b.get_text()
		text += r"\end{" + self.name + "}\n"
		return text
	def append(self, block):
		self.contents.append(block)


class BlockSection(BlockBundle):
	def __init__(self, title, contents):
		self.title = title
		BlockBundle.__init__(self, contents)
		self.cmd = "section"
	def get_text(self):
		text = "\\" + self.cmd + "{" + self.title + "}\n"
		text += super(BlockSection, self).get_text()
		return text

class BlockSubSection(BlockSection):
	def __init__(self, title, contents):
		BlockSection.__init__(self, title, contents)
		self.cmd = "subsection"

class BlockSubSubSection(BlockSection):
	def __init__(self, title, contents):
		BlockSection.__init__(self, title, contents)
		self.cmd = "subsubsection"




# By default color schemes color high values and go down to white with lower values.
# If _r is appended to the color scheme name, then colored are low values instead.

color_scheme_blue = BlockLatex(r"""\definecolor{lightred}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{lightyellow}{rgb}{0.83, 0.89, 0.98} % light blue
\definecolor{lightgreen}{rgb}{0.63, 0.79, 0.95} % blue
""")
color_scheme_green = BlockLatex(r"""\definecolor{lightred}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{lightyellow}{rgb}{0.76, 0.98, 0.76} % lightgreen
\definecolor{lightgreen}{rgb}{0.66, 0.90, 0.66} % green
""")
color_scheme_yellow = BlockLatex(r"""\definecolor{lightred}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{lightyellow}{rgb}{0.98, 0.91, 0.71} % light yellow
\definecolor{lightgreen}{rgb}{1.0, 0.75, 0.0} % yellow
""")
color_scheme_red = BlockLatex(r"""\definecolor{lightred}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{lightyellow}{rgb}{0.95, 0.6, 0.6}
\definecolor{lightgreen}{rgb}{0.8, 0, 0} % red
""")
color_scheme_red_r = BlockLatex(r"""\definecolor{lightred}{rgb}{0.8, 0, 0} % red
\definecolor{lightyellow}{rgb}{0.95, 0.6, 0.6}
\definecolor{lightgreen}{rgb}{1.0, 1.0, 1.0} % white
""")