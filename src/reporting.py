import subprocess
from subprocess import call, STDOUT


class ReportPDF(object):
    """PDF generated from an automatically generated LaTeX source containing results of experiments.
    Content of the PDF is defined by a template defined by a user.
    """
    GEOM_PARAMS = "[paperwidth=65cm, paperheight=40cm, margin=0.3cm]"

    def __init__(self, contents = None, geometry_params=GEOM_PARAMS):
        if contents is None:
            contents = []
        assert isinstance(contents, list)
        self.geometry_params = geometry_params
        self.root = BlockEnvironment("document", [BlockBundle(contents)])
        self.blocks = [self.get_preamble(), self.root]

    def get_packages_list(self):
        return ["[utf8]{inputenc}",
                self.geometry_params + "{geometry}",
                "[table]{xcolor}",
                "{hyperref}"]

    def add(self, block):
        """Adds a block inside a document environment."""
        self.root.append(block)

    def apply(self):
        """Creates report for the data and returns its LaTeX code."""
        text = ""
        for b in self.blocks:
            text += b.get_text(opts={})
        return text

    def save(self, filename):
        """Saves LaTeX source file under the given name."""
        file_ = open(filename, 'w')
        file_.write(self.apply())
        file_.close()

    def save_and_compile(self, filename):
        """Saves LaTeX source file under the given name and compiles it using pdflatex."""
        self.save(filename)
        try:
            subprocess.check_output(["pdflatex", "-interaction=nonstopmode", filename], stderr=STDOUT, universal_newlines=True)
            subprocess.check_output(["pdflatex", "-interaction=nonstopmode", filename], stderr=STDOUT, universal_newlines=True) # for index to catch up
        except subprocess.CalledProcessError as exc:
            print("Status: FAIL, return code: {0}, msg: {1}".format(exc.returncode, exc.output.replace("\\n", "\n")))
        noext = filename[:filename.rfind('.')]
        call(["rm", "-f", noext+".aux", noext+".log", noext+".bbl", noext+".blg", noext+".out"])

    def get_preamble(self):
        text = r"\documentclass[12pt]{article}" + "\n\n"
        for p in self.get_packages_list():
            text += r"\usepackage" + p + "\n"
        text += "\n"
        text += r"\DeclareUnicodeCharacter{00A0}{~} % replacing non-breaking spaces" + "\n"
        text += r"\setlength{\tabcolsep}{10pt}" + "\n"
        text += "\n\n"
        return BlockLatex(text)



class BlockBundle(object):
    """Simply stores several blocks in a collection."""
    def __init__(self, contents):
        assert isinstance(contents, list)
        self.contents = contents
    def get_text(self, opts):
        return self.merge_items(opts=opts)
    def merge_items(self, opts):
        text = ""
        d = opts
        for b in self.contents:
            text += b.get_text(opts=d)
        return text
    def add(self, b):
        self.contents.append(b)


class BlockLatex(object):
    """Simply stores as a single string blob several LaTeX instructions or whole text paragraphs."""
    def __init__(self, text):
        self.text = text
    def get_text(self, opts):
        return self.text


class BlockEnvironment(object):
    def __init__(self, name, contents):
        assert isinstance(contents, list)
        self.name = name
        self.contents = contents

    def get_text(self, opts):
        text = r"\begin{" + self.name + "}\n\n"
        for b in self.contents:
            text += b.get_text(opts=opts)
        text += r"\end{" + self.name + "}\n"
        return text

    def append(self, block):
        self.contents.append(block)


class Section(BlockBundle):
    def __init__(self, title, contents=None):
        if contents is None:
            contents = []
        assert isinstance(contents, list)
        BlockBundle.__init__(self, contents)
        self.title = title
        self.level = 0
        self.cmd = "section"

    def get_text(self, opts):
        text = "\\" + self.cmd + "{" + self.title + "}\n"
        opts["section_level"] = self.level + 1 # to pass deeper
        text += self.merge_items(opts=opts)
        opts["section_level"] = self.level  # retract for the other cmds on the same level
        return text


class SectionRelative(BlockBundle):
    """Section which detects the current level of nested sections and posits itself
    either under the last section or on the same level, depending on user's options.
    
    move argument in constructor defines, on which level relative to the current
    """
    def __init__(self, title, contents=None, move=0):
        if contents is None:
            contents = []
        assert isinstance(contents, list)
        BlockBundle.__init__(self, contents)
        self.title = title
        self.move = move

    def get_text(self, opts):
        opts["section_level"] = opts.get("section_level", 0) + self.move
        sect_level = opts["section_level"]  # remember current section level
        assert sect_level <= 2, "Latex supports nested sections only up to subsubsection."
        subs = "sub" * opts["section_level"]
        text = "\\" + subs + "section{" + self.title + "}\n"
        opts["section_level"] += 1  # to pass deeper
        text += self.merge_items(opts)
        opts["section_level"] = sect_level  # retract for the other cmds on the same level
        return text


class Subsection(Section):
    def __init__(self, title, contents=None):
        if contents is None:
            contents = []
        assert isinstance(contents, list)
        Section.__init__(self, title, contents)
        self.level = 1
        self.cmd = "subsection"


class Subsubsection(Section):
    def __init__(self, title, contents=None):
        if contents is None:
            contents = []
        assert isinstance(contents, list)
        Section.__init__(self, title, contents)
        self.level = 2
        self.cmd = "subsubsection"




# By default color schemes color high values and go down to white with lower values.
# If _r is appended to the color scheme name, then colored are low values instead.

color_scheme_blue = BlockLatex(r"""\definecolor{colorLow}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{colorMedium}{rgb}{0.83, 0.89, 0.98} % light blue
\definecolor{colorHigh}{rgb}{0.63, 0.79, 0.95} % blue
""")
color_scheme_green = BlockLatex(r"""\definecolor{colorLow}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{colorMedium}{rgb}{0.76, 0.98, 0.76} % light green
\definecolor{colorHigh}{rgb}{0.66, 0.90, 0.66} % green
""")
color_scheme_green_r = BlockLatex(r"""\definecolor{colorLow}{rgb}{0.66, 0.90, 0.66} % green
\definecolor{colorMedium}{rgb}{0.76, 0.98, 0.76} % light green
\definecolor{colorHigh}{rgb}{1.0, 1.0, 1.0} % white
""")
color_scheme_yellow = BlockLatex(r"""\definecolor{colorLow}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{colorMedium}{rgb}{0.98, 0.91, 0.71} % light yellow
\definecolor{colorHigh}{rgb}{1.0, 0.75, 0.0} % yellow
""")
color_scheme_violet = BlockLatex(r"""\definecolor{colorLow}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{colorMedium}{rgb}{0.85, 0.65, 0.92} % light violet
\definecolor{colorHigh}{rgb}{0.65, 0.45, 0.85} % violet
""")
color_scheme_teal = BlockLatex(r"""\definecolor{colorLow}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{colorMedium}{rgb}{0.67, 0.87, 0.88} % light teal
\definecolor{colorHigh}{rgb}{0.47, 0.72, 0.73} % teal
""")
color_scheme_brown = BlockLatex(r"""\definecolor{colorLow}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{colorMedium}{rgb}{0.96, 0.8, 0.62} % light teal
\definecolor{colorHigh}{rgb}{0.76, 0.6, 0.42} % teal
""")
color_scheme_red = BlockLatex(r"""\definecolor{colorLow}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{colorMedium}{rgb}{0.95, 0.6, 0.6} % light red
\definecolor{colorHigh}{rgb}{0.8, 0, 0} % red
""")
color_scheme_red_r = BlockLatex(r"""\definecolor{colorLow}{rgb}{0.8, 0, 0} % red
\definecolor{colorMedium}{rgb}{0.95, 0.6, 0.6} % light red
\definecolor{colorHigh}{rgb}{1.0, 1.0, 1.0} % white
""")
color_scheme_red2yellow2green = BlockLatex(r"""\definecolor{colorLow}{rgb}{0.94, 0.5, 0.5} % red
\definecolor{colorMedium}{rgb}{1.0, 1.0, 0.0} % yellow
\definecolor{colorHigh}{rgb}{0.56, 0.93, 0.56} % green
""")
color_scheme_green2yellow2red = BlockLatex(r"""\definecolor{colorLow}{rgb}{0.56, 0.93, 0.56} % green
\definecolor{colorMedium}{rgb}{1.0, 1.0, 0.0} % yellow
\definecolor{colorHigh}{rgb}{0.94, 0.5, 0.5} % red
""")