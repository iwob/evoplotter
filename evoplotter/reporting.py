import subprocess
from subprocess import call, STDOUT


class ReportPDF(object):
    """PDF generated from an automatically generated LaTeX source containing results of experiments.
    Content of the PDF is defined by a template defined by a user.
    """
    GEOM_PARAMS = "[paperwidth=65cm, paperheight=40cm, margin=0.3cm]"

    def __init__(self, contents=None, packages=None, geometry_params=GEOM_PARAMS, user_declarations="", tabcolsep=8):
        if contents is None:
            contents = []
        if packages is None:
            packages = []
        else:
            packages = [p if p[0]=="{" and p[-1]=="}" else "{" + p + "}" for p in packages]
        assert isinstance(contents, list)
        assert isinstance(packages, list)
        self.tabcolsep = tabcolsep
        self.geometry_params = geometry_params
        self.user_declarations = user_declarations
        self.root = BlockEnvironment("document", [BlockBundle(contents)])
        self.packages = ["[utf8]{inputenc}",
                         self.geometry_params + "{geometry}",
                         "[usenames,dvipsnames,table]{xcolor}",
                         "{hyperref}",
                         "{graphicx}",
                         "{booktabs}",
                         "{float}"]
        self.packages.extend(packages)
        self.blocks = [self.get_preamble(), self.root]

    def get_packages_list(self):
        return self.packages

    def add(self, block):
        """Adds a block inside a document environment."""
        self.root.append(block)

    def apply(self):
        """Creates report for the data and returns its LaTeX code."""
        text = ""
        for b in self.blocks:
            text += b.getText(opts={})
        return text

    def save(self, filename):
        """Saves LaTeX source file under the given name."""
        file_ = open(filename, 'w')
        file_.write(self.apply())
        file_.close()

    def save_and_compile(self, filename, output_dir=None):
        """Saves LaTeX source file under the given name and compiles it using pdflatex."""
        output_dir = "." if output_dir is None else str(output_dir)  # str in case output_dir was provided as a Path
        filename = str(filename)  # str in case filename was provided as a Path
        self.save(filename)
        try:
            subprocess.check_output(["pdflatex", "-interaction=nonstopmode", f"-output-directory={output_dir}", filename], stderr=STDOUT, universal_newlines=True)
            subprocess.check_output(["pdflatex", "-interaction=nonstopmode", f"-output-directory={output_dir}", filename], stderr=STDOUT, universal_newlines=True) # for index to catch up
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
        text += r"\setlength{\tabcolsep}{" + str(self.tabcolsep) + "pt}" + "\n"
        text += "\n"
        text += self.user_declarations + "\n"
        text += "\n\n"
        return BlockLatex(text)



class BlockBundle(object):
    """Simply stores several blocks in a collection."""
    def __init__(self, contents):
        assert isinstance(contents, list)
        self.contents = contents
    def getText(self, opts):
        return self.merge_items(opts=opts)
    def merge_items(self, opts):
        text = ""
        d = opts
        for b in self.contents:
            if isinstance(b, str):
                text += b
            else:
                text += b.getText(opts=d)
        return text
    def add(self, b):
        self.contents.append(b)


class BlockLatex(object):
    """Simply stores as a single string blob several LaTeX instructions or whole text paragraphs."""
    def __init__(self, text):
        self.text = text
    def __str__(self):
        return self.text
    def getText(self, opts):
        return self.text


class BlockEnvironment(object):
    def __init__(self, name, contents):
        assert isinstance(contents, list)
        self.name = name
        self.contents = contents

    def getText(self, opts):
        text = r"\begin{" + self.name + "}\n\n"
        for b in self.contents:
            if isinstance(b, str):
                text += b
            else:
                text += b.getText(opts=opts)
        text += r"\end{" + self.name + "}\n"
        return text

    def append(self, block):
        self.contents.append(block)


class Section(BlockBundle):
    def __init__(self, title, contents=None, label=None):
        if contents is None:
            contents = []
        assert isinstance(contents, list)
        BlockBundle.__init__(self, contents)
        self.title = title
        self.label = label
        self.level = 0
        self.cmd = "section"

    def getText(self, opts):
        text = "\\" + self.cmd + "{" + self.title + "}"
        text += r"\label{" + self.label + "}\n" if self.label is not None else "\n"
        opts["section_level"] = self.level + 1 # to pass deeper
        text += self.merge_items(opts=opts)
        opts["section_level"] = self.level  # retract for the other cmds on the same level
        return text


class SectionRelative(BlockBundle):
    """Section which detects the current level of nested sections and posits itself
    either under the last section or on the same level, depending on user's options.
    
    move argument in constructor defines, on which level relative to the current
    """
    def __init__(self, title, contents=None, move=0, label=None):
        if contents is None:
            contents = []
        assert isinstance(contents, list)
        BlockBundle.__init__(self, contents)
        self.title = title
        self.move = move
        self.label = label

    def getText(self, opts):
        opts["section_level"] = opts.get("section_level", 0) + self.move
        sect_level = opts["section_level"]  # remember current section level
        assert sect_level <= 2, "Latex supports nested sections only up to subsubsection."
        subs = "sub" * opts["section_level"]
        text = "\\" + subs + "section{" + self.title + "}"
        text += r"\label{" + self.label + "}\n" if self.label is not None else "\n"
        opts["section_level"] += 1  # to pass deeper
        text += self.merge_items(opts)
        opts["section_level"] = sect_level  # retract for the other cmds on the same level
        return text


class Subsection(Section):
    def __init__(self, title, contents=None, label=None):
        if contents is None:
            contents = []
        assert isinstance(contents, list)
        Section.__init__(self, title, contents, label=label)
        self.level = 1
        self.cmd = "subsection"


class Subsubsection(Section):
    def __init__(self, title, contents=None, label=None):
        if contents is None:
            contents = []
        assert isinstance(contents, list)
        Section.__init__(self, title, contents, label=label)
        self.level = 2
        self.cmd = "subsubsection"


class FloatFigure:
    def __init__(self, path, caption=None, label=None, pos="H", graphics_opts=""):
        self.path = path
        self.caption = caption
        self.label = label
        self.pos = pos
        self.graphics_opts = graphics_opts

    def getText(self, opts):
        text  = r"\begin{figure}[" + self.pos + "]\n"
        text += r"\includegraphics[" + self.graphics_opts + "]{" + self.path + "}\n"
        if self.caption is not None:
            text += r"\caption{" + self.caption + "}\n"
        if self.label is not None:
            text += r"\label{" + self.label + "}\n"
        text += r"\end{figure}" + "\n\n"
        return text




class ColorScheme3:
    def __init__(self, colors, comments=None, nameLow="colorLow", nameMedium="colorMedium", nameHigh="colorHigh"):
        if comments is None:
            comments = ["", "", ""]
        assert isinstance(colors, list), "ColorScheme3 expects a list with RGB values."
        assert len(colors) == 3, "ColorScheme3 must be composed from exactly three colors."
        assert isinstance(comments, list)
        assert len(comments) == 3
        self.colors = colors
        self.comments = comments
        self.nameLow = nameLow
        self.nameMedium = nameMedium
        self.nameHigh = nameHigh

    def getColorNames(self):
        return [self.nameLow, self.nameMedium, self.nameHigh]

    def toBlockLatex(self):
        return BlockLatex(self.__str__())

    def __reversed__(self):
        newColors = list(reversed(self.colors[:]))
        newComments = list(reversed(self.comments[:]))
        return ColorScheme3(newColors, comments=newComments, nameLow=self.nameLow,
                            nameMedium=self.nameMedium, nameHigh=self.nameHigh)

    def __str__(self):
        text = ""
        for rgb, comment, colorName in zip(self.colors, self.comments, self.getColorNames()):
            comment = " % {0}".format(comment) if comment != "" else ""
            text += "\definecolor{" + str(colorName) + "}{rgb}{" + str(rgb) + "}" + str(comment) + "\n"
        return text


color_scheme_darkgreen = ColorScheme3(["1.0, 1.0, 1.0", "0.3, 0.6, 0.3", "0.0, 0.4, 0.0"],
                                      ["white", "light green", "dark green"])
color_scheme_gray_light = ColorScheme3(["1.0, 1.0, 1.0", "0.9, 0.9, 0.9", "0.75, 0.75, 0.75"],
                                       ["white", "light gray", "gray"])
color_scheme_gray_dark = ColorScheme3(["1.0, 1.0, 1.0", "0.75, 0.75, 0.75", "0.5, 0.5, 0.5"],
                                       ["white", "light gray", "gray"])
color_scheme_blue = ColorScheme3(["1.0, 1.0, 1.0", "0.83, 0.89, 0.98", "0.63, 0.79, 0.95"],
                                 ["white", "light blue", "blue"])
color_scheme_green = ColorScheme3(["1.0, 1.0, 1.0", "0.76, 0.98, 0.76", "0.66, 0.90, 0.66"],
                                  ["white", "light green", "green"])
color_scheme_yellow = ColorScheme3(["1.0, 1.0, 1.0", "0.98, 0.91, 0.71", "1.0, 0.75, 0.0"],
                                   ["white", "light yellow", "yellow"])
color_scheme_violet = ColorScheme3(["1.0, 1.0, 1.0", "0.85, 0.65, 0.92", "0.65, 0.45, 0.85"],
                                   ["white", "light violet", "violet"])
color_scheme_teal = ColorScheme3(["1.0, 1.0, 1.0", "0.67, 0.87, 0.88", "0.47, 0.72, 0.73"],
                                 ["white", "light teal", "teal"])
color_scheme_brown = ColorScheme3(["1.0, 1.0, 1.0", "0.96, 0.8, 0.62", "0.76, 0.6, 0.42"],
                                  ["white", "light brown", "brown"])
color_scheme_red = ColorScheme3(["1.0, 1.0, 1.0", "0.95, 0.6, 0.6", "0.8, 0, 0"],
                                ["white", "light red", "red"])

color_scheme_red2yellow2green = ColorScheme3(["0.94, 0.5, 0.5", "1.0, 1.0, 0.0", "0.56, 0.93, 0.56"],
                                             ["red", "yellow", "green"])
color_scheme_red2white2green = ColorScheme3(["0.94, 0.5, 0.5", "1.0, 1.0, 1.0", "0.56, 0.93, 0.56"],
                                            ["red", "white", "green"])
color_scheme_red2white2darkgreen = ColorScheme3(["0.94, 0.5, 0.5", "1.0, 1.0, 1.0", "0.0, 0.4, 0.0"],
                                            ["red", "white", "green"])
