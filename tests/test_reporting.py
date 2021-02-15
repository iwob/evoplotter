import unittest
from evoplotter.reporting import *

class TestsPrinter(unittest.TestCase):

    def test_sections(self):
        s1 = Section("s1")
        ss1 = Subsection("s1.1")
        s1.add(ss1)
        text = ReportPDF(contents=[s1]).apply()
        res = r"""\documentclass[12pt]{article}

\usepackage[utf8]{inputenc}
\usepackage[paperwidth=65cm, paperheight=40cm, margin=0.3cm]{geometry}
\usepackage[usenames,dvipsnames,table]{xcolor}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}

\DeclareUnicodeCharacter{00A0}{~} % replacing non-breaking spaces
\setlength{\tabcolsep}{8pt}




\begin{document}

\section{s1}
\subsection{s1.1}
\end{document}
"""
        self.assertEqual(res, text)

    def test_relative_sections1(self):
        s1 = SectionRelative("s1")
        s2 = SectionRelative("s1.1", move=1)
        s3 = SectionRelative("s1.1.1", move=1)
        s4 = SectionRelative("s2", move=-1)
        s5 = SectionRelative("s3")
        text = ReportPDF(contents=[s1, s2, s3, s4, s5]).apply()
        res = r"""\documentclass[12pt]{article}

\usepackage[utf8]{inputenc}
\usepackage[paperwidth=65cm, paperheight=40cm, margin=0.3cm]{geometry}
\usepackage[usenames,dvipsnames,table]{xcolor}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}

\DeclareUnicodeCharacter{00A0}{~} % replacing non-breaking spaces
\setlength{\tabcolsep}{8pt}




\begin{document}

\section{s1}
\subsection{s1.1}
\subsubsection{s1.1.1}
\subsection{s2}
\subsection{s3}
\end{document}
"""
        self.assertEqual(res, text)


    def test_relative_sections2(self):
        s1 = SectionRelative("s1")
        s2 = SectionRelative("s2", move=1, contents=[SectionRelative("s3")])
        s4 = SectionRelative("s4", move=1)
        text = ReportPDF(contents=[s1, s2, s4]).apply()
        res = r"""\documentclass[12pt]{article}

\usepackage[utf8]{inputenc}
\usepackage[paperwidth=65cm, paperheight=40cm, margin=0.3cm]{geometry}
\usepackage[usenames,dvipsnames,table]{xcolor}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}

\DeclareUnicodeCharacter{00A0}{~} % replacing non-breaking spaces
\setlength{\tabcolsep}{8pt}




\begin{document}

\section{s1}
\subsection{s2}
\subsubsection{s3}
\subsubsection{s4}
\end{document}
"""
        self.assertEqual(res, text)


    def test_relative_sections3(self):
        s1 = SectionRelative("s1")
        s2 = SectionRelative("s2", move=1, contents=[SectionRelative("s3", move=1), SectionRelative("s4", move=1)])
        s4 = SectionRelative("s5", move=1)
        with self.assertRaises(Exception) as context:
            text = ReportPDF(contents=[s1, s2, s4]).apply()
        self.assertEquals("Latex supports nested sections only up to subsubsection.", str(context.exception))
