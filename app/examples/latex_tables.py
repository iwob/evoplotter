from src import printer



def isfloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False


def table_color_map(text, MinNumber, MidNumber, MaxNumber, MinColor, MidColor, MaxColor):
	def color_cell(s):
		if s == "-" or s == "-" or not isfloat(s.strip()):
			return s
		else:
			# Computing color gradient.
			val = float(s.strip())
			if val > MidNumber:
				PercentColor = max(min(100.0 * (val - MidNumber) / (MaxNumber-MidNumber), 100.0), 0.0)
				color = "{0}!{1}!{2}".format(MaxColor, str(PercentColor), MidColor)
			else:
				PercentColor = max(min(100.0 * (MidNumber - val) / (MidNumber - MinNumber), 100.0), 0.0)
				color = "{0}!{1}!{2}".format(MinColor, str(PercentColor), MidColor)
			return "\cellcolor{" + color + "}" + s


	res = printer.decorate_table(text, color_cell, d_cols=" & ")
	print("\n\n\nDECORATED:")
	print(res)





text = r"""
& & & \multicolumn{3}{c}{EPS-L} & \multicolumn{3}{c}{EPS-B} \\
& $GP$ & $GP_{T}$ & $C$ & $V$ & $CV$ & $C$ & $V$ & $CV$\\
\texttt{Keijzer12} & 15 & 11331 & 772 & 488 & 1579 & 15440 & 21173 & 28354\\
\texttt{Koza1} & 5 & 291 & 670 & - & 801 & 652 & - & 696\\
\texttt{Koza1-p} & 5 & 963 & 892 & - & 972 & 978 & - & 982\\
\texttt{Koza1-2D} & 16 & 7636 & 793 & 479 & 1791 & 9077 & 16281 & 23034\\
\texttt{Koza1-p-2D} & 15 & 9206 & 750 & 511 & 1726 & 11986 & 12391 & 27875\\
"""
MinNumber = 0.0
MaxNumber = 30000.0
MidNumber = 600 #MaxNumber / 2
MinColor = "lightgreen"
MidColor = "yellow"
MaxColor = "lightred"
table_color_map(text, MinNumber, MidNumber, MaxNumber, MinColor, MidColor, MaxColor)







text = r"""
\texttt{Keijzer12} & 0.058 & 0.004 & 0.104 & 0.229 & 0.106 & 0.297\\
\texttt{Koza1} & 0.080 & - & 0.058 & 0.127 & - & 0.120\\
\texttt{Koza1-p} & 0.078 & - & 0.060 & 0.113 & - & 0.112\\
\texttt{Koza1-2D} & 0.065 & 0.006 & 0.118 & 0.276 & 0.117 & 0.372\\
\texttt{Koza1-p-2D} & 0.062 & 0.004 & 0.112 & 0.301 & 0.051 & 0.407\\
"""
MinNumber = 0.0
MaxNumber = 0.5
MidNumber = 0.2 #MaxNumber / 2
MinColor = "lightgreen"
MidColor = "yellow"
MaxColor = "lightred"
table_color_map(text, MinNumber, MidNumber, MaxNumber, MinColor, MidColor, MaxColor)








text = r"""
\texttt{Keijzer12} & 0 & 0 & 0 & 0 & 1 & 39 & 1 & 0\\
\texttt{Koza1} & 19 & 68 & 33 & - & 32 & 100 & - & 100\\
\texttt{Koza1-p} & 0 & 0 & 5 & - & 3 & 100 & - & 100\\
\texttt{Koza1-2D} & 1 & 12 & 2 & 0 & 11 & 80 & 21 & 23\\
\texttt{Koza1-p-2D} & 0 & 0 & 0 & 0 & 1 & 75 & 0 & 0\\
"""
MinNumber = 0
MaxNumber = 100
MidNumber = 20 #MaxNumber / 2
MinColor = "lightred"
MidColor = "yellow"
MaxColor = "lightgreen"
table_color_map(text, MinNumber, MidNumber, MaxNumber, MinColor, MidColor, MaxColor)