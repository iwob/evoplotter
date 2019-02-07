
def generateRanksVertical(label, data):
    text = "\n" + label + "\n"
    text += r"\begin{tabular}{lr}" + "\n"
    text += r"\hline" + "\n"
    text += "{0} & {1}\\\\".format("method", "rank") + "\n"
    text += r"\hline" + "\n"
    values = sorted(zip(data.keys(), data.values()), key=lambda tup: tup[1])
    # print(values)
    for k, v in values:
        val = "{0:.2f}".format(round(v, 2))
        text += "{0} & {1}\\\\".format(k, val) + "\n"
    text += r"\hline" + "\n"
    text += r"\end{tabular}"
    # print("\n\nFRIEDMAN for: " + label)
    print(text)
    print(r"%")

def generateRanksHorizontal(label, data):
    text = "\n" + label + "\n"
    text += r"\begin{tabular}{" + "l" + ("r" * len(data))  + "}\n"
    text += r"\hline" + "\n"
    values = sorted(zip(data.keys(), data.values()), key=lambda tup: tup[1])
    # print(values)
    rowMethods = "method"
    rowRanks = "rank"
    for k, v in values:
        val = "{0:.2f}".format(round(v, 2))
        rowMethods += " & {0}".format(k)
        rowRanks += " & {0}".format(val)
    rowMethods += r"\\" + "\n"
    rowRanks += r"\\" + "\n"
    text += rowMethods
    text += rowRanks
    # print("\n\nFRIEDMAN for: " + label)
    text += r"%\hline" + "\n"
    text += r"\end{tabular}"
    print(text)
    print(r"\smallskip")


def generate(label, data):
    generateRanksVertical(label, data)
    # generateRanksHorizontal(label, data)


ranksPropsMSETests3 = {"\mnameProps\_0.1": 1.6, "\mname\_0.1": 2.8,
                       "\mnameProps\_0.01":3.8, "GP\_0.1":4.1,
                       "\mname\_0.01": 4.2, "GP\_0.01":4.5}

ranksPropsMSETests5 = {"\mnameProps\_0.1": 1.6, "\mname\_0.1": 2.5,
                       "GP\_0.1":4.1, "\mnameProps\_0.01": 4.3,
                       "\mname\_0.01": 4.4, "GP\_0.01":4.5}

ranksPropsMSETests10 = {"\mnameProps\_0.1": 1.7, "\mname\_0.1": 2.1,
                        "GP\_0.1":4.1, "GP\_0.01":4.2,
                        "\mname\_0.01": 4.4, "\mnameProps\_0.01":4.5 }

ranksPropsMSETests3510 = {"\mnameProps\_0.1": 1.6, "\mname\_0.1": 2.5,
                          "GP\_0.1":4.0, "\mnameProps\_0.01":4.2,
                          "\mname\_0.01": 4.3, "GP\_0.01":4.4}


print("\n\n")
generate("MSE + properties: 3 tests", ranksPropsMSETests3)
generate("MSE + properties: 5 tests", ranksPropsMSETests5)
generate("MSE + properties: 10 tests", ranksPropsMSETests10)
generate("MSE + properties: 3,5,10 tests", ranksPropsMSETests3510)






ranksPropsTests3 = {"\mnameProps\_0.1": 1.4, "\mnameProps\_0.01":1.8,
                    "\mname\_0.1": 3.6, "\mname\_0.01": 3.9,
                    "GP\_0.1":5.1, "GP\_0.01":5.3}

ranksPropsTests5 = {"\mnameProps\_0.1": 1.5, "\mnameProps\_0.01":1.6,
                    "\mname\_0.1": 3.3, "\mname\_0.01": 4.2,
                    "GP\_0.1":5.1, "GP\_0.01":5.4}

ranksPropsTests10 = {"\mnameProps\_0.01": 1.5, "\mnameProps\_0.1":1.7,
                    "\mname\_0.1": 3.1, "\mname\_0.01": 4.2,
                    "GP\_0.1":5.1, "GP\_0.01":5.3}

ranksPropsTests3510 = {"\mnameProps\_0.1": 1.5, "\mnameProps\_0.01":1.6,
                       "\mname\_0.1": 3.3, "\mname\_0.01": 4.1,
                       "GP\_0.1":5.1, "GP\_0.01":5.4}

print("\n\n")
generate("properties: 3 tests", ranksPropsTests3)
generate("properties: 5 tests", ranksPropsTests5)
generate("properties: 10 tests", ranksPropsTests10)
generate("properties: 3,5,10 tests", ranksPropsTests3510)


