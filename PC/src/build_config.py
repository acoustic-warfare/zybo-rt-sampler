import json

"Create and open a config files"
c_config = open('./src/config.h', "w")
py_config = open('./config.py', "w")

#Open config JSON file
cfg = open('./src/config.json')
data = json.load(cfg)

for constants in data["general"].items():
    if constants[0] == "expression":
        for expression in constants[1].items():
            c_line = "#define " + expression[0] + " " + expression[1] + "\n"
            py_line = expression[0] + " = " + expression[1] + "\n"

            c_config.write(c_line)
            py_config.write(py_line)
    else: 
        #For c header config
        line = "#define " + constants[0] + " "
        if isinstance(constants[1], str):
            line += "\"" + constants[1] + "\""
        else:
            line += str(constants[1])
        line += "\n"
        c_config.write(line)

        #For python config
        line = constants[0] + " = "
        if isinstance(constants[1], str):
            line += "\"" + constants[1] + "\""
        else:
            line += str(constants[1])
        line += "\n"
        py_config.write(line)

for constants in data["python"].items():
    #Imports
    if constants[0] == "imports":
        for lib in constants[1]:
            py_config.write("import " + lib + "\n")
    else:
    #Expression handler
        if constants[0] == "expression":
            for expr in constants[1].items():
                line = expr[0] + " = " + expr[1] + "\n"
                py_config.write(line)
        else:
        #String, Int and float declarations
            line = constants[0] + " = "
            if isinstance(constants[1], str):
                line += "\"" + constants[1] + "\""
            else:
                line += str(constants[1])
            line += "\n"
            py_config.write(line)

for constants in data["c"].items():
    #Expression handler
    if constants[0] == "expression":
        for expression in constants[1].items():
            line = "#define " + expression[0] + " " + expression[1] + "\n"
            c_config.write(line)
    else:
        #String, int and float handler
        line = "#define " + constants[0] + " "
        if isinstance(constants[1], str):
            line += "\"" + constants[1] + "\""
        else:
            line += str(constants[1])
        line += "\n"
        c_config.write(line)


#Close file descriptors
c_config.close()
py_config.close()
cfg.close()