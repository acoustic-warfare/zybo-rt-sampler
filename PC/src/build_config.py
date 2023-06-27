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
    if constants[0] == "ctypes":
        py_config.write("import ctypes\n")
        for ctype_constants in constants[1].items():
            line = ctype_constants[0] + " = " + ctype_constants[1] + "\n"
            py_config.write(line)
    else:
        line = constants[0] + " = "
        if isinstance(constants[1], str):
            line += "\"" + constants[1] + "\""
        else:
            line += str(constants[1])
        line += "\n"
        py_config.write(line)

for constants in data["c"].items():
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