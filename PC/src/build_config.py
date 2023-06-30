import json

"Create and open a config files"
c_config = open('./src/config.h', "w")
py_config = open('./config.py', "w")

#Open config JSON file
cfg = open('./src/config.json')
data = json.load(cfg)

py_data = "#Do not edit this config file! Add constants and expressions in config.json and build with make. \n"
c_data = "//Do not edit this config file! Add constants and expressions in config.json and build with make. \n"

py_data += "\n#General constants for both c and python.\n"
c_data += "\n//General constants for both c and python.\n"
for constants in data["general"].items():
    if constants[0] == "expression":
        for expression in constants[1].items():
            c_data += "#define " + expression[0] + " " + expression[1] + "\n"
            py_data += expression[0] + " = " + expression[1] + "\n"

    else: 
        #For c header config
        c_data += "#define " + constants[0] + " "
        if isinstance(constants[1], str):
            c_data += "\"" + constants[1] + "\""
        else:
            c_data += str(constants[1])
        c_data += "\n"

        #For python config
        py_data += constants[0] + " = "
        if isinstance(constants[1], str):
            py_data += "\"" + constants[1] + "\""
        else:
            py_data += str(constants[1])
        py_data += "\n"

py_data += "\n#Python specific constants\n"
for constants in data["python"].items():
    #Imports
    if constants[0] == "imports":
        for lib in constants[1]:
            py_data = "import " + lib + "\n" + py_data
    else:
    #Expression handler
        if constants[0] == "expression":
            for expr in constants[1].items():
                py_data += expr[0] + " = " + expr[1] + "\n"
        else:
        #String, Int and float declarations
            py_data += constants[0] + " = "
            if isinstance(constants[1], str):
                py_data += "\"" + constants[1] + "\""
            else:
                py_data += str(constants[1])
            py_data += "\n"

c_data += "\n//C specific constants\n"
for constants in data["c"].items():
    #Expression handler
    if constants[0] == "expression":
        for expression in constants[1].items():
            c_data += "#define " + expression[0] + " " + expression[1] + "\n"
    else:
        #String, int and float handler
        c_data += "#define " + constants[0] + " "
        if isinstance(constants[1], str):
            c_data += "\"" + constants[1] + "\""
        else:
            c_data += str(constants[1])
        c_data += "\n"

#Write to config.h and config.h
c_config.write(c_data)
py_config.write(py_data)

#Close file descriptors
c_config.close()
py_config.close()
cfg.close()