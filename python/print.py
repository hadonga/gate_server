import pathlib


file_name=pathlib.Path("sss.onnx").stem

print(file_name)

import pdb; pdb.set_trace()
if file_name.find("simp") == -1:
    pass