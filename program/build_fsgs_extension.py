from __future__ import annotations

import runpy

import torch.utils.cpp_extension as cpp_extension


# PyTorch 2.11's Windows helper decodes cl.exe output as OEM.  This machine's
# OEM codec is not usable for that output, while the compiler itself is fine.
cpp_extension.SUBPROCESS_DECODE_ARGS = ()
runpy.run_path("setup.py", run_name="__main__")
