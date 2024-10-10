# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.patch.monkey_patch import patch_all; patch_all()
print("patching startup.py")
from .entrypoint import run

if __name__ == '__main__':
    run()
