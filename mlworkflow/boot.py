"""
Directory mostly made for running __main__ modules not as __main__
"""
from mlworkflow import boot
import os
import sys


if __name__ == '__main__':
    import importlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--module', default="run")
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args, rest = parser.parse_known_args()
    sys.argv = [args.module, *rest, *args.command]

    boot.main = args.module

    # sys.path.insert(0, os.getcwd())  # Already added
    importlib.import_module(boot.main)
else:
    main = "__main__"
    def backup(dirname, files='*'):
        from mlworkflow.versioning import TimeCapsule
        return TimeCapsule(os.getcwd(), dirname, files=files).build()


    def rm_backup(target):
        from mlworkflow.file_handling import find_files
        files = set(find_files("**", base_dir=target))

        dirs = set()
        for file in files:
            dirs.add(os.path.dirname(file))

        for file in files:
            os.remove(os.path.join(target, file))
        for dir_ in sorted(dirs, reverse=True):
            os.rmdir(os.path.join(target, dir_))
