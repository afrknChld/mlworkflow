from itertools import takewhile
from datetime import datetime
import os
import re


def _format_filename(filename):
    datetime_ = datetime.now()
    date = datetime_.strftime("%Y%m%d")
    time = datetime_.strftime("%H%M%S")
    datetime_ = "{}_{}".format(date, time)
    return filename.format(datetime_, datetime=datetime_, date=date, time=time)


class PatternFactory:
    """Transforms strings composed of * and ** to matches on strings with
    particular separator"""
    def __init__(self, separator):
        separator = re.escape(separator)
        self.substitutions = {"*":  r"[^{}]*".format(separator),
                              "**": r".*"}

    def _substitute(self, match):
        match = match.group()
        return self.substitutions.get(match, match)

    def create_regex(self, pattern):
        pattern = pattern.replace(".", r"\.")
        pattern = re.sub(r"\*+", self._substitute, pattern)
        return re.compile("^{}$".format(pattern))

_factory = PatternFactory("/")


def find_files(filename, base_dir=""):
    """Given a filename/dirname/pattern, returns matching files on the system"""
    if isinstance(filename, list):
        files = [find_files(file, base_dir=base_dir) for file in filename]
        files = sorted(set(sum(files, [])))
        return files
    if os.path.isdir(filename):
        filename = filename+"/**"
    sections = re.split(r"/+", filename)

    root = tuple(takewhile(lambda section: "*" not in section, sections[:-1]))
    pattern = sections[len(root):]
    recursive = len(pattern) > 1 or any("**" in p for p in pattern)

    pattern = "/".join(pattern)
    root = os.sep.join(root)
    _root = (root if root else ".") + os.sep
    if base_dir:
        _root = os.path.join(base_dir, _root)
    pattern = _factory.create_regex(pattern)
    
    lst = []
    prefix_length = len(_root)
    for dirpath, _dirnames, filenames in os.walk(_root, topdown=True,
                                                 followlinks=True):
        dirpath = dirpath[prefix_length:]
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            complete_path = os.path.join(root, path)
            matchable_path = path.replace(os.sep, "/")
            match = pattern.match(matchable_path)
            if match is not None:
                lst.append(complete_path)
        if not recursive:
            break
    lst.sort()
    return lst
