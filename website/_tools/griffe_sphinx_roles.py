"""Griffe extension for the Python API pages: Sphinx roles in docstrings as inline code.

Some docstrings of the jetscape package use Sphinx's cross-reference roles
(``:meth:`jet_minus_background```, ``:class:`HadronFileReader```, ...). The API pages are
Markdown, where such a role shows up literally. This extension rewrites each role to inline
code while the docstrings are loaded, for the site only: the source keeps its roles, so
``help()`` and Sphinx-aware tools are unchanged. A leading ``~`` (Sphinx's "last part only")
keeps only the last dotted part, as Sphinx would show it.
"""
import re

import griffe

_ROLE = re.compile(r":(?:py:)?(?:meth|class|func|attr|mod|data|obj|exc|const):`(~?)([^`]+)`")


def _code(match):
    tilde, target = match.groups()
    return f"`{target.rsplit('.', 1)[-1] if tilde else target}`"


class SphinxRoles(griffe.Extension):
    def on_object(self, *, obj, **kwargs):
        if obj.docstring is not None and ":`" in obj.docstring.value:
            obj.docstring.value = _ROLE.sub(_code, obj.docstring.value)
