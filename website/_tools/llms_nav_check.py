"""MkDocs hook: every page of the navigation must be listed for /llms.txt.

The `sections` of the llmstxt plugin in mkdocs.yml name the pages one by one, with a line of
description each. A page added to `nav:` and forgotten there would be missing from llms.txt and
llms-full.txt without a word; this hook warns instead (an error with `mkdocs build --strict`).
"""
import logging

log = logging.getLogger("mkdocs.hooks.llms_nav_check")


def on_nav(nav, config, files):
    plugin = config.plugins.get("llmstxt")
    if plugin is None:
        return nav
    listed = set()
    for entries in plugin.config.sections.values():
        for entry in entries:
            listed.update(entry if isinstance(entry, dict) else [entry])
    for page in nav.pages:
        if page.file.src_uri not in listed:
            log.warning("%s is in the nav but not in the llmstxt plugin's sections "
                        "(mkdocs.yml): it would be missing from llms.txt", page.file.src_uri)
    return nav
