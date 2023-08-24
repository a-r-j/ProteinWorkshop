#! /usr/bin/env python3
# -*- coding: utf-8; mode: python -*-
""" Substitue any :emojis: in the input document with its UTF-8 code.

- See https://stackoverflow.com/questions/42087466/sphinx-extension-to-use-github-markdown-emoji-in-sphinx
- Requirements: 'emoji' package, from https://github.com/carpedm20/emoji/

I use it with a small Bash script :

```bash
BUILDDIR=_build/html

for i in "$BUILDDIR"/*.html; do
    # Convert :emojis: to UTF-8 in HTML output (from GFM Markdown), see https://stackoverflow.com/questions/42087466/sphinx-extension-to-use-github-markdown-emoji-in-sphinx
    emojize.py "$i" > "$i".new   # new file
    wdiff -3 "$i" "$i".new       # print the difference
    mv -vf "$i".new "$i"         # write back to the first file
done
```

- *Date:* 07/04/2017
- *Author:* Lilian Besson, (C) 2017
- *Licence:* MIT Licence (http://lbesson.mit-license.org)
"""

from __future__ import print_function, division  # Python 2 compatibility if needed

import re

# Install from https://github.com/carpedm20/emoji/
# with pip install emoji
try:
    from emoji import emojize
except ImportError:
    print("Error: package not found, install 'emoji' package with 'pip install emoji'")


def match_to_emoji(m):
    """Call emoji.emojize on m)."""
    return emojize(m.group(), language="alias")


def emojize_all(s):
    """Convert all emojis :aliases: of the string s to emojis in UTF-8."""
    return re.sub(r":([a-z_-]+):", match_to_emoji, s)


def main(path):
    """Handle the file given by its path."""
    with open(path, 'r') as f:
        for line in f.readlines():
            print(emojize_all(line), end='')


if __name__ == '__main__':
    from sys import argv
    for arg in argv[1:]:
        main(arg)

# End of emojize.py
