#!/bin/sh

command -v coverage >/dev/null && coverage erase
command -v python-coverage >/dev/null && python-coverage erase

rm tests/images/*.png
nosetests --with-coverage --cover-erase --cover-package=bamboo --cover-html
