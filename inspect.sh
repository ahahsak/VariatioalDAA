#!/bin/bash

pylint --disable=C,E vbhmm || true
pylint -E vbhmm
nosetests --with-coverage --cover-package=vbhmm
