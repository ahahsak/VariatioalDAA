#!/bin/bash

pylint --disable=C,E vardaa || true
pylint -E vardaa
pylint -E tests
nosetests --with-coverage --cover-package=vardaa --exe
