rm -rf dist/*
rm -rf build/*
.venv/bin/python3 setup.py test
.venv/bin/python3 setup.py sdist bdist_wheel
twine upload dist/*