rm -rf build
rm -rf coviar.egg-info
rm -rf dist
pip uninstall coviar -y
python setup.py build_ext
python setup.py install
