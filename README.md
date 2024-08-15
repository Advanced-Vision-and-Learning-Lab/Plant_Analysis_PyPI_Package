# Plant Phenotyping Analysis

This package provides a GUI tool with ML powered backend to analyse plant images for phenotyping.

Check out the [documentation](https://plant-analysis-avll.readthedocs.io/en/latest/) to get started

## Build and Upload

* After updating src directory with latest code, update the version number in pyprojects and run 'python3 -m build'. After building, you will see built files are added to a new folder 'dist'
* Install twine if not installed by 'python3 -m pip install --upgrade twine'
* To upload, run 'python3 -m twine upload --repository testpypi dist/*'
* Edit the build files with appropriate user details. The username and password of the GitHub user need to be entered when uploading.
