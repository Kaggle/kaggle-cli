Changelog
====

### 1.8.4

* Rename `kaggle-api` to `kaggle-cli`
* Allow auth to happen multiple times (#922)
* Add --acc to set accelerator for: kaggle kernels push ... (#907)
* Add automatic retry and resume to download_file (#905) Thanks katoue!
* Restore model validation check (#902)
* Add file pattern matching in output download (#901) Thanks piotr-ginal!

### 1.8.3

* Add packaging dep (#883)
* Add version checking against server known-version (#880)
* Fix edit error (#876)
* Use kagglesdk from pypi (#875)
* Fix Kaggle access token auth KeyError when KAGGLE_API_TOKEN is unset (#874)

### 1.8.2

* Changes to build script

### 1.8.1

* Fix memory exhaustion when downloading large files (#869)
* Add python-dateutil to pyproject.toml dependencies (#866)

### 1.8.0

* Fix resumable download error (#865)
* Fix dataset version spec (#862)
* Add machine_shape to the metadata of kaggle kernels pull (#856)
* Add pagination options to models
* Add pagination options for submissions (#832)
* Add pagination options to list commands (#815)
* Add canonical aliases for push/pull (#787)
* Add parquet as a filter option (#786)
* Add variations as alt for instances (#784)
* Enable (and rename) synonyms i and v (#782)

### 1.7.5.0 (not released)

* Require Python 3.11.
* Add KernelExecutionType (#775)
* Output docker_image as part of the pull metadata (#773)
* Allow user to specify docker_image during kernel push (#774)
* Add kernel version type to save request (#771)
* Add tests for delete and de-flake (#769)
* Rename "yes" params and make confirmation consistent (#765)
* Fix bug that caused double serialization (#764)
* Add kaggle kernels delete (#762)
* Add test for dataset_delete() and make script more robust (#760)
* Check dataset status before uploading (#759)
* Add kaggle datasets delete (#755)
* Fix calls to download_file() (#752)
* Add type annotations for mypy (#746)
* Use Optional[...] in cases where the proto file does (#744)
* Improve some type hints and fix a bug (#741)
* Reformat everything with black (#737)
* Add more type hints (#736)
* Add type annotations to main file (#735)
* Bulk reformat docstrings (#732)
* Merge envars before sending a request (#729)
* Use PROD if no environment is specified. (#726)
* Add a no response action to auto-close issues (#723)

### 1.7.4.2

* Fix a problem in downloading kernel output files.

### 1.7.4.1

* Fix a dataset download problem. Datasets that had a license were failing to download.
* Update the documentation to include code competition submit.

### 1.7.4

Version 1.7.3 was never released. There were errors in versioning on
test.pypi.org. For consistency, we decided to jump several version numbers.
This is the first release since 1.6.17.

The actual changes are described in 1.7.3.

### 1.7.3

There was an error in versioning. We went from 1.6.17 to 1.7.3.

* Added the ability to submit to a code competition. Some required arguments have been made optional.
* Added a `--timeout` option to `kaggle kernels push` to limit the run-time to the specified number of seconds.
* Removed Swagger. Projects that use `kaggle/api/kaggle_api.py` may be affected. That file is deprecated and will be
  removed. Most of its functions still work, but those that involve uploading files no longer work.
  The command-line tool uses a higher-level abstraction for uploading, and client code needs
  to be converted to use that.

### 1.7.3b2

* Added the ability to submit to a code competition. Some required arguments have been made optional.
* Added a `--timeout` option to `kaggle kernels push` to limit the run-time to the specified number of seconds.

### 1.7.3b1

* Fix escaped-quote issue in HTTP requests.

### 1.7.3b0

* Remove Swagger. No user-visible changes to the command-line tool. However, projects that
use `kaggle/api/kaggle_api.py` may be affected. That file is deprecated and will be removed.
Most of its functions still work, but those that involve uploading files no longer work.
The command-line tool uses a higher-level abstraction for uploading and client code needs 
to be converted to use that.

### 1.6.17

* No changes; release 1.6.16 did not complete.

### 1.6.16

* No changes; release 1.6.15 isn't usable. We're working on process updates to prevent this from happening again.

### 1.6.15
* Support XDG base directory specification on Linux
* Disable out-of-date API version warning with -W
* Allow an array of strings in "source" when uploading .ipynb files (thanks to GitHub user mgallifrey for the contribution!)
* Add triton framework for models
* Update model licenses

### 1.6.14

* No changes; release 1.6.13 isn't usable.

### 1.6.13

* Add --page-size and --page-token CLI options to all commands that display lists of files.

### 1.6.12

* Re-release 1.6.11 without the `src` directory included in the package.

### 1.6.11

* Allow unauthenticated usage of "datasets download", "datasets files".
  * This will only work after April 8th, 2024. More more details, see:
    <https://www.kaggle.com/discussions/product-feedback/485439>
* Allow "help" and "version" to be used for all commands, unauthenticated.
* Fix: "dataset download -f" can accept a specific dataset version.

### 1.6.10

Repackage of 1.6.8 as a new release, to fix the problematic 1.6.9 release.

### 1.6.9

* Do not use. Problematic release that causes an error:
  `ModuleNotFoundError: No module named 'kaggle.api'`

### 1.6.8

* Add "gguf"

### 1.6.7

* Add "TensorRtLlm" model framework.

### 1.6.6

* Add "GemmaCpp" and "GGML" model frameworks.

### 1.6.5

* Add "MaxText" model framework.

### 1.6.4

* Add "Transformers" model framework.

### 1.6.3

Release date: 01/11/24
* Add "Flax" and "Pax" model frameworks.

### 1.6.2

Release date: 01/09/24
* Add "Other" model framework.

### 1.6.1
Release date: 01/08/24
* Fix dataset/model upload.

### 1.6.0
Release date: 01/04/24
* Release the pre-release branch with models endpoints.

#### 1.6.0a7
Release date: 11/22/23
* Add model_instance_type and base_model_instance_id to ModelInstance

#### 1.6.0a6
Release date: 9/19/23
* Include version_number and version_id in the model-instance-metadata.json file

#### 1.6.0a5
Release date: 8/02/23
* Add Keras model framework.

#### 1.5.16
Release date: 7/17/23
* Fix dataset download bug with locale
* Resumable uploads
* Retry some failed requests

#### 1.6.0a4
Release date: 7/07/23
* Resumable uploads
* Retry some failed requests
* Flag `-y` to delete model/instance/version without confirmation

#### 1.6.0a3
Release date: 7/06/23
* Confirmation for deleting a model, instance or version
* Merge changes from 1.5.14 and 1.5.15

#### 1.5.15
Release date: 6/30/23
* Add missing licenses for datasets
* Re-add option to pass dataset with `-d`
* Download / list files for a specific version of a dataset
* Documentation improvements

#### 1.5.14
Release date: 6/29/23
* Show the full error message from the API
* Improve and fix documentation
* Fix kernel's data sources bug, and add the model data source to push/pull
* Implement resumable downloads
* Fix unreachable code bug
* Make some arguments required
* Add enable_tpu to kernel's push/pull

#### 1.6.0a2
Release date: 6/12/23
* Add endpoint to get a modelInstance
* Simplify the modelInstanceVersion creation
* Fix Model files zipping

#### 1.6.0a0
Release date: 6/07/23
* Add Models endpoints

#### 1.5.13
Release date: 2/27/23
* Add ability to add a model to a kernel

### 1.5.12
Release date: 03/12/21
* No changes

### 1.5.11
Release date: 03/12/21
* Add support for non-ASCII characters for kernels.

### 1.5.10
Release date: 11/30/20
* Remove dependency on slugify.

### 1.5.9
Release date: 10/21/20
* Drop version restriction on urllib3 in setup.py.

### 1.5.8
Release date: 09/03/20
* No user-facing changes

#### 1.5.7
Release date: 8/31/20
* Add ability to specify the kernel docker image pinning type
* Kernels have internet enabled by default
* Various competitions fixes

#### 1.5.6
Release date: 9/19/19
* Downloading all files for a competition downloads a zip instead of individual files

#### 1.5.5
Release date: 8/30/19
* Add vote count and usability rating to datasets listing
* Add min and max dataset size filters to datasets listing
* Add additional information to dataset metadata API
* Allow updating dataset metdata

#### 1.5.4
Release date: 5/28/19
* Make kernels init more friendly
* Make directories if needed for kernels output

#### 1.5.3
Release date: 2/20/19
* Bump urllib3 version

#### 1.5.2
Release date: 1/28/19
* Don't error on encoding errors when printing tables
* Exit with error code when an exception is caught

#### 1.5.1.1
Release date: 12/5/18
* Add missing cli option for dataset subfolders

#### 1.5.1
Release date: 12/5/18
* Allow custom ca_cert files
* Support uplodaing datasets with subfolders
* Fix kaggle.json permissions warning

#### 1.5.0
Release date: 10/19/18
* Update API to work with new competitions submissions backend.  This change will force old API clients to update.
* Update error message when config file is not found.

#### 1.4.7.1
Release date: 8/28/18
* Fix host

#### 1.4.7
Release date: 8/28/18
* Make dataset version `-p` argument actually optinal
* Don't require the `resources` field when updating a dataset
* Don't automatically unzip datasets
* Add an unzip option for dataset downloads
* Add validation for kernel title and slug length
* Give a warning if kernel title does not resolve to the specified slug
* Show kernel version number after pushing
* Respect `code_file` value in kernel metadata when pulling kernels

#### 1.4.6
Release date: 8/7/18
* Allow setting config values through environmental variables

#### 1.4.5
Release date: 8/1/18
* Add error if dataset metadata repeats files

#### 1.4.4
Release date: 7/30/18
* Fix issue with reading kernel metadata

#### 1.4.3
Release date: 7/30/18
* Add more competitions list options
* Add more datasets list options
* Add a couple more fields to kernels list display
* Add support for kernel and dataset ID's
* Allow generating metadata for an existing dataset
* Fix issue with downloading from datasets whose titles don't match their slugs
* Use kernel slug as filename for kernel output
* Make upload and download directory default to current working directory
* Use a default username on downloading kernel or dataset data if none is specified
* Support extended data types on datasets
* Stop requiring `-c`, `-d`, and `-k` arguments
* Don't require `resources` field in dataset metadata

#### 1.4.2
Release date: 7/20/18
* Validate dataset slug and title length before uploading
* Fix issue with dataset metadata file detection
* Cleaned up KeyboardInterrupt errors
* Validate all specified files in a dataset exist prior to uploading
* Make ApiExceptions (slightly) less ugly

#### 1.4.1
Release date: 7/20/18
* Add python 3.7 compatibility

#### 1.4.0
Release date: 7/19/18
* Add kernels support
** List and search kernels
** Push kernels code
** Pull kernels code
** Download kernel output
** Get latest kernel run status

#### 1.3.12
Release date: 6/25/18
* Allow setting a `'KAGGLE_CONFIG_DIR'` environmental token
* Return metadata file after creating
* Alert users that dataset creation takes time

#### 1.3.11.1
* Fix other invalid tags check

#### 1.3.11
Release date: 6/12/18
* Improve version check
* Fix invalid tags check

#### 1.3.10
Release date: 6/10/18
* Restrict urllib3's version due to requests dependency problem

#### 1.3.9.1
Release date: 6/9/18
* Fix bug with competitions submissions.

#### 1.3.9
Release date: 6/8/18
* Improve error message for closed competitions
* Remove stacktrace on errors
* Print any invalid tags
* Warn if there are no competition files to download
* Implement resumable uploads
* Add subtitle metadata to dataset uploads
* Add progress bars for uploads and downloads
* Add command for downloading competitions leaderboard
* Add command for viewing the top of the leaderboard

#### 1.3.8
Release date: 5/18/18
* Add option to delete all previous dataset versions

#### 1.3.7
Release date: 5/18/18
* Add aliases for subcommands (ex. `kaggle c` is the same thing as `kaggle competitions`)
* Add version command
* Show full download path for files
* Remove file size limitation from uploads

#### 1.3.6
Release date: 5/7/18
* Give the option to add tags to datasets.
  * Known limitiation - you cannot delete tags through the API.  Those changes must be done through the website.

#### 1.3.5
Release date: 5/4/18
* Fix schema declaration in dataset resources

#### 1.3.4
Release date: 4/30/18
* Rename `columns` to `fields`

#### 1.3.3
Release date: 4/26/18
* Fix UnicodeEncodeError for certain datasets
* Include Swagger yaml and config files

#### 1.3.2.1
Release date: 4/24/18
* Fix bug with column metadata

#### 1.3.2
Release date: 4/24/18
* Give the option to specify a schema for uploaded datasets
* Give the option to set the dataset description during updates

#### 1.3.1
Release date: 4/19/18
* Give the option to set dataset file descriptions
* Give the option to not convert tabular datasets to csv

#### 1.3.0
Release date: 4/18/18

* Give the option to set the dataset description during creation

#### 1.2.1
Release date: 4/17/18

* [Issue #5](https://github.com/Kaggle/kaggle-api/issues/5) -  Reformat code for consistency and to align with [Google's python coding style](https://google.github.io/styleguide/pyguide.html).  Most of the changes are cosmetic, but most cases of `camelCasing` other than class names have been changed to `snake_case`.  This is a breaking change for anyone directly using the python code rather than simply using the command line.
