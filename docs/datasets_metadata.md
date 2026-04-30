The Kaggle API follows the [Data Package specification](https://frictionlessdata.io/specs/data-package/) for specifying metadata when creating new Datasets and Dataset versions. Next to your files, you have to put a special `dataset-metadata.json` file in your upload folder alongside the files for each new Dataset (version). 

Here's a basic example for `dataset-metadata.json`:
```
{
  "title": "My Awesome Dataset", 
  "id": "timoboz/my-awesome-dataset", 
  "licenses": [{"name": "CC0-1.0"}]
}
```
You can also use the API command `kaggle datasets init -p /path/to/dataset` to have the API create this file for you.

Here's an example containing file metadata:
```
{
  "title": "My Awesome Dataset", 
  "subtitle": "My awesomer subtitle",
  "description": "My awesomest description",
  "id": "timoboz/my-awesome-dataset", 
  "id_no": 12345,
  "licenses": [{"name": "CC0-1.0"}],
  "resources": [
    {
      "path": "my-awesome-data.csv",
      "description": "This is my awesome data!",
      "schema": {
        "fields": [
          {
            "name": "StringField",
            "description": "String field description",
            "type": "string"
          },
          {
            "name": "NumberField",
            "description": "Number field description",
            "type": "number"
          },
          {
            "name": "DateTimeField",
            "description": "Date time field description",
            "type": "datetime"
          }
        ]
      }
    },
    {
      "path": "my-awesome-extra-file.txt",
      "description": "This is my awesome extra file!"
    }
  ],
  "keywords": [
    "beginner",
    "tutorial"
  ],
  "expectedUpdateFrequency": "monthly",
  "userSpecifiedSources": "World Bank and OECD ([link](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD))",
  "image": "relative/path/to/new/image.png"
}
```

## Contents
The following metadata is currently supported:
* `kaggle datasets create` (create a new Dataset):
  * `title`: Title of the dataset, must be between 6 and 50 characters in length.
  * `subtitle`: Subtitle of the dataset, must be between 20 and 80 characters in length.
  * `description`: Description of the dataset. 
  * `id`: The URL slug of your new dataset, a combination of:
    1. Your username or organization slug (if you are a member of an organization).
    2. A unique Dataset slug, must be between 3 and 50 characters in length.
  * `licenses`: Must have exactly one entry that specifies the license. Only `name` is evaluated, all other information is ignored. See below for options.
  * `resources`: Contains an array of files that are being uploaded.  (Note - this is not required, nor if included, does it need to include all of the files to be uploaded.):
    * `path`: File path.
    * `description`: File description.
    * `schema`: File schema (definition below):
      * `fields`: Array of fields in the dataset.  Please note that this needs to include ALL of the fields in the data in order or they will not be matched up correctly.  A later version of the API will fix this bug.
        * `name`: Field name
        * `title`: Field description
        * `type`: Field type. A best-effort list of types will be kept at the bottom of this page, but new types may be added that are not documented here.
  * `keywords`: Contains an array of strings that correspond to an existing tag on Kaggle.  If a specified tag doesn't exist, the upload will continue, but that specific tag won't be added.  
* `kaggle datasets version` (create a new version for an existing Dataset):
  * `subtitle`: Subtitle of the dataset, must be between 20 and 80 characters in length.
  * `description`: Description of the dataset. 
  * `id`: The URL slug of the dataset you want to update (see above). You must be the owner or otherwise have edit rights for this dataset. One of `id` or `id_no` must be specified. If both are, `id_no` will be preferred.
  * `id_no`: The ID of the dataset. One of `id` or `id_no` must be specified. You must be the owner or otherwise have edit rights for this dataset. If both are, `id_no` will be preferred.
  * `resources`: Contains an array of files that are being uploaded.  (Note - this is not required, nor if included, does it need to include all of the files to be uploaded.):
    * `path`: File path.
    * `description`: File description.
    * `schema`: File schema (definition below):
      * `fields`: Array of fields in the dataset.  Please note that this needs to include ALL of the fields in the data in order or they will not be matched up correctly.  A later version of the API will fix this bug.
        * `name`: Field name
        * `title`: Field description
        * `type`: Field type. A best-effort list of types will be kept at the bottom of this page, but new types may be added that are not documented here.
  * `keywords`: Contains an array of strings that correspond to an existing tag on Kaggle.  If a specified tag doesn't exist, the upload will continue, but that specific tag won't be added.  
* `kaggle datasets metadata --update` (update metadata for an existing Dataset) supports all fields mentioned above for `kaggle datasets version`, and additionally:
  * `expectedUpdateFrequency`: How often you expect to update your dataset with new versions. See [section below](#expected-update-frequencies) for possible values.
  * `userSpecifiedSources`: An explanation of the source(s) of your dataset. Most basic markdown features are supported for this string.
  * `image`: A relative file path to a new image file you want to use for your dataset. The path should be relative to the location of the dataset-metadata.json file. See [section below](#images) for more specifics about file types and expected image size.

We will add further metadata processing in upcoming versions of the API.

## Licenses
You can specify the following licenses for your datasets:
* `CC0-1.0`: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
* `CC-BY-SA-3.0`: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
* `CC-BY-SA-4.0`: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
* `CC-BY-NC-SA-4.0`: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
* `GPL-2.0`: [GPL 2](http://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
* `ODbL-1.0`: Database: [Open Database](http://opendatacommons.org/licenses/odbl/1.0/), Contents: © Original Authors
* `DbCL-1.0`: Database: [Open Database](http://opendatacommons.org/licenses/odbl/1.0/), Contents: [Database Contents](http://opendatacommons.org/licenses/dbcl/1.0/)
* `copyright-authors`: Data files © Original Authors
* `other`: Other (specified in description)
* `unknown`: Unknown
* `CC-BY-4.0`: 	
https://creativecommons.org/licenses/by/4.0/
* `CC-BY-NC-4.0`: https://creativecommons.org/licenses/by-nc/4.0/
* `PDDL`: https://opendatacommons.org/licenses/pddl/1.0/
* `CC-BY-3.0`: 	
https://creativecommons.org/licenses/by/3.0/
* `CC-BY-3.0-IGO`: 	
https://creativecommons.org/licenses/by/3.0/igo/
* `US-Government-Works`: 	
https://www.usa.gov/government-works/
* `CC-BY-NC-SA-3.0-IGO`: 	
https://creativecommons.org/licenses/by-nc-sa/3.0/igo/
* `CDLA-Permissive-1.0`: 	
https://cdla.io/permissive-1-0/
* `CDLA-Sharing-1.0`: 	
https://cdla.io/sharing-1-0/
* `CC-BY-ND-4.0`: 	
https://creativecommons.org/licenses/by-nd/4.0/
* `CC-BY-NC-ND-4.0`: 	
https://creativecommons.org/licenses/by-nc-nd/4.0/
* `ODC-BY-1.0`: 	
https://opendatacommons.org/licenses/by/1-0/index.html
* `LGPL-3.0`: 	
http://www.gnu.org/licenses/lgpl-3.0.html
* `AGPL-3.0`: 	
http://www.gnu.org/licenses/agpl-3.0.html
* `FDL-1.3`: 	
http://www.gnu.org/licenses/fdl-1.3.html
* `EU-ODP-Legal-Notice`: https://ec.europa.eu/info/legal-notice_en
* `apache-2.0`: 	
https://www.apache.org/licenses/LICENSE-2.0
* `GPL-3.0`: [GPL 2](https://www.gnu.org/licenses/gpl-3.0.html)

## Data types
You can specify the following data types
* `string`
* `boolean`
* `numeric`
* `datetime`
* `id`
* `uuid`
* `latitude`
* `longitude`
* `coordinates`
* `country`
* `province` (these are states in the US)
* `postalcode`
* `address`
* `email`
* `url`
* `integer`
* `decimal`
* `city`

## Expected update frequencies
You can specify the following values for `expectedUpdateFrequency`:
* `not specified`
* `never`
* `annually`
* `quarterly`
* `monthly`
* `weekly`
* `daily`
* `hourly`

## Images
The recommended way to update your dataset's image is by placing a file named `dataset-cover-image.png` (or `.jpg`, `.jpeg`, `.webp`), as a sibling file to your `datasets-metadata.json`.

Example:
- `/some/path/dataset-metadata.json`
- `/some/path/dataset-cover-image.png`

The image file will only be used for dataset metadata, and not be uploaded as a file within your dataset.

### Specifying an image with a relative path
As an alternative, you can update your dataset image by providing a relative path from your `datasets-metadata.json` to an image file, using the `image` property.

If your files were located at:
- `/some/path/dataset-metadata.json`
- `/some/path/to/my-image.jpg`

This property should be specified as:
```
"image": "to/my-image.jpg"
```

### Supported image file types and expected dimensions

The following file types are supported:

* `.png`
* `.jpg`
* `.jpeg`
* `.webp`

The image needs to have a minimum width of 560px and a minimum height of 280px.

The same image file will be used for two different crops:

- Header, 2:1 ratio
  - Crop rectangle: width: 560px, height: 280px, top: 0, left: 0
  - For an image with dimensions 560px x 280px, this will be the entire rectangular image.
- Thumbnail, 1:1 ratio
  - Crop rectangle: width: 280px, height: 280px, top: 0, left: 140px
  - For an image with dimensions 560px x 280px, this will be a centered 280px square.

While you can upload a larger image than 560px x 280px, the crops as specified above will be applied, and this may not look good. These crops can always be edited in the UI on kaggle.com on the settings page for your dataset.