A full model is composed of 3 types of entities:

1. The model
2. The variations
3. The variation versions

Let's take the example of [efficientnet](https://www.kaggle.com/models/tensorflow/efficientnet) to explain these entities.

A model like `efficientnet` contains multiple variations.

A variation is a specific variation of the model (e.g. B0, B1, ...) with a certain framework (e.g. TensorFlow2).

## Model

To create a model, a special `model-metadata.json` file must be specified. 

Here's a basic example for `model-metadata.json`:
```
{
  "ownerSlug": "INSERT_OWNER_SLUG_HERE",
  "title": "INSERT_TITLE_HERE",
  "slug": "INSERT_SLUG_HERE",
  "subtitle": "",
  "isPrivate": true,
  "description": "Model Card Markdown, see below",
  "publishTime": "",
  "provenanceSources": ""
}
```

You can also use the API command `kaggle models init -p /path/to/model` to have the API create this file for you for a new model. If you wish to get the metadata for an existing model, you can use `kaggle models get username/model-slug`.

### Contents

We currently support the following metadata fields for models.

* `ownerSlug`: the slug of the user or organization
* `title`: the model's title
* `slug`: the model's slug (unique per owner)
* `licenseName`: the name of the license (see the list below)
* `subtitle`: the model's subtitle
* `isPrivate`: whether or not the model should be private (only visible by the owners). If not specified, will be `true`
* `description`: the model's card in markdown syntax (see the template below)
* `publishTime`: the original publishing time of the model
* `provenanceSources`: the provenance of the model

## Model Variation

To create a model variation, a special `model-instance-metadata.json` file must be specified. 

Here's a basic example for `model-instance-metadata.json`:
```
{
  "ownerSlug": "INSERT_OWNER_SLUG_HERE",
  "modelSlug": "INSERT_EXISTING_MODEL_SLUG_HERE",
  "instanceSlug": "INSERT_INSTANCE_SLUG_HERE",
  "framework": "INSERT_FRAMEWORK_HERE",
  "overview": "",
  "usage": "Usage Markdown, see below",
  "licenseName": "Apache 2.0",
  "fineTunable": False,
  "trainingData": [],
  "modelInstanceType": "Unspecified",
  "baseModelInstance": "",
  "externalBaseModelUrl": ""
}
```

You can also use the API command `kaggle models variations init -p /path/to/model-variation` to have the API create this file for you for a new model variation.

### Contents

We currently support the following metadata fields for model variations.

* `ownerSlug`: the slug of the user or organization of the model
* `modelSlug`: the existing model's slug
* `instanceSlug`: the slug of the variation
* `framework`: the variation's framework (possible options: `tensorFlow1`,`tensorFlow2`,`tfLite`,`tfJs`,`pyTorch`,`jax`,`coral`, ...)
* `overview`: a short overview of the variation
* `usage`: the variation's usage in markdown syntax (see the template below)
* `fineTunable`: whether the variation is fine tunable
* `trainingData`: a list of training data in the form of strings, URLs, Kaggle Datasets, etc...
* `modelInstanceType`: whether the model variation is a base model, external variant, internal variant, or unspecified
* `baseModelInstance`: if this is an internal variant, the `{owner-slug}/{model-slug}/{framework}/{variation-slug}` of the base model variation
* `externalBaseModelUrl`: if this is an external variant, a URL to the base model

### Licenses

Here is a list of the available licenses for models:

- Apache 2.0
- Attribution 3.0 IGO (CC BY 3.0 IGO)
- Attribution 3.0 Unported (CC BY 3.0)
- Attribution 4.0 International (CC BY 4.0)
- Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)
- Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
- Attribution-NonCommercial-ShareAlike 3.0 IGO (CC BY-NC-SA 3.0 IGO)
- BSD-3-Clause
- CC BY-NC-SA 4.0
- CC BY-SA 3.0
- CC BY-SA 4.0
- CC0: Public Domain
- Community Data License Agreement - Permissive - Version 1.0
- Community Data License Agreement - Sharing - Version 1.0
- GNU Affero General Public License 3.0
- GNU Free Documentation License 1.3
- GNU Lesser General Public License 3.0
- GPL 2
- MIT
- ODC Attribution License (ODC-By)
- ODC Public Domain Dedication and Licence (PDDL)
- GPL 3

### Usage

The following template variables can be used in this markdown: 

- `${VERSION_NUMBER}` is replaced by the version number when rendered
- `${VARIATION_SLUG}` is replaced by the variation slug when rendered
- `${FRAMEWORK}` is replaced by the framework name
- `${PATH}` is replaced by `/kaggle/input/<model_slug>/<framework>/<variation_slug>/<version>`.
- `${FILEPATH}` is replaced by `/kaggle/input/<model_slug>/<framework>/<variation_slug>/<version>/<filename>`. This value is only defined if the databundle contain a single file
- `${URL}` is replaced by the absolute URL of the model