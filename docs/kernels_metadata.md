To upload and run a kernel, a special `kernel-metadata.json` file must be specified. 

Here's a basic example for `kernel-metadata.json`:
```
{
  "id": "timoboz/my-awesome-kernel",
  "id_no": 12345,
  "title": "My Awesome Kernel",
  "code_file": "my-awesome-kernel.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "false",
  "enable_gpu": "false",
  "enable_internet": "false",
  "machine_shape": "",
  "dataset_sources": ["timoboz/my-awesome-dataset"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```
You can also use the API command `kaggle kernels init -p /path/to/kernel` to have the API create this file for you for a new kernel. If you wish to get the metadata for an existing kernel, you can use `kaggle kernels pull -p /path/to/download -k username/kernel-slug -m`.

## Contents
We currently support the following metadata fields for kernels.
* `id`: The URL slug of your kernel. One of `id` or `id_no` must be specified. If both are, `id_no` will be preferred.
  1. Your username slug
  2. A unique kernel slug
* `id_no`: The kernel's numeric ID.  One of `id` or `id_no` must be specified. If both are, `id_no` will be preferred.
* `title`: The title of the kernel. Required for new kernels - optional for existing ones. Please be aware that kernel titles and slugs are linked to each other. A kernel slug is always the title lowercased with dashes (`-`) replacing spaces. 
  * If you wish to rename your kernel, you may change the title within the metadata. However, you will need to update the `id` as well AFTER the rename is complete.
* `code_file`: The path to your kernel source code. Required. If not an absolute path, it should be relative to the location of `kernel-metadata.json`.
* `language`: The language your kernel is written in. Valid options are `python`, `r`, and `rmarkdown`. Required.
* `kernel_type`: The type of kernel. Valid options are `script` and `notebook`. Required.
* `is_private`: Whether or not the kernel should be private. If not specified, will be `true`.
* `enable_gpu`: Whether or not the kernel should run on a GPU. If not specified, will be `false`.
* `enable_internet`: Whether or not the kernel should be able to access the internet. If not specified, will be `false`.
* `machine_shape`: The accelerator/GPU type to use (e.g., `NvidiaTeslaT4`, `NvidiaTeslaP100`, or `Tpu1VmV38`).
* `dataset_sources`: A list of dataset sources, specified as `"username/dataset-slug"`
* `competition_sources`: A list of competition sources, specified as `"competition-slug"`
* `kernel_sources`: A list of kernel sources, specified as `"username/kernel-slug"`
* `model_sources`: A list of model sources, specified as `"username/model-slug/framework/variation-slug/version-number"`

We will add further metadata processing in upcoming versions of the API.