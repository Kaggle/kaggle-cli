#!/usr/bin/python
#
# Copyright 2023 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8

# NOTE: THIS FILE IS OBSOLETE AND WILL BE REMOVED IN 2025H2

import kaggle_api_extended as api
import json
import os
import re  # noqa: F401
import tempfile


class KaggleApi(object):

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = api.KaggleApi()
        self.api_client = api_client

    def competition_download_leaderboard(self, id):  # noqa: E501
        """Download competition leaderboard  # noqa: E501.

        :param str id: Competition name (required)
        :return: Result
        """
        return self.api_client.competition_leaderboard_download(id, None)

    def competition_view_leaderboard(self, id):  # noqa: E501
        """View competition leaderboard  # noqa: E501.

        :param str id: Competition name (required)
        :return: Result
        """
        return self.api_client.competition_leaderboard_view(id)

    def competitions_data_download_file(self, id, file_name):  # noqa: E501
        """Download competition data file  # noqa: E501.

        :param str id: Competition name (required)
        :param str file_name: Competition name (required)
        :return: Result
        """
        return self.api_client.competition_download_file(id, file_name)

    def competitions_data_download_files(self, id):  # noqa: E501
        """Download all competition data files  # noqa: E501.

        :param str id: Competition name (required)
        :return: Result
        """
        return self.api_client.competition_download_files(id)

    def competitions_data_list_files(self, id):  # noqa: E501
        """List competition data files  # noqa: E501.

        :param str id: Competition name (required)
        :return: Result
        """
        return self.api_client.competition_list_files(id)

    def competitions_list(self, group=None, category=None, sort_by=None, page=1, search=None):  # noqa: E501
        """List competitions  # noqa: E501.

        :param str group: Filter competitions by a particular group
        :param str category: Filter competitions by a particular category
        :param str sort_by: Sort the results
        :param int page: Page number
        :param str search: Search terms
        :return: ApiListCompetitionsResponse
        """
        return self.api_client.competitions_list(group, category, sort_by, page, search)

    def competitions_submissions_list(self, id, page=0):  # noqa: E501
        """List competition submissions  # noqa: E501.

        :param str id: Competition name (required)
        :param int page: Page number
        :return: Result
        """
        return self.api_client.competition_submissions(id, page)

    def competitions_submissions_submit(self, blob_file_tokens, submission_description, id):  # noqa: E501
        """Submit to competition  # noqa: E501.

        :param str blob_file_tokens: Token identifying location of uploaded
            submission file (required)
        :param str submission_description: Description of competition
            submission (required)
        :param str id: Competition name (required)
        :return: Result
        """
        print("competitions_submissions_submit() not implemented")
        print("use kaggle_api_extended.KaggleApi.competition_submit() instead")
        raise NotImplementedError()

    def competitions_submissions_upload(self, file, guid, content_length, last_modified_date_utc):  # noqa: E501
        """Upload competition submission file  # noqa: E501.

        :param file file: Competition submission file (required)
        :param str guid: Location where submission should be uploaded
            (required)
        :param int content_length: Content length of file in bytes
            (required)
        :param int last_modified_date_utc: Last modified date of file in
            seconds since epoch in UTC (required)
        :return: Result
        """
        print("competitions_submissions_upload() not implemented")
        print("use kaggle_api_extended.KaggleApi.competitions_submissions_submit() instead")
        raise NotImplementedError()

    def competitions_submissions_url(self, id, content_length, last_modified_date_utc):  # noqa: E501
        """Generate competition submission URL  # noqa: E501.

        :param str id: Competition name, as it appears in the competition's
            URL (required)
        :param int content_length: Content length of file in bytes
            (required)
        :param int last_modified_date_utc: Last modified date of file in
            seconds since epoch in UTC (required)
        :param str file_name: Competition submission file name
        :return: Result
        """
        print("competitions_submissions_url() not implemented")
        print("use kaggle_api_extended.KaggleApi.competition_submit() instead")
        raise NotImplementedError()

    def create_inbox_file(self, create_inbox_file_request):  # noqa: E501
        """Creates (aka \&quot;drops\&quot;) a new file into the inbox.  # noqa:
        E501.

        :param CreateInboxFileRequest create_inbox_file_request: (required)
        :return: Result
        """
        print("create_inbox_file() not implemented")
        print("use kaggle_api_extended.KaggleApi.files_upload_cli() instead")
        raise NotImplementedError()

    def datasets_create_new(self, request):  # noqa: E501
        """Create a new dataset  # noqa: E501.

        :param DatasetNewRequest request: Information for creating a new
            dataset (required)
        :return: Result
        """
        print("datasets_create_new() not implemented")
        print("use kaggle_api_extended.KaggleApi.dataset_create_new() instead")
        raise NotImplementedError()

    def datasets_create_version_by_id(self, id, dataset_new_version_request, **kwargs):  # noqa: E501
        """Create a new dataset version by id  # noqa: E501.

        :param int id: Dataset ID (required)
        :param DatasetNewVersionRequest dataset_new_version_request:
            Information for creating a new dataset version (required)
        :return: throws NotImplementedError
        """
        print("datasets_create_version_by_id() not implemented")
        print("use kaggle_api_extended.KaggleApi.dataset_create_version() instead")
        raise NotImplementedError()

    def datasets_download(self, owner_slug, dataset_slug, dataset_version_number=None):  # noqa: E501
        """Download dataset file  # noqa: E501.

        :param str owner_slug: Dataset owner (required)
        :param str dataset_slug: Dataset name (required)
        :param str dataset_version_number: Dataset version number
        :return: Result
        """
        dataset = f"{owner_slug}/{dataset_slug}"
        if dataset_version_number is not None:
            dataset += f"/{dataset_version_number}"
        return self.api_client.dataset_download_files(dataset)

    def datasets_download_file(self, owner_slug, dataset_slug, file_name, dataset_version_number=None):  # noqa: E501
        """Download dataset file  # noqa: E501.

        :param str owner_slug: Dataset owner (required)
        :param str dataset_slug: Dataset name (required)
        :param str file_name: File name (required)
        :param str dataset_version_number: Dataset version number
        :return: True if the file was downloaded successfully, False otherwise
        """
        dataset = f"{owner_slug}/{dataset_slug}"
        if dataset_version_number is not None:
            dataset += f"@{dataset_version_number}"
        return self.api_client.dataset_download_file(dataset, file_name)

    def datasets_list(
        self,
        sort_by=None,
        size=None,
        file_type=None,
        license_name=None,
        tag_ids=None,
        search=None,
        user=None,
        mine=False,
        page=1,
        max_size=None,
        min_size=None,
    ):  # noqa: E501
        """List datasets  # noqa: E501.

        :param str group: Display datasets by a particular group
        :param str sort_by: Sort the results
        :param str size: (DEPRECATED). Please use --max-size and --min-size
            to filter dataset sizes.
        :param str filetype: Display datasets of a specific file type
        :param str license: Display datasets with a specific license
        :param str tagids: A comma separated list of tags to filter by
        :param str search: Search terms
        :param str user: Display datasets by a specific user or organization
        :param int page: Page number
        :param int max_size: Max Dataset Size (bytes)
        :param int min_size: Max Dataset Size (bytes)
        :return: List of ApiDatasetFile.
        """
        return self.api_client.dataset_list(
            sort_by=sort_by,
            size=size,
            file_type=file_type,
            license_name=license_name,
            tag_ids=tag_ids,
            search=search,
            user=user,
            mine=mine,
            page=page,
            max_size=max_size,
            min_size=min_size,
        )

    def datasets_status(self, owner_slug, dataset_slug, **kwargs):  # noqa: E501
        """Get dataset creation status  # noqa: E501.

        :param str owner_slug: Dataset owner (required)
        :param str dataset_slug: Dataset name (required)
        :return: Result
        """
        return self.api_client.dataset_status(f"{owner_slug}/{dataset_slug})")

    def delete_model(self, owner_slug, model_slug):  # noqa: E501
        """Delete a model  # noqa: E501.

        :param str owner_slug: Model owner (required)
        :param str model_slug: Model name (required)
        :return: ApiDeleteModelResponse
        """
        return self.api_client.model_delete(f"{owner_slug}/{model_slug})")

    def delete_model_instance(self, owner_slug, model_slug, framework, instance_slug):  # noqa: E501
        """Delete a model instance  # noqa: E501.

        :param str owner_slug: Model owner (required)
        :param str model_slug: Model name (required)
        :param str framework: Model instance framework (required)
        :param str instance_slug: Model instance slug (required)
        :return: Result
        """
        return self.api_client.model_instance_delete(f"{owner_slug}/{model_slug}/{framework}/{instance_slug}", yes=True)

    def delete_model_instance_version(
        self, owner_slug, model_slug, framework, instance_slug, version_number
    ):  # noqa: E501
        """Delete a model instance version  # noqa: E501.

        :param str owner_slug: Model owner (required)
        :param str model_slug: Model name (required)
        :param str framework: Model instance framework (required)
        :param str instance_slug: Model instance slug (required)
        :param str version_number: Model instance version number (required)
        :return: Result
        """
        return self.api_client.model_instance_version_delete(
            f"{owner_slug}/{model_slug}/{framework}/{instance_slug}/{version_number}", yes=True
        )

    def get_model(self, owner_slug, model_slug):  # noqa: E501
        """Get a model  # noqa: E501.

        :param str owner_slug: Model owner (required)
        :param str model_slug: Model name (required)
        :return: Result
        """
        return self.api_client.model_get(f"{owner_slug}/{model_slug}")

    def get_model_instance(self, owner_slug, model_slug, framework, instance_slug):  # noqa: E501
        """Get a model instance  # noqa: E501.

        :param str owner_slug: Model owner (required)
        :param str model_slug: Model name (required)
        :param str framework: Model instance framework (required)
        :param str instance_slug: Model instance slug (required)
        :return: Result
        """
        return self.api_client.model_instance_get(f"{owner_slug}/{model_slug}/{framework}/{instance_slug}")

    def kernel_output(self, user_name, kernel_slug):  # noqa: E501
        """Download the latest output from a kernel  # noqa: E501.

        :param str user_name: Kernel owner (required)
        :param str kernel_slug: Kernel name (required)
        :return: Result
        """
        return self.api_client.kernels_output(f"{user_name}/{kernel_slug}", path=None, force=True)

    def kernel_pull(self, user_name, kernel_slug):  # noqa: E501
        """Pull the latest code from a kernel  # noqa: E501.

        :param str user_name: Kernel owner (required)
        :param str kernel_slug: Kernel name (required)
        :return: Result
        """
        return self.api_client.kernels_pull(f"{user_name}/{kernel_slug}", path=None)

    def kernel_push(self, kernel_push_request):  # noqa: E501
        """Push a new kernel version.  Can be used to create a new kernel and
        update an existing one.  # noqa: E501.

        :param KernelPushRequest kernel_push_request: Information for
            pushing a new kernel version (required)
        :return: Result
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_file = os.path.join(tmpdir, "kernel-metadata.json")
            (fd, code_file) = tempfile.mkstemp("code", "py", tmpdir, text=True)
            fd.write(json.dumps(kernel_push_request.code))
            os.close(fd)
            with open(meta_file, "w") as f:
                params = kernel_push_request.to_dict()
                params["code_file"] = code_file
                params["competition_sources"] = params.get("competition_data_sources")
                params["dataset_sources"] = params.get("dataset_data_sources")
                params["kernel_sources"] = params.get("kernel_data_sources")
                params["model_sources"] = params.get("model_data_sources")
                params["title"] = params.get("new_title")
                entries_to_remove = (
                    "competition_data_sources",
                    "dataset_data_sources",
                    "kernel_data_sources",
                    "model_data_sources",
                    "new_title",
                )
                for k in entries_to_remove:
                    params.pop(k, None)
                f.write(json.dumps(params))
            return self.api_client.kernels_push(tmpdir)

    def kernel_status(self, user_name, kernel_slug):  # noqa: E501
        """Get the status of the latest kernel version  # noqa: E501.

        :param str user_name: Kernel owner (required)
        :param str kernel_slug: Kernel name (required)
        :return: Result
        """
        return self.api_client.kernels_status(f"{user_name}/{kernel_slug}")

    def kernels_list(
        self,
        page=1,
        page_size=20,
        dataset=None,
        competition=None,
        parent_kernel=None,
        search=None,
        group="everyone",
        user=None,
        language=None,
        kernel_type=None,
        output_type=None,
        sort_by=None,
    ):  # noqa: E501
        """List kernels  # noqa: E501.

        :param int page: Page number
        :param int page_size: Page size
        :param str search: Search terms
        :param str group: Display only your kernels
        :param str user: Display kernels by a particular group
        :param str language: Display kernels in a specific language
        :param str kernel_type: Display kernels of a specific type
        :param str output_type: Display kernels with a specific output type
        :param str sort_by: Sort the results. 'relevance' only works if
            there is a search query
        :param str dataset: Display kernels using the specified dataset
        :param str competition: Display kernels using the specified
            competition
        :param str parent_kernel: Display kernels that have forked the
            specified kernel
        :return: Result
        """
        return self.api_client.kernels_list(
            page=page,
            page_size=page_size,
            dataset=dataset,
            competition=competition,
            parent_kernel=parent_kernel,
            search=search,
            mine=group != "everyone",
            user=user,
            language=language,
            kernel_type=kernel_type,
            output_type=output_type,
            sort_by=sort_by,
        )

    def metadata_get(self, owner_slug, dataset_slug):  # noqa: E501
        """Get the metadata for a dataset  # noqa: E501.

        :param str owner_slug: Dataset owner (required)
        :param str dataset_slug: Dataset name (required)
        :return: Result
        """
        return self.api_client.dataset_metadata(f"{owner_slug}/{dataset_slug}", None)

    def metadata_post(self, owner_slug, dataset_slug, settings, request):  # noqa: E501
        """Update the metadata for a dataset  # noqa: E501.

        :param str owner_slug: Dataset owner (required)
        :param str dataset_slug: Dataset name (required)
        :param DatasetUpdateSettingsRequest settings: Dataset metadata to
            update (required)
        :return: Result
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_file = os.path.join(tmpdir, "dataset-metadata.json")
            with open(meta_file, "w") as f:
                params = request.to_dict()
                params["isPrivate"] = params.get("is_private")
                params.pop("is_private", None)
                f.write(json.dumps(params))
            return self.api_client.dataset_metadata_update(f"{owner_slug}/{dataset_slug}", meta_file)

    def model_instance_versions_download(
        self, owner_slug, model_slug, framework, instance_slug, version_number
    ):  # noqa: E501
        """Download model instance version files  # noqa: E501.

        :param str owner_slug: Model owner (required)
        :param str model_slug: Model name (required)
        :param str framework: Model instance framework (required)
        :param str instance_slug: Model instance slug (required)
        :param str version_number: Model instance version number (required)
        :return: Result
        """
        v = f"{owner_slug}/{model_slug}/{framework}/{instance_slug}/{version_number}"
        return self.api_client.model_instance_version_download(v)

    def models_create_instance(self, owner_slug, model_slug, model_new_instance_request):  # noqa: E501
        """Create a new model instance  # noqa: E501.

        :param str owner_slug: Model owner (required)
        :param str model_slug: Model slug (required)
        :param ModelNewInstanceRequest model_new_instance_request:
            Information for creating a new model instance (required)
        :return: Result
        """
        print("models_create_instance() not implemented")
        print("use kaggle_api_extended.KaggleApi.model_instance_create() instead")
        raise NotImplementedError()

    def models_create_instance_version(
        self, owner_slug, model_slug, framework, instance_slug, model_instance_new_version_request
    ):  # noqa: E501
        """Create a new model instance version  # noqa: E501.

        :param str owner_slug: Model owner (required)
        :param str model_slug: Model slug (required)
        :param str framework: Model instance framework (required)
        :param str instance_slug: Model instance slug (required)
        :param ModelInstanceNewVersionRequest
            model_instance_new_version_request: Information for creating a
            new model instance version (required)
        :return: Result
        """
        print("models_create_instance_version() not implemented")
        print("use kaggle_api_extended.KaggleApi.model_instance_version_create() instead")
        raise NotImplementedError()

    def models_create_new(self, model_new_request):  # noqa: E501
        """Create a new model  # noqa: E501.

        :param ModelNewRequest model_new_request: Information for creating a
            new model (required)
        :return: Result
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_file = os.path.join(tmpdir, "model-metadata.json")
            with open(meta_file, "w") as f:
                params = model_new_request.to_dict()
                params["ownerSlug"] = params.get("owner_slug")
                params["isPrivate"] = params.get("is_private")
                params["publishTime"] = params.get("publish_time")
                params["provenanceSources"] = params.get("provenance_sources")
                entries_to_remove = ("owner_slug", "is_private", "publish_time", "provenance_sources")
                for k in entries_to_remove:
                    params.pop(k, None)
                f.write(json.dumps(params))
            return self.api_client.model_create_new(tmpdir)

    def models_list(self, sort_by=None, search=None, owner=None, page_size=20, page_token=None):
        """List models

        :param str search: Search terms
        :param str sort_by: Sort the results
        :param str owner: Display models by a specific user or organization
        :param int page_size: Page size
        :param str page_token: Page token for pagination
        :return: list of models
        """
        return self.api_client.model_list(
            sort_by=sort_by, search=search, owner=owner, page_size=page_size, page_token=page_token
        )

    def update_model(
        self,
        owner_slug,
        model_slug,
        model_update_request,
    ):  # noqa: E501
        """Update a model  # noqa: E501.

        :param str owner_slug: Model owner (required)
        :param str model_slug: Model name (required)
        :param ModelUpdateRequest model_update_request: Information for
            updating a model (required)
        :return: Result
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_file = os.path.join(tmpdir, "model-metadata.json")
            with open(meta_file, "w") as f:
                params = model_update_request.to_dict()
                params["ownerSlug"] = params.get("owner_slug")
                params["isPrivate"] = params.get("is_private")
                params["publishTime"] = params.get("publish_time")
                params["provenanceSources"] = params.get("provenance_sources")
                params["updateMask"] = params.get("update_mask")
                entries_to_remove = ("owner_slug", "is_private", "publish_time", "provenance_sources", "update_mask")
                for k in entries_to_remove:
                    params.pop(k, None)
                f.write(json.dumps(params))
            return self.api_client.model_update(tmpdir)

    def update_model_instance(
        self, owner_slug, model_slug, framework, instance_slug, model_instance_update_request
    ):  # noqa: E501
        """Update a model  # noqa: E501.

        :param str owner_slug: Model owner (required)
        :param str model_slug: Model name (required)
        :param str framework: Model instance framework (required)
        :param str instance_slug: Model instance slug (required)
        :param ModelInstanceUpdateRequest model_instance_update_request:
            Information for updating a model instance (required)
        :return: Result
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_file = os.path.join(tmpdir, "model-instance-metadata.json")
            with open(meta_file, "w") as f:
                params = model_instance_update_request.to_dict()
                params["licenseName"] = params.get("license_name")
                params["fineTunable"] = params.get("fine_tunable")
                params["trainingData"] = params.get("training_data")
                params["modelInstanceType"] = params.get("model_instance_type")
                params["baseModelInstance"] = params.get("base_model_instance")
                params["externalBaseModelUrl"] = params.get("external_base_model_url")
                params["updateMask"] = params.get("update_mask")
                entries_to_remove = (
                    "license_name",
                    "fine_tunable",
                    "training_data",
                    "model_instance_type",
                    "base_model_instance",
                    "external_base_model_url",
                    "update_mask",
                )
                for k in entries_to_remove:
                    params.pop(k, None)
                f.write(json.dumps(params))
            return self.api_client.model_instance_update(tmpdir)

    def upload_file(self, start_blob_upload_request):  # noqa: E501
        """Start uploading a file  # noqa: E501 :param StartBlobUploadRequest
        start_blob_upload_request: (required) :return: StartBlobUploadResponse."""
        print("upload_file() not implemented")
        print("it should not be needed after converting your code")
        print("to use kaggle_api_extended.KaggleApi.<high-level-fn>() instead")
        raise NotImplementedError()
