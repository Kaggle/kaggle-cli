# coding=utf-8
import json
import shutil
import unittest
import os
import sys
import time

from requests import HTTPError

# noinspection PyTypeChecker
sys.path.insert(0, "..")

from kaggle import api

ApiException = IOError
# Unit test names include a letter to sort them in run order.
# That seemed easier and more obvious than defining a test suite.

# Run these tests in the container with the server running.
# In the case of unresolvable failure, run reset_database.sh.

# To run from Rider, create a Python tests>Unittests run config.
# Give it the module: unit_tests.TestKaggleApi
# Set the working directory to: kaggle-cli/tests
# Define some envars:
# KAGGLE_API_ENVIRONMENT=LOCALHOST
# KAGGLE_CONFIG_DIR=/home/kaggle/.config/kaggle/dev
# KAGGLE_KEY=local_api_token
# KAGGLE_USERNAME=<kaggle-user-name>

# Add those envars to the Python Tests>Unittest template to
# make running individual tests easier.

test_user = api.config_values["username"]
model_title = "testing"
instance_name = "test"
framework_name = "jax"
kernel_name = "testing-x"
dataset_name = "kaggleapi-dataset-x"
up_file = "sample_submission.csv"
description = "House prices submission message"
competition = "house-prices-advanced-regression-techniques"
dataset_directory = "dataset"
kernel_directory = "kernel"
model_directory = "model"
model_inst_directory = os.path.join(model_directory, "instance")
model_inst_vers_directory = os.path.join(model_inst_directory, "version")

# Max retries to get kernel status
max_status_tries = 10


def tearDownModule():
    file = os.path.join(dataset_directory, api.DATASET_METADATA_FILE)
    if os.path.exists(file):
        os.remove(file)
    file = os.path.join(kernel_directory, api.KERNEL_METADATA_FILE)
    if os.path.exists(file):
        os.remove(file)
    file = os.path.join(model_directory, api.MODEL_METADATA_FILE)
    if os.path.exists(file):
        os.remove(file)
    file = os.path.join(model_inst_directory, api.MODEL_INSTANCE_METADATA_FILE)
    if os.path.exists(file):
        os.remove(file)


def update_kernel_metadata_file(metadata_file, k_name):
    with open(metadata_file) as f:
        meta_data = json.load(f)
        meta_id = meta_data["id"]
        if "INSERT_KERNEL_SLUG_HERE" in meta_id:
            meta_id = meta_id.replace("INSERT_KERNEL_SLUG_HERE", k_name)
        meta_title = meta_data["title"]
        if "INSERT_TITLE_HERE" == meta_title:
            meta_title = k_name
        meta_path = meta_data["code_file"]
        if "INSERT_CODE_FILE_PATH_HERE" == meta_path:
            meta_path = f"{k_name}.ipynb"
        meta_data["id"] = meta_id
        meta_data["title"] = meta_title
        meta_data["code_file"] = meta_path
        meta_data["language"] = "python"
        meta_data["kernel_type"] = "notebook"
    with open(metadata_file, "w") as f:
        json.dump(meta_data, f, indent=2)
    return meta_data


def initialize_dataset_metadata_file(dataset_dir):
    metadata_file = os.path.join(dataset_dir, api.DATASET_METADATA_FILE)
    try:
        with open(metadata_file) as f:
            original = json.load(f)
            if "versionNumber" in original:
                version_num = int(original["versionNumber"])
            else:
                version_num = 0
    except FileNotFoundError:
        version_num = 0
    version_num += 1
    return version_num, metadata_file


def update_dataset_metadata_file(metadata_file, data_name, version_num):
    with open(metadata_file) as f:
        meta_data = json.load(f)
        meta_id = meta_data["id"]
        if "INSERT_SLUG_HERE" in meta_id:
            meta_id = meta_id.replace("INSERT_SLUG_HERE", data_name)
        meta_title = meta_data["title"]
        if "INSERT_TITLE_HERE" == meta_title:
            meta_title = data_name
        meta_data["id"] = meta_id
        meta_data["title"] = meta_title
        meta_data["versionNumber"] = version_num
    if not "resources" in meta_data:
        resource_list = [
            {
                "path": "data.csv",
                "description": "Description",
                "schema": {
                    "fields": [
                        {"name": "NumberField", "description": "id", "type": "number"},
                        {"name": "StringField", "description": "label", "type": "string"},
                    ]
                },
            }
        ]
        meta_data.update({"resources": resource_list})
    with open(metadata_file, "w") as f:
        json.dump(meta_data, f)


def update_model_metadata(metadata_file, owner, title, slug):
    with open(metadata_file) as f:
        meta_data = json.load(f)
        meta_id = meta_data["ownerSlug"]
        if "INSERT_OWNER_SLUG_HERE" == meta_id:
            meta_id = owner
        meta_title = meta_data["title"]
        if "INSERT_TITLE_HERE" == meta_title:
            meta_title = title
        meta_path = meta_data["slug"]
        if "INSERT_SLUG_HERE" == meta_path:
            meta_path = slug
        meta_data["ownerSlug"] = meta_id
        meta_data["title"] = meta_title
        meta_data["slug"] = meta_path
    with open(metadata_file, "w") as f:
        json.dump(meta_data, f, indent=2)
    return meta_data


def update_model_instance_metadata(metadata_file, owner, model_slug, instance_slug, framework):
    with open(metadata_file) as f:
        meta_data = json.load(f)
        meta_owner = meta_data["ownerSlug"]
        if "INSERT_OWNER_SLUG_HERE" == meta_owner:
            meta_owner = owner
        meta_framework = meta_data["framework"]
        if "INSERT_FRAMEWORK_HERE" == meta_framework:
            meta_framework = framework
        meta_instance = meta_data["instanceSlug"]
        if "INSERT_INSTANCE_SLUG_HERE" == meta_instance:
            meta_instance = instance_slug
        meta_model = meta_data["modelSlug"]
        if "INSERT_EXISTING_MODEL_SLUG_HERE" == meta_model:
            meta_model = model_slug
        meta_data["ownerSlug"] = meta_owner
        meta_data["modelSlug"] = meta_model
        meta_data["framework"] = meta_framework
        meta_data["instanceSlug"] = meta_instance
    with open(metadata_file, "w") as f:
        json.dump(meta_data, f, indent=2)
    return meta_data


def print_fields(instance, fields):  # For debugging.
    for f in fields:
        if not hasattr(instance, api.camel_to_snake(f)):
            print(f"Missing field: {f} named: {api.camel_to_snake(f)}")


# noinspection PyTypeChecker
class TestKaggleApi(unittest.TestCase):

    version_number, meta_file = initialize_dataset_metadata_file(dataset_directory)

    # Initialized from Response objects.
    competition_file = None
    kernel_slug = ""
    kernel_metadata_path = ""
    dataset = ""
    dataset_file = None
    model_instance = ""
    model_meta_data = None
    model_metadata_file = ""
    instance_metadata_file = ""

    # Inbox

    def test_files_upload(self):
        a = 2  # Change this value to run this test.
        if a - 1 == 1:
            return  # Only run this test when needed because it uploads an inbox file.
        filename = "tmp_file.test"
        with open(filename, "w") as f:
            f.write("test")
        try:
            api.files_upload_cli([filename], "kaggle-cli-test", False, False)
        finally:
            if os.path.exists("tmp_file.test"):
                os.remove("tmp_file.test")

    # Kernels

    def test_kernels_a_list(self):
        try:
            kernels = api.kernels_list(sort_by="dateCreated", user="stevemessick", language="python")
            self.assertGreater(len(kernels), 0)
            api.kernels_list_cli(user="stevemessick", csv_display=True)
        except ApiException as e:
            self.fail(f"kernels_list failed: {e}")

    def test_kernels_b_initialize(self):
        try:
            self.kernel_metadata_path = api.kernels_initialize(kernel_directory)
            self.assertTrue(os.path.exists(self.kernel_metadata_path))
        except ApiException as e:
            self.fail(f"kernels_initialize failed: {e}")

    def test_kernels_c_push(self):
        if self.kernel_metadata_path == "":
            self.test_kernels_b_initialize()
        try:
            md = update_kernel_metadata_file(self.kernel_metadata_path, kernel_name)
            push_result = api.kernels_push(kernel_directory)
            self.assertIsNotNone(push_result.ref)
            self.assertTrue(isinstance(push_result.version_number, int))
            self.kernel_slug = md["id"]
            time.sleep(30)
        except ApiException as e:
            self.fail(f"kernels_push failed: {e}")

    def test_kernels_d_status(self):
        if self.kernel_slug == "":
            self.test_kernels_c_push()
        try:
            status_result = api.kernels_status(self.kernel_slug)
            start_time = time.time()
            # If this loop is stuck because the kernel stays queued, go to the Kaggle website
            # on localhost and cancel the active event. That will exit the loop, but you may
            # need to clean up other active kernels to get it to run again.
            count = 0
            while status_result.status == "running" or status_result.status == "queued" or count >= max_status_tries:
                time.sleep(5)
                status_result = api.kernels_status(self.kernel_slug)
                print(status_result.status)
            if count >= max_status_tries:
                self.fail(f"Could not get kernel status in allowed trys. Status: {status_result.status}")
            end_time = time.time()
            print(f"kernels_status ready in {end_time-start_time}s")
        except ApiException as e:
            self.fail(f"kernels_status failed: {e}")

    def test_kernels_e_list_files(self):
        if self.kernel_slug == "":
            self.test_kernels_c_push()
        try:
            fs = api.kernels_list_files(self.kernel_slug)
            # TODO Make sure the test uses a kernel that has at least one file.
            self.assertGreaterEqual(len(fs.files), 0)
        except ApiException as e:
            self.fail(f"kernels_list_files failed: {e}")

    def test_kernels_f_output(self):
        fs = []
        try:
            fs, token = api.kernels_output("kerneler/sqlite-global-default", "kernel/tmp")
            self.assertIsInstance(fs, list)
            if token:
                print(token)
        except ApiException as e:
            self.fail(f"kernels_output failed: {e}")
        finally:
            for file in fs:
                if os.path.exists(file):
                    os.remove(file)
            if os.path.exists("kernel/tmp"):
                os.rmdir("kernel/tmp")

    def test_kernels_g_pull(self):
        if self.kernel_metadata_path == "":
            self.test_kernels_c_push()
        fs = ""
        try:
            fs = api.kernels_pull(f"{test_user}/{kernel_name}", "kernel/tmp", metadata=True)
            self.assertTrue(os.path.exists(fs))
            with open(f'{fs}/{self.kernel_metadata_path.split("/")[1]}') as f:
                metadata = json.load(f)
                [
                    self.assertTrue(metadata.get(f))
                    for f in ["id", "id_no", "title", "code_file", "language", "kernel_type"]
                ]
                [
                    self.assertTrue(metadata.get(f) is not None)
                    for f in [
                        "is_private",
                        "enable_gpu",
                        "enable_tpu",
                        "enable_internet",
                        "keywords",
                        "dataset_sources",
                        "kernel_sources",
                        "competition_sources",
                        "model_sources",
                    ]
                ]
        except ApiException as e:
            self.fail(f"kernels_pull failed: {e}")
        finally:
            shutil.rmtree(fs)

    # Competitions

    def test_competition_a_list(self):
        try:
            competitions = api.competitions_list(group="general").competitions
            self.assertGreater(len(competitions), 0)
            self.assertLessEqual(len(competitions), 20)
            [self.assertTrue(hasattr(competitions[0], api.camel_to_snake(f))) for f in api.competition_fields]
            competitions = api.competitions_list(page=2, category="gettingStarted", sort_by="prize").competitions
            self.assertEqual(len(competitions), 1)
        except ApiException as e:
            self.fail(f"competitions_list failed: {e}")

    def test_competition_b_submit(self):
        try:
            self.skip_submissions = False
            result = api.competition_submit_cli(up_file, description, competition)
            if not result or not result.startswith("Successfully submitted"):
                self.fail(f"competition_submit failed: {result}")
        except HTTPError:
            # Handle submission limit reached gracefully (potentially skip the test)
            print("Competition submission limit reached for the day")
            self.skip_submissions = True
            pass
        except ApiException as e:
            self.fail(f"competition_submit failed: {e}")

    def test_competition_c_submissions(self):
        self.test_competition_b_submit()
        try:
            submissions = api.competition_submissions(competition)
            self.assertIsInstance(submissions, list)
            if not self.skip_submissions:
                self.assertGreater(len(submissions), 0)
                [self.assertTrue(hasattr(submissions[0], api.camel_to_snake(f))) for f in api.submission_fields]
        except ApiException as e:
            self.fail(f"competition_submissions failed: {e}")

    def test_competition_d_list_files(self):
        try:
            competition_files = api.competition_list_files(competition).files
            self.assertIsInstance(competition_files, list)
            self.assertGreater(len(competition_files), 0)
            self.competition_file = competition_files[0]
            [self.assertTrue(hasattr(competition_files[0], api.camel_to_snake(f))) for f in api.competition_file_fields]
        except ApiException as e:
            self.fail(f"competition_list_files failed: {e}")

    def test_competition_e_download_file(self):
        if self.competition_file is None:
            self.test_competition_d_list_files()
        try:
            api.competition_download_file(competition, self.competition_file.ref, force=True)
            self.assertTrue(os.path.exists(self.competition_file.ref))
            api.competition_download_file(competition, self.competition_file.ref, force=False)
        except ApiException as e:
            self.fail(f"competition_download_file failed: {e}")
        finally:
            if os.path.exists(self.competition_file.ref):
                os.remove(self.competition_file.ref)

    def test_competition_f_download_files(self):
        try:
            api.competition_download_files(competition)
            self.assertTrue(os.path.exists(f"{competition}.zip"))
            self.assertTrue(os.path.getsize(f"{competition}.zip") > 0)
        except ApiException as e:
            self.fail(f"competition_download_files failed: {e}")
        finally:
            if os.path.exists(f"{competition}.zip"):
                os.remove(f"{competition}.zip")

    def test_competition_g_leaderboard_view(self):
        try:
            result = api.competition_leaderboard_view(competition)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            [self.assertTrue(hasattr(result[0], api.camel_to_snake(f))) for f in api.competition_leaderboard_fields]
        except ApiException as e:
            self.fail(f"competition_leaderboard_view failed: {e}")

    def test_competition_h_leaderboard_download(self):
        try:
            api.competition_leaderboard_download(competition, "tmp")
            self.assertTrue(os.path.exists(f"tmp/{competition}.zip"))
        except ApiException as e:
            self.fail(f"competition_leaderboard_download failed: {e}")
        finally:
            if os.path.exists(f"tmp/{competition}.zip"):
                os.remove(f"tmp/{competition}.zip")
            if os.path.exists("tmp"):
                os.rmdir("tmp")

    def test_competition_i_topics(self):
        try:
            response = api.competition_list_topics(competition)
            self.assertTrue(hasattr(response, "topics"))
            self.assertTrue(hasattr(response, "total_count"))
            topics = response.topics
            self.assertIsInstance(topics, list)
            self.assertGreaterEqual(response.total_count, 0)
            if topics:
                [self.assertTrue(hasattr(topics[0], api.camel_to_snake(f))) for f in api.competition_topic_fields]
                self.competition_topic_id = topics[0].id
            else:
                self.competition_topic_id = None
        except ApiException as e:
            self.fail(f"competition_list_topics failed: {e}")

    def test_competition_j_topic_messages(self):
        if not getattr(self, "competition_topic_id", None):
            self.test_competition_i_topics()
        if not getattr(self, "competition_topic_id", None):
            self.skipTest("No topics available to fetch messages from")
        try:
            response = api.competition_list_topic_messages(competition, self.competition_topic_id)
            self.assertTrue(hasattr(response, "messages"))
            self.assertIsInstance(response.messages, list)
            if response.messages:
                [
                    self.assertTrue(hasattr(response.messages[0], api.camel_to_snake(f)))
                    for f in api.competition_topic_message_fields
                ]
        except ApiException as e:
            self.fail(f"competition_list_topic_messages failed: {e}")

    # Datasets

    def test_dataset_a_list(self):
        try:
            datasets = api.dataset_list(sort_by="votes")
            self.assertGreater(len(datasets), 0)
            self.dataset = str(datasets[0].ref)
            [self.assertTrue(hasattr(datasets[0], api.camel_to_snake(f))) for f in api.dataset_fields]
            datasets = api.dataset_list(license_name="other", file_type="bigQuery")
            self.assertGreater(len(datasets), 10)
        except ApiException as e:
            self.fail(f"dataset_list failed: {e}")

    def test_dataset_b_metadata(self):
        if self.dataset == "":
            self.test_dataset_a_list()
        try:
            m = api.dataset_metadata(self.dataset, dataset_directory)
            self.assertTrue(os.path.exists(m))
        except ApiException as e:
            self.fail(f"dataset_metadata failed: {e}")

    def test_dataset_c_metadata_update(self):
        if self.dataset == "":
            self.test_dataset_a_list()
        if not os.path.exists(os.path.join(dataset_directory, api.DATASET_METADATA_FILE)):
            self.test_dataset_b_metadata()
        try:
            api.dataset_metadata_update(self.dataset, dataset_directory)
            # TODO Make the API method return something, and not exit when it fails.
        except ApiException as e:
            self.fail(f"dataset_metadata_update failed: {e}")

    def test_dataset_d_list_files(self):
        if self.dataset == "":
            self.test_dataset_a_list()
        try:
            response = api.dataset_list_files(self.dataset)
            self.assertIsInstance(response.dataset_files, list)
            self.assertGreater(len(response.dataset_files), 0)
            self.dataset_file = response.dataset_files[0]
            [self.assertTrue(hasattr(self.dataset_file, api.camel_to_snake(f))) for f in api.dataset_file_fields]
            api.dataset_list_files_cli(self.dataset)
        except ApiException as e:
            self.fail(f"dataset_list_files failed: {e}")

    def test_dataset_e_status(self):
        if self.dataset == "":
            self.test_dataset_a_list()
        try:
            status = api.dataset_status(self.dataset)
            self.assertIn(status, ["ready", "pending", "error"])
        except ApiException as e:
            self.fail(f"dataset_status failed: {e}")

    def test_dataset_f_download_file(self):
        if self.dataset_file is None:
            self.test_dataset_d_list_files()
        try:
            api.dataset_download_file(self.dataset, self.dataset_file.name, "tmp")
            self.assertTrue(os.path.exists(f"tmp/{self.dataset_file.name}"))
        except ApiException as e:
            self.fail(f"dataset_download_file failed: {e}")
        finally:
            if os.path.exists(f"tmp/{self.dataset_file.name}"):
                os.remove(f"tmp/{self.dataset_file.name}")
            if os.path.exists("tmp"):
                os.rmdir("tmp")

    def test_dataset_g_download_files(self):
        if self.dataset == "":
            self.test_dataset_a_list()
        ds = ["a", "b"]
        try:
            api.dataset_download_files(self.dataset + "/1")
            ds = self.dataset.split("/")
            self.assertTrue(os.path.exists(f"{ds[1]}.zip"))
        except ApiException as e:
            self.fail(f"dataset_download_files failed: {e}")
        finally:
            if os.path.exists(f"{ds[1]}.zip"):
                os.remove(f"{ds[1]}.zip")

    def test_dataset_h_initialize(self):
        try:
            api.dataset_initialize("dataset")
            self.assertTrue(os.path.exists(os.path.join(dataset_directory, api.DATASET_METADATA_FILE)))
        except ApiException as e:
            self.fail(f"dataset_initialize failed: {e}")

    def test_dataset_ia_create_new(self, fail_if_exists=True):
        if not os.path.exists(os.path.join(dataset_directory, api.DATASET_METADATA_FILE)):
            self.test_dataset_h_initialize()
        try:
            update_dataset_metadata_file(self.meta_file, dataset_name, self.version_number)
            new_dataset = api.dataset_create_new(dataset_directory)
            self.assertIsNotNone(new_dataset)
            if len(new_dataset.error) > 0:
                if fail_if_exists:
                    self.fail(f"dataset_create_new failed: {new_dataset.error}")
        except ApiException as e:
            if fail_if_exists:
                self.fail(f"dataset_create_new failed: {e}")

    def test_dataset_ib_wait_after_create(self):
        # When running all tests sequentially, give the new dataset some time to stabilize.
        time.sleep(10)  # TODO: Use a status check.

    def test_dataset_j_create_version(self):
        if not os.path.exists(os.path.join(dataset_directory, api.DATASET_METADATA_FILE)):
            self.test_dataset_ia_create_new(fail_if_exists=False)
        try:
            new_version = api.dataset_create_version(dataset_directory, "Notes")
            self.assertIsNotNone(new_version)
            self.assertTrue(new_version.error == "")
            self.assertFalse(new_version.ref == "")
        except ApiException as e:
            self.fail(f"dataset_create_version failed: {e}")

    def test_dataset_k_delete(self):
        self.test_dataset_ia_create_new(fail_if_exists=False)
        api.dataset_delete(None, dataset_name, True)

    # Models

    def test_model_a_list(self):
        try:
            ms = api.model_list()
            self.assertIsInstance(ms, list)
            self.assertGreater(len(ms), 0)
            [self.assertTrue(hasattr(ms[0], api.camel_to_snake(f))) for f in api.model_fields]
        except ApiException as e:
            self.fail(f"models_list failed: {e}")

    def test_model_b_initialize(self):
        try:
            self.model_metadata_file = api.model_initialize(model_directory)
            self.assertTrue(os.path.exists(self.model_metadata_file))
            self.model_meta_data = update_model_metadata(self.model_metadata_file, test_user, model_title, model_title)
            self.model_instance = f'{test_user}/{self.model_meta_data["slug"]}/{framework_name}/{instance_name}'
        except ApiException as e:
            self.fail(f"model_initialize failed: {e}")

    def test_model_c_create_new(self):
        if self.model_metadata_file == "":
            self.test_model_b_initialize()
        try:
            model = api.model_create_new(model_directory)
            if model.error:
                if "already used" in model.error:
                    delete_response = api.model_delete(f"{test_user}/{model_title}", True)
                    if delete_response.error:
                        self.fail(delete_response.error)
                    else:
                        model = api.model_create_new(model_directory)
                        if model.error:
                            self.fail(model.error)
                else:
                    self.fail(model.error)
            self.assertIsNotNone(model.ref)
            self.assertGreater(len(model.ref), 0)
            [self.assertTrue(hasattr(model, api.camel_to_snake(f))) for f in ["id", "url"]]
        except ApiException as e:
            self.fail(f"model_create_new failed: {e}")

    def test_model_d_get(self):
        try:
            model_data = api.model_get(f"{test_user}/{model_title}")
            self.assertIsNotNone(model_data.ref)
            self.assertGreater(len(model_data.ref), 0)
            self.assertEqual(model_data.title, model_title)
            [self.assertTrue(hasattr(model_data, api.camel_to_snake(f))) for f in api.model_all_fields]
        except ApiException as e:
            self.fail(f"model_get failed: {e}")

    def test_model_e_update(self):
        if self.model_metadata_file == "":
            self.test_model_c_create_new()
        try:
            update_response = api.model_update(model_directory)
            self.assertEqual(len(update_response.error), 0)
            self.assertIsNotNone(update_response.ref)
            self.assertGreater(len(update_response.ref), 0)
        except ApiException as e:
            self.fail(f"model_update failed: {e}")

    # Model instances

    def test_model_instance_a_initialize(self):
        try:
            self.instance_metadata_file = api.model_instance_initialize(model_inst_directory)
            self.assertTrue(os.path.exists(self.instance_metadata_file))
        except ApiException as e:
            self.fail(f"model_instance_initialize failed: {e}")

    def test_model_instance_b_create(self, check_result: bool = True):
        if self.model_meta_data is None:
            self.test_model_b_initialize()
        if self.instance_metadata_file == "":
            self.test_model_instance_a_initialize()
        try:
            assert self.model_meta_data is not None
            update_model_instance_metadata(
                self.instance_metadata_file, test_user, self.model_meta_data["slug"], instance_name, framework_name
            )
            inst_create_resp = api.model_instance_create(model_inst_directory)
            if check_result:
                if inst_create_resp.error:
                    if "already exists" in inst_create_resp.error:
                        delete_response = api.model_instance_delete(f"{test_user}/{model_title}", True)
                        if delete_response.error:
                            self.fail(delete_response.error)
                        else:
                            inst_create_resp = api.model_instance_create(model_inst_directory)
                            if inst_create_resp.error:
                                self.fail(inst_create_resp.error)
                    else:
                        self.fail(inst_create_resp.error)
                self.assertIsNotNone(inst_create_resp.ref)
                self.assertGreater(len(inst_create_resp.ref), 0)
        except ApiException as e:
            self.fail(f"model_instance_create failed: {e}")

    def test_model_instance_b_wait_after_create(self):
        # When running all tests sequentially, give the new model some time to stabilize.
        time.sleep(10)  # TODO: Find a better way to detect model stability.

    def test_model_instance_c_get(self):
        if self.model_instance == "":
            self.test_model_b_initialize()
            self.test_model_c_create_new()
            self.test_model_instance_a_initialize()
            self.test_model_instance_b_create()
            self.test_model_instance_b_wait_after_create()
            self.test_model_instance_c_get()
        try:
            inst_get_resp = api.model_instance_get(self.model_instance)
            self.assertIsNotNone(inst_get_resp.url)
            self.assertGreater(len(inst_get_resp.url), 0)
            os.makedirs("model/tmp", exist_ok=True)
            api.model_instance_get_cli(self.model_instance, "model/tmp")
        except ApiException as e:
            self.fail(f"model_instance_get failed: {e}")
        finally:
            shutil.rmtree("model/tmp")

    def test_model_instance_d_files(self):
        if self.model_instance == "":
            self.test_model_b_initialize()
        try:
            inst_files_resp = api.model_instance_files(self.model_instance)
            self.assertIsInstance(inst_files_resp.files, list)
            self.assertGreater(len(inst_files_resp.files), 0)
            [self.assertTrue(hasattr(inst_files_resp.files[0], api.camel_to_snake(f))) for f in api.model_file_fields]
        except ApiException as e:
            self.fail(f"model_instance_files failed: {e}")

    def test_model_instance_e_update(self):
        if self.model_instance == "":
            self.test_model_b_initialize()
            self.test_model_instance_a_initialize()
            try:
                self.test_model_c_create_new()
            except AssertionError:
                pass
            try:
                update_model_instance_metadata(
                    self.instance_metadata_file, test_user, self.model_meta_data["slug"], instance_name, framework_name
                )
            except AssertionError:
                pass
            self.test_model_instance_b_create(check_result=False)
            self.test_model_instance_b_wait_after_create()
        try:
            inst_update_resp = api.model_instance_update(model_inst_directory)
            self.assertIsNotNone(inst_update_resp)
            self.assertIsNotNone(inst_update_resp.ref)
            self.assertGreater(len(inst_update_resp.ref), 0)
            [self.assertTrue(hasattr(inst_update_resp, api.camel_to_snake(f))) for f in ["error", "id", "ref", "url"]]
        except ApiException as e:
            self.fail(f"model_instance_update failed: {e}")

    def test_model_instance_f_list(self):
        if self.model_instance == "":
            self.test_model_instance_c_get()
        try:
            a, b, c, d = self.model_instance.split("/")
            inst_files_resp = api.model_instances_list(f"{a}/{b}")
            self.assertIsInstance(inst_files_resp.instances, list)
            self.assertGreater(len(inst_files_resp.instances), 0)
            [
                self.assertTrue(hasattr(inst_files_resp.instances[0], api.camel_to_snake(f)))
                for f in api.model_instance_fields
            ]
        except ApiException as e:
            self.fail(f"model_instance_files failed: {e}")

    # Model instance versions

    def test_model_instance_version_a_create(self):
        if self.model_instance == "":
            self.test_model_b_initialize()
        try:
            version_metadata_resp = api.model_instance_version_create(self.model_instance, model_inst_vers_directory)
            self.assertIsNotNone(version_metadata_resp.ref)
            [self.assertTrue(hasattr(version_metadata_resp, api.camel_to_snake(f))) for f in ["id", "url", "error"]]
        except ApiException as e:
            self.fail(f"model_instance_version_create failed: {e}")

    def test_model_instance_version_b_files(self):
        if self.model_instance == "":
            self.test_model_b_initialize()
        try:
            r = api.model_instance_version_files(f"{self.model_instance}/1")
            self.assertIsInstance(r.files, list)
            self.assertGreater(len(r.files), 0)
            api.model_instance_version_files_cli(f"{self.model_instance}/1")
        except ApiException as e:
            self.fail(f"model_instance_version_files failed: {e}")

    def test_model_instance_version_c_download(self):
        for x in range(0, 5):
            if self.model_instance == "":
                self.test_model_b_initialize()
            version_file = ""
            try:
                version_file = api.model_instance_version_download(f"{self.model_instance}/1", "tmp", force=True)
                self.assertTrue(os.path.exists(version_file))
            except KeyError:
                pass  # TODO Create a version that has content.
            except ApiException as e:
                if e.strerror.startsWith("The archive is not ready yet"):
                    continue
                self.fail(f"model_instance_version_download failed: {e}")
            finally:
                if os.path.exists(version_file):
                    os.remove(version_file)
                if os.path.exists("tmp"):
                    os.rmdir("tmp")
                return

    def test_model_instance_version_d_list(self):
        pass

    # Kernel deletion

    def test_kernels_h_delete(self):
        print("Running test_kernels_h_delete")
        if self.kernel_slug == "":
            self.test_kernels_c_push()
        try:
            api.kernels_delete(self.kernel_slug, no_confirm=True)
            # The kernels_delete method prints success, no specific return to assert.
            # If no exception is raised, we assume success.
        except ApiException as e:
            self.fail(f"kernels_delete failed: {e}")

    # Model deletion

    def test_model_instance_version_d_delete(self):
        if self.model_instance == "":
            self.test_model_b_initialize()
        try:
            version_delete_resp = api.model_instance_version_delete(f"{self.model_instance}/1", True)
            self.assertEqual(len(version_delete_resp.error), 0, msg=version_delete_resp.error)
        except ApiException as e:
            self.fail(f"model_instance_version_delete failed: {e}")

    def test_model_instance_x_delete(self):
        if self.model_instance == "":
            self.test_model_b_initialize()
        try:
            inst_update_resp = api.model_instance_delete(self.model_instance, True)
            self.assertIsNotNone(inst_update_resp)
            if len(inst_update_resp.error):
                print(inst_update_resp.error)
            self.assertEqual(len(inst_update_resp.error), 0)
        except ApiException as e:
            self.fail(f"model_instance_delete failed: {e}")

    def test_model_z_delete(self):
        try:
            delete_response = api.model_delete(f"{test_user}/{model_title}", True)
            if delete_response.error:
                self.fail(delete_response.error)
            else:
                pass
        except ApiException as e:
            self.fail(f"model_delete failed: {e}")


if __name__ == "__main__":
    unittest.main()
