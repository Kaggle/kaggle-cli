"""Metadata and task files are read and written as UTF-8, whatever the system's default encoding is.

On Windows, open() without an encoding uses the ANSI code page: cp1252 on most Western systems, cp932 on
Japanese ones. UTF-8 text in a metadata file was then silently garbled ("×" was uploaded as "Ã—") or the
command failed with UnicodeDecodeError. The autouse fixture below reproduces a cp1252 machine on any OS.

Metadata files are read as utf-8-sig, so a file saved with a UTF-8 BOM (Notepad, or PowerShell 5's
`Set-Content -Encoding UTF8`) works too, and a file saved in another encoding gets an error that says what to do.
"""

import builtins
import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import HTTPError

from kaggle.api.kaggle_api_extended import BENCHMARKS_SYNTAX_REF, KaggleApi
from kagglesdk.models.types.model_api_service import ApiCreateModelResponse

GARBLED = "144,606 trees × 28 columns"  # cp1252 decodes it without an error, as "144,606 trees Ã— 28 columns"
UNDECODABLE = "東京の天気"  # cp1252 can't decode it at all
TEXTS = [pytest.param(GARBLED, id="garbled"), pytest.param(UNDECODABLE, id="undecodable")]

MODEL = {"ownerSlug": "testuser", "slug": "test-model", "title": "Test Model Title", "isPrivate": True}
MODEL_INSTANCE = {
    "ownerSlug": "testuser",
    "modelSlug": "test-model",
    "instanceSlug": "test-instance",
    "framework": "keras",
    "licenseName": "Apache 2.0",
}
DATASET = {"id": "testuser/test-dataset", "title": "Test Dataset Title", "licenses": [{"name": "CC0-1.0"}]}


@pytest.fixture(autouse=True)
def cp1252_default_encoding(monkeypatch):
    """Make open() in kaggle_api_extended default to cp1252, as it does on Western European Windows."""

    def open_like_windows(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        if "b" not in mode and encoding is None:
            encoding = "cp1252"
        return builtins.open(file, mode, buffering, encoding, *args, **kwargs)

    monkeypatch.setattr(sys.modules[KaggleApi.__module__], "open", open_like_windows, raising=False)


@pytest.fixture
def api():
    a = KaggleApi()
    a.config_values = {"username": "testuser"}
    a.already_printed_version_warning = True
    a.build_kaggle_client = MagicMock()  # type: ignore[method-assign]
    a.upload_files = MagicMock()  # type: ignore[method-assign]
    return a


def _client(api):
    return api.build_kaggle_client.return_value.__enter__.return_value


@pytest.fixture(params=["utf-8", "utf-8-sig"], ids=["utf8", "utf8_bom"])
def write_json(request):
    """Writes a metadata file as UTF-8, without and with a BOM."""

    def write(path, data):
        # ensure_ascii=False keeps the characters themselves in the file, as an editor would.
        path.write_text(json.dumps(data, ensure_ascii=False), encoding=request.param)

    return write


@pytest.mark.parametrize("text", TEXTS)
def test_dataset_metadata_update_reads_utf8(api, tmp_path, write_json, text):
    write_json(tmp_path / "dataset-metadata.json", {"title": "Test Dataset Title", "description": text})
    update = _client(api).datasets.dataset_api_client.update_dataset_metadata
    update.return_value.errors = []

    api.dataset_metadata_update("testuser/test-dataset", str(tmp_path))

    assert update.call_args[0][0].settings.description == text


@pytest.mark.parametrize("text", TEXTS)
def test_dataset_create_new_reads_utf8(api, tmp_path, write_json, text):
    api.dataset_status = MagicMock(side_effect=HTTPError())  # type: ignore[method-assign]
    write_json(tmp_path / "dataset-metadata.json", {**DATASET, "description": text})
    create = _client(api).datasets.dataset_api_client.create_dataset
    create.return_value.error = ""

    api.dataset_create_new(str(tmp_path))

    assert create.call_args[0][0].description == text


@pytest.mark.parametrize("text", TEXTS)
def test_dataset_create_version_reads_utf8(api, tmp_path, write_json, text):
    write_json(tmp_path / "dataset-metadata.json", {**DATASET, "description": text})

    api.dataset_create_version(str(tmp_path), "version notes")

    request = _client(api).datasets.dataset_api_client.create_dataset_version.call_args[0][0]
    assert request.body.description == text


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize(
    "method, client_method", [("model_create_new", "create_model"), ("model_update", "update_model")]
)
def test_model_metadata_is_read_as_utf8(api, tmp_path, write_json, text, method, client_method):
    write_json(tmp_path / "model-metadata.json", {**MODEL, "description": text})
    call = getattr(_client(api).models.model_api_client, client_method)
    call.return_value = ApiCreateModelResponse()

    getattr(api, method)(str(tmp_path))

    assert call.call_args[0][0].description == text


@pytest.mark.parametrize("text", TEXTS)
def test_model_instance_create_reads_utf8(api, tmp_path, write_json, text):
    write_json(tmp_path / "model-instance-metadata.json", {**MODEL_INSTANCE, "overview": text})
    create = _client(api).models.model_api_client.create_model_instance
    create.return_value = ApiCreateModelResponse()

    api.model_instance_create(str(tmp_path), quiet=True)

    assert create.call_args[0][0].body.overview == text


@pytest.mark.parametrize("text", TEXTS)
def test_model_instance_update_reads_utf8(api, tmp_path, write_json, text):
    write_json(tmp_path / "model-instance-metadata.json", {**MODEL_INSTANCE, "overview": text})

    api.model_instance_update(str(tmp_path))

    assert _client(api).models.model_api_client.update_model_instance.call_args[0][0].overview == text


@pytest.mark.parametrize("text", TEXTS)
def test_competition_create_new_reads_utf8(api, tmp_path, write_json, text):
    meta = {"title": "My Awesome Comp", "slug": "my-awesome-comp", "briefDescription": text, "privacy": "PUBLIC"}
    write_json(tmp_path / KaggleApi.COMPETITION_METADATA_FILE, meta)
    create = _client(api).competitions.competition_api_client.create_competition
    create.return_value.url = "https://www.kaggle.com/c/my-awesome-comp"

    api.competition_create_new(str(tmp_path))

    assert create.call_args[0][0].brief_description == text


@pytest.mark.parametrize("kernel", [None, "testuser/my-kernel"], ids=["from_metadata", "named"])
def test_kernels_pull_reads_existing_metadata_as_utf8(api, tmp_path, write_json, kernel):
    # Only id and code_file are used, so a garbled title was harmless; one cp1252 can't decode made the pull fail.
    write_json(
        tmp_path / "kernel-metadata.json", {"id": "testuser/my-kernel", "title": UNDECODABLE, "code_file": "mine.py"}
    )
    blob = _client(api).kernels.kernels_api_client.get_kernel.return_value.blob
    blob.language, blob.kernel_type, blob.slug, blob.source = "python", "script", "my-kernel", "print('pulled')"

    api.kernels_pull(kernel, path=str(tmp_path))

    assert (tmp_path / "mine.py").read_text(encoding="utf-8") == "print('pulled')"


@pytest.mark.parametrize("text", TEXTS)
def test_kernels_push_reads_metadata_as_utf8(api, tmp_path, write_json, text):
    # A title that doesn't match the slug only prints a warning.
    meta = {"id": "testuser/my-kernel", "title": f"My Kernel {text}", "code_file": "script.py"}
    write_json(tmp_path / "kernel-metadata.json", {**meta, "language": "python", "kernel_type": "script"})
    (tmp_path / "script.py").write_text("print('hi')", encoding="utf-8")
    response = _client(api).kernels.kernels_api_client.save_kernel.return_value
    response.error, response.invalid_tags, response.url, response.version_number = None, [], "https://k", 1

    api.kernels_push(str(tmp_path))

    request = _client(api).kernels.kernels_api_client.save_kernel.call_args[0][0]
    assert request.new_title == f"My Kernel {text}"


def test_metadata_in_another_encoding_says_how_to_fix_it(api, tmp_path):
    meta_file = tmp_path / "dataset-metadata.json"
    meta_file.write_text(json.dumps({**DATASET, "description": GARBLED}, ensure_ascii=False), encoding="cp1252")

    with pytest.raises(ValueError, match="is not UTF-8 encoded, please re-save it as UTF-8") as raised:
        api.dataset_create_version(str(tmp_path), "version notes")

    assert str(meta_file) in str(raised.value)


@pytest.mark.parametrize("text", TEXTS)
def test_benchmarks_push_reads_task_file_as_utf8(api, tmp_path, text):
    content = f'@task(name="my-task")\ndef evaluate():\n    return "{text}"\n'
    task_file = tmp_path / "task.py"
    task_file.write_text(content, encoding="utf-8")
    benchmarks = _client(api).benchmarks.benchmark_tasks_api_client
    benchmarks.get_benchmark_task.side_effect = HTTPError(response=MagicMock(status_code=404))
    created = benchmarks.create_benchmark_task.return_value
    created.slug.task_slug, created.url, created.error = "my-task", "https://kaggle.com/benchmarks/my-task", None
    jupytext = MagicMock()
    jupytext.reads.return_value.metadata = {}
    jupytext.writes.return_value = '{"cells": []}'

    with patch.dict("sys.modules", {"jupytext": jupytext}):
        api.benchmarks_tasks_push_cli("my-task", str(task_file))

    assert jupytext.reads.call_args[0][0] == content


def test_benchmarks_reference_is_written_as_utf8(api, tmp_path):
    assert any(ord(c) > 127 for c in BENCHMARKS_SYNTAX_REF)  # the reference has an em dash

    api._write_benchmarks_reference(str(tmp_path), quiet=True)

    assert (tmp_path / "kaggle_benchmarks_reference.md").read_text(encoding="utf-8") == BENCHMARKS_SYNTAX_REF
