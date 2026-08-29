from unittest.mock import MagicMock, patch

from kaggle.api.kaggle_api_extended import KaggleApi


def test_dataset_files_json_keeps_stdout_parseable(capsys):
    api = KaggleApi.__new__(KaggleApi)
    response = MagicMock(error_message=None, next_page_token="next-token", files=[])

    with (
        patch.object(api, "dataset_list_files", return_value=response),
        patch.object(api, "print_results") as print_results,
    ):
        api.dataset_list_files_cli("owner/dataset", output_format="json")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Next Page Token = next-token\n"
    print_results.assert_called_once_with([], ["name", "size", "creationDate"], csv_display=False, output_format="json")


def test_dataset_files_table_keeps_pagination_on_stdout(capsys):
    api = KaggleApi.__new__(KaggleApi)
    response = MagicMock(error_message=None, next_page_token="next-token", files=[])

    with (
        patch.object(api, "dataset_list_files", return_value=response),
        patch.object(api, "print_results"),
    ):
        api.dataset_list_files_cli("owner/dataset")

    captured = capsys.readouterr()
    assert captured.out == "Next Page Token = next-token\n"
    assert captured.err == ""
