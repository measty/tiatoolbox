"""Tests for code related to obtaining slide information."""

import pathlib

from click.testing import CliRunner

from tiatoolbox import cli

# -------------------------------------------------------------------------------------
# Command Line Interface
# -------------------------------------------------------------------------------------


def test_command_line_slide_info(sample_all_wsis, tmp_path):
    """Test the Slide information CLI."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img-input",
            str(pathlib.Path(sample_all_wsis)),
            "--mode",
            "save",
            "--file-types",
            "*.ndpi, *.svs",
            "--output-path",
            str(tmp_path),
            "--verbose",
            "True",
        ],
    )

    assert slide_info_result.exit_code == 0
    assert pathlib.Path(tmp_path, "CMU-1-Small-Region.yaml").exists()
    assert pathlib.Path(tmp_path, "CMU-1.yaml").exists()
    assert not pathlib.Path(tmp_path, "test1.yaml").exists()


def test_command_line_slide_info_jp2(sample_all_wsis, tmp_path):
    """Test the Slide information CLI JP2, svs."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img-input",
            str(pathlib.Path(sample_all_wsis)),
            "--mode",
            "save",
        ],
    )

    output_dir = pathlib.Path(sample_all_wsis).parent
    assert slide_info_result.exit_code == 0
    assert pathlib.Path(output_dir, "meta-data", "CMU-1-Small-Region.yaml").exists()
    assert pathlib.Path(output_dir, "meta-data", "CMU-1.yaml").exists()
    assert pathlib.Path(output_dir, "meta-data", "test1.yaml").exists()


def test_command_line_slide_info_svs(sample_svs):
    """Test CLI slide info for single file."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img-input",
            sample_svs,
            "--file-types",
            "*.ndpi, *.svs",
            "--mode",
            "show",
            "--verbose",
            "True",
        ],
    )

    assert slide_info_result.exit_code == 0


def test_command_line_slide_info_file_not_found(sample_svs):
    """Test CLI slide info file not found error."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img-input",
            str(sample_svs)[:-1],
            "--file-types",
            "*.ndpi, *.svs",
            "--mode",
            "show",
        ],
    )

    assert slide_info_result.output == ""
    assert slide_info_result.exit_code == 1
    assert isinstance(slide_info_result.exception, FileNotFoundError)


def test_command_line_slide_info_output_none_mode_save(sample_svs):
    """Test CLI slide info for single file."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--img-input",
            str(sample_svs),
            "--file-types",
            "*.ndpi, *.svs",
            "--mode",
            "save",
            "--verbose",
            "False",
        ],
    )

    assert slide_info_result.exit_code == 0
    assert pathlib.Path(
        sample_svs.parent, "meta-data", "CMU-1-Small-Region.yaml"
    ).exists()


def test_command_line_slide_info_no_input():
    """Test CLI slide info for single file."""
    runner = CliRunner()
    slide_info_result = runner.invoke(
        cli.main,
        [
            "slide-info",
            "--file-types",
            "*.ndpi, *.svs",
            "--mode",
            "save",
        ],
    )

    assert "No image input provided." in slide_info_result.output
    assert slide_info_result.exit_code != 0
