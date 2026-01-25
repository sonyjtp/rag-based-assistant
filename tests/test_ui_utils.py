"""Unit tests for UI utility functions."""

from unittest.mock import mock_open, patch

import pytest

from src.ui_utils import configure_page, load_custom_styles


@pytest.fixture
def ui_mocks():
    """Fixture providing mocked UI components."""
    with patch("src.ui_utils.st") as mock_st, patch(
        "src.ui_utils.os.path.dirname"
    ) as mock_dirname:
        yield {
            "st": mock_st,
            "dirname": mock_dirname,
        }


# pylint: disable=redefined-outer-name
class TestLoadCustomStyles:
    """Test custom styles loading functionality."""

    def test_load_custom_styles_success(self, ui_mocks):
        """Test successfully loading and applying custom CSS styles."""
        css_content = "body { color: red; }"
        ui_mocks["dirname"].side_effect = ["/project/src", "/project"]

        with patch("builtins.open", mock_open(read_data=css_content)):
            load_custom_styles()

        ui_mocks["st"].markdown.assert_called_once()
        call_args = ui_mocks["st"].markdown.call_args[0][0]
        assert "body { color: red; }" in call_args

    def test_load_custom_styles_file_not_found(self, ui_mocks):
        """Test handling when CSS file is not found."""
        ui_mocks["dirname"].side_effect = ["/project/src", "/project"]

        with patch("builtins.open", side_effect=FileNotFoundError):
            load_custom_styles()

        ui_mocks["st"].warning.assert_called_once()
        warning_text = ui_mocks["st"].warning.call_args[0][0]
        assert "not found" in warning_text.lower()

    @pytest.mark.parametrize(
        "css_content,expected_in_output",
        [
            ("body { color: red; }", "color: red"),
            ("@media (max-width: 600px) { body { font-size: 14px; } }", "font-size"),
            (":root { --primary-color: #007bff; }", "--primary-color"),
        ],
    )
    def test_load_custom_styles_various_css(
        self, ui_mocks, css_content, expected_in_output
    ):
        """Parametrized test for various CSS content types."""
        ui_mocks["dirname"].side_effect = ["/project/src", "/project"]

        with patch("builtins.open", mock_open(read_data=css_content)):
            load_custom_styles()

        ui_mocks["st"].markdown.assert_called_once()
        call_args = ui_mocks["st"].markdown.call_args[0][0]
        assert expected_in_output in call_args

    def test_load_custom_styles_encoding(self, ui_mocks):
        """Test that file is opened with UTF-8 encoding."""
        css_content = "body { color: red; }"
        ui_mocks["dirname"].side_effect = ["/project/src", "/project"]

        with patch("builtins.open", mock_open(read_data=css_content)) as mock_file:
            load_custom_styles()

        call_kwargs = mock_file.call_args[1]
        assert call_kwargs["encoding"] == "utf-8"

    def test_load_custom_styles_markdown_parameters(self, ui_mocks):
        """Test that markdown is called with correct parameters."""
        css_content = "body { color: red; }"
        ui_mocks["dirname"].side_effect = ["/project/src", "/project"]

        with patch("builtins.open", mock_open(read_data=css_content)):
            load_custom_styles()

        call_kwargs = ui_mocks["st"].markdown.call_args[1]
        assert call_kwargs["unsafe_allow_html"] is True


# pylint: disable=redefined-outer-name
class TestConfigurePage:
    """Test Streamlit page configuration functionality."""

    def test_configure_page_calls_set_page_config(self, ui_mocks):
        """Test that set_page_config is called during page configuration."""
        configure_page()

        ui_mocks["st"].set_page_config.assert_called_once()

    def test_configure_page_title(self, ui_mocks):
        """Test that correct page title is configured."""
        configure_page()

        call_kwargs = ui_mocks["st"].set_page_config.call_args[1]
        assert call_kwargs["page_title"] == "RAG Chatbot"

    def test_configure_page_icon(self, ui_mocks):
        """Test that correct page icon is configured."""
        configure_page()

        call_kwargs = ui_mocks["st"].set_page_config.call_args[1]
        assert call_kwargs["page_icon"] == "🤖"

    def test_configure_page_layout(self, ui_mocks):
        """Test that layout is set to wide."""
        configure_page()

        call_kwargs = ui_mocks["st"].set_page_config.call_args[1]
        assert call_kwargs["layout"] == "wide"

    def test_configure_page_sidebar(self, ui_mocks):
        """Test that sidebar is set to expanded."""
        configure_page()

        call_kwargs = ui_mocks["st"].set_page_config.call_args[1]
        assert call_kwargs["initial_sidebar_state"] == "expanded"

    @pytest.mark.parametrize(
        "param_name,expected_value",
        [
            ("page_title", "RAG Chatbot"),
            ("page_icon", "🤖"),
            ("layout", "wide"),
            ("initial_sidebar_state", "expanded"),
        ],
    )
    def test_configure_page_parameters(self, ui_mocks, param_name, expected_value):
        """Parametrized test for all page configuration parameters."""
        configure_page()

        call_kwargs = ui_mocks["st"].set_page_config.call_args[1]
        assert call_kwargs[param_name] == expected_value

    def test_configure_page_no_return_value(self):
        """Test that configure_page returns None."""
        assert configure_page() is None

    def test_configure_page_called_once(self, ui_mocks):
        """Test that set_page_config is called exactly once."""
        configure_page()

        assert ui_mocks["st"].set_page_config.call_count == 1

    def test_configure_page_multiple_calls(self, ui_mocks):
        """Test that multiple calls to configure_page are independent."""
        configure_page()
        configure_page()

        assert ui_mocks["st"].set_page_config.call_count == 2
