import pytest

from airtable import Airtable


@pytest.fixture
def table_names():
    """Dict of some test table names with characters that are not url safe with
    url-escaped versions.
    """
    return {
        "Test table 1": "Test%20table%201",
        "Test/Table 2": "Test%2FTable%202",
        "Another (test) table": "Another%20%28test%29%20table",
        "A & test & table": "A%20%26%20test%20%26%20table",
        "percentage % table": "percentage%20%25%20table",
    }


def test_url_escape(table_names):
    """Test for proper escaping of urls including unsafe characters in table
    names (which airtable allows).
    """
    for table_name, escaped in table_names.items():
        airtable = Airtable("x", table_name, api_key="z")
        # Class-generated url should be properly escaped
        assert escaped in airtable.url_table
