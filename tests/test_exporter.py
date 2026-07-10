"""Unit tests for the Data Exporter."""

import os
import tempfile
import unittest
import json
import csv
from crawl.exporter import to_csv, to_json, export_csv, export_json, export_txt
from crawl.exceptions import ExportError


class TestExporter(unittest.TestCase):
    """Verifies that exporter correctly formats and saves data to CSV, JSON, and TXT files."""

    def setUp(self) -> None:
        self.test_data = [
            {"title": "Page 1", "url": "https://example.com/1"},
            {"title": "Page 2", "url": "https://example.com/2"}
        ]
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_to_csv(self) -> None:
        """Tests converting a list of dicts to CSV string."""
        csv_str = to_csv(self.test_data)
        self.assertIn("title,url", csv_str)
        self.assertIn("Page 1,https://example.com/1", csv_str)
        self.assertIn("Page 2,https://example.com/2", csv_str)

        # Empty data handling
        self.assertEqual(to_csv([]), "")

    def test_to_json(self) -> None:
        """Tests converting data to JSON string."""
        json_str = to_json(self.test_data)
        data = json.loads(json_str)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["title"], "Page 1")

        # Serializing invalid structures should raise ExportError
        with self.assertRaises(ExportError):
            to_json(set([1, 2, 3]))  # sets are not JSON serializable

    def test_export_csv_file(self) -> None:
        """Tests exporting to CSV file and verifying contents."""
        path = os.path.join(self.temp_dir.name, "output.csv")
        export_csv(self.test_data, path)

        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["title"], "Page 1")
            self.assertEqual(rows[1]["url"], "https://example.com/2")

    def test_export_json_file(self) -> None:
        """Tests exporting to JSON file and verifying contents."""
        path = os.path.join(self.temp_dir.name, "output.json")
        export_json(self.test_data, path)

        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[1]["title"], "Page 2")

    def test_export_txt_file(self) -> None:
        """Tests exporting string/list of strings to TXT file."""
        # Single string
        path_str = os.path.join(self.temp_dir.name, "output_str.txt")
        export_txt("Hello World\nLine 2", path_str)
        self.assertTrue(os.path.exists(path_str))
        with open(path_str, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "Hello World\nLine 2")

        # List of strings
        path_list = os.path.join(self.temp_dir.name, "output_list.txt")
        export_txt(["Line 1", "Line 2"], path_list)
        self.assertTrue(os.path.exists(path_list))
        with open(path_list, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "Line 1\nLine 2\n")


if __name__ == "__main__":
    unittest.main()
