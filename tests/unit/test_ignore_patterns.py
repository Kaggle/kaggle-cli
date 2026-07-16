import os
import shutil
import tempfile
import unittest
import zipfile
import tarfile

from kaggle.api.kaggle_api_extended import should_ignore, DirectoryArchive, DEFAULT_IGNORE_PATTERNS


class TestIgnorePatterns(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_should_ignore_default_patterns(self):
        # .git/ at root should be ignored
        self.assertTrue(should_ignore(".git", True, DEFAULT_IGNORE_PATTERNS))
        self.assertTrue(should_ignore(".git/", True, DEFAULT_IGNORE_PATTERNS))

        # .git/ in sub dir should be ignored (due to */.git/)
        self.assertTrue(should_ignore("sub/.git", True, DEFAULT_IGNORE_PATTERNS))
        self.assertTrue(should_ignore("sub/.git/", True, DEFAULT_IGNORE_PATTERNS))
        self.assertTrue(should_ignore("sub/dir/.git", True, DEFAULT_IGNORE_PATTERNS))

        # .git file (not dir) should NOT be ignored
        self.assertFalse(should_ignore(".git", False, DEFAULT_IGNORE_PATTERNS))
        self.assertFalse(should_ignore("sub/.git", False, DEFAULT_IGNORE_PATTERNS))

        # .cache/ at root should be ignored
        self.assertTrue(should_ignore(".cache", True, DEFAULT_IGNORE_PATTERNS))

        # .cache/ in sub dir should NOT be ignored (no */.cache/ in defaults)
        self.assertFalse(should_ignore("sub/.cache", True, DEFAULT_IGNORE_PATTERNS))

        # .huggingface/ at root should be ignored
        self.assertTrue(should_ignore(".huggingface", True, DEFAULT_IGNORE_PATTERNS))

        # .huggingface/ in sub dir should NOT be ignored
        self.assertFalse(should_ignore("sub/.huggingface", True, DEFAULT_IGNORE_PATTERNS))

    def test_should_ignore_custom_patterns(self):
        patterns = ["*.tmp", "ignore_dir/", "*/nested_ignore_dir/", "exact_file.txt"]

        # Wildcard file pattern
        self.assertTrue(should_ignore("foo.tmp", False, patterns))
        self.assertTrue(should_ignore("sub/foo.tmp", False, patterns))
        self.assertFalse(should_ignore("foo.tmp.bak", False, patterns))

        # Directory pattern (root only)
        self.assertTrue(should_ignore("ignore_dir", True, patterns))
        self.assertFalse(should_ignore("sub/ignore_dir", True, patterns))
        # Directory pattern (root only) should not match file
        self.assertFalse(should_ignore("ignore_dir", False, patterns))

        # Directory pattern (nested)
        self.assertTrue(should_ignore("sub/nested_ignore_dir", True, patterns))
        self.assertTrue(should_ignore("sub/dir/nested_ignore_dir", True, patterns))
        self.assertFalse(should_ignore("nested_ignore_dir", True, patterns))

        # Exact file pattern
        self.assertTrue(should_ignore("exact_file.txt", False, patterns))
        # fnmatch "exact_file.txt" will not match "sub/exact_file.txt" if we don't have wildcard
        # Wait, fnmatch("sub/exact_file.txt", "exact_file.txt") is False.
        self.assertFalse(should_ignore("sub/exact_file.txt", False, patterns))

    def test_directory_archive_zip_filtering(self):
        # Create dummy directory structure
        src_dir = os.path.join(self.temp_dir, "src")
        os.makedirs(src_dir)
        os.makedirs(os.path.join(src_dir, ".git"))
        os.makedirs(os.path.join(src_dir, "sub"))
        os.makedirs(os.path.join(src_dir, "sub", ".git"))
        os.makedirs(os.path.join(src_dir, "ignore_dir"))

        # Create files
        self._create_file(os.path.join(src_dir, "keep.txt"))
        self._create_file(os.path.join(src_dir, ".git", "config"))
        self._create_file(os.path.join(src_dir, "sub", "keep2.txt"))
        self._create_file(os.path.join(src_dir, "sub", ".git", "config"))
        self._create_file(os.path.join(src_dir, "sub", "temp.tmp"))
        self._create_file(os.path.join(src_dir, "ignore_dir", "file.txt"))

        patterns = DEFAULT_IGNORE_PATTERNS + ["*.tmp", "ignore_dir/"]

        # Archive
        with DirectoryArchive(src_dir, "zip", ignore_patterns=patterns) as ar:
            archive_path = ar.path
            self.assertTrue(os.path.exists(archive_path))

            # Extract and verify
            extract_dir = os.path.join(self.temp_dir, "extract")
            os.makedirs(extract_dir)
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # Check files
            self.assertTrue(os.path.exists(os.path.join(extract_dir, "keep.txt")))
            self.assertTrue(os.path.exists(os.path.join(extract_dir, "sub", "keep2.txt")))

            # Checked ignored
            self.assertFalse(os.path.exists(os.path.join(extract_dir, ".git")))
            self.assertFalse(os.path.exists(os.path.join(extract_dir, "sub", ".git")))
            self.assertFalse(os.path.exists(os.path.join(extract_dir, "sub", "temp.tmp")))
            self.assertFalse(os.path.exists(os.path.join(extract_dir, "ignore_dir")))

    def test_directory_archive_tar_filtering(self):
        # Create dummy directory structure
        src_dir = os.path.join(self.temp_dir, "src")
        os.makedirs(src_dir)
        os.makedirs(os.path.join(src_dir, ".git"))
        os.makedirs(os.path.join(src_dir, "sub"))
        os.makedirs(os.path.join(src_dir, "sub", ".git"))
        os.makedirs(os.path.join(src_dir, "ignore_dir"))

        # Create files
        self._create_file(os.path.join(src_dir, "keep.txt"))
        self._create_file(os.path.join(src_dir, ".git", "config"))
        self._create_file(os.path.join(src_dir, "sub", "keep2.txt"))
        self._create_file(os.path.join(src_dir, "sub", ".git", "config"))
        self._create_file(os.path.join(src_dir, "sub", "temp.tmp"))
        self._create_file(os.path.join(src_dir, "ignore_dir", "file.txt"))

        patterns = DEFAULT_IGNORE_PATTERNS + ["*.tmp", "ignore_dir/"]

        # Archive
        with DirectoryArchive(src_dir, "tar", ignore_patterns=patterns) as ar:
            archive_path = ar.path
            self.assertTrue(os.path.exists(archive_path))

            # Extract and verify
            extract_dir = os.path.join(self.temp_dir, "extract")
            os.makedirs(extract_dir)
            with tarfile.open(archive_path, "r") as tar_ref:
                tar_ref.extractall(extract_dir)

            # Check files
            self.assertTrue(os.path.exists(os.path.join(extract_dir, "keep.txt")))
            self.assertTrue(os.path.exists(os.path.join(extract_dir, "sub", "keep2.txt")))

            # Checked ignored
            self.assertFalse(os.path.exists(os.path.join(extract_dir, ".git")))
            self.assertFalse(os.path.exists(os.path.join(extract_dir, "sub", ".git")))
            self.assertFalse(os.path.exists(os.path.join(extract_dir, "sub", "temp.tmp")))
            self.assertFalse(os.path.exists(os.path.join(extract_dir, "ignore_dir")))

    def _create_file(self, path):
        with open(path, "w") as f:
            f.write("dummy content")
