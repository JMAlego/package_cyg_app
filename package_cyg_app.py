"""Build "portable" application with dependencies from Cygwin.

This application takes a specified Cygwin package and downloads that package
and all it's dependencies into a minimal system tree for running that
package's applications. The resulting package does not require a separate
Cygwin install to function.

This application is not affiliated with the Cygwin project.
"""

# BSD 3-Clause License
#
# Copyright (c) 2022, Jacob Allen
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import platform
import sys
import tarfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass, field
from hashlib import sha256, sha512
from os import listdir, makedirs
from os import name as os_name
from os import path
from posixpath import basename as posix_basename
from shutil import copyfileobj, move, rmtree
from tempfile import TemporaryDirectory, TemporaryFile, gettempdir
from typing import Dict, Generic, Iterable, List, Literal, Optional, Sequence, Set, Tuple, TypeVar
from urllib import request
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit

__AUTHOR__ = "Jacob Allen"
__VERSION__ = "1.0.0"

ZST_SUPPORTED = False
try:
    import zstandard
    ZST_SUPPORTED = True
except ImportError:
    pass

Architecture = Literal["x86", "x86_64"]

DEFAULT_MIRROR = "https://www.mirrorservice.org/sites/sourceware.org/pub/cygwin/"
ARCHITECTURES: Tuple[Architecture, ...] = ("x86", "x86_64")
SETUP_FILE = "setup.ini"
CACHE_DIRECTORY = path.join(gettempdir(), ".cyg_build_app_cache")
EXEC_EXT = (".exe", ".bat", ".cmd")

PLATFORM_BITS, _ = platform.architecture()
DEFAULT_ARCHITECTURE: Architecture
if PLATFORM_BITS == "32bit":
    DEFAULT_ARCHITECTURE = "x86"
elif PLATFORM_BITS == "64bit":
    DEFAULT_ARCHITECTURE = "x86_64"
else:
    raise Exception("Cygwin only supports x86 and x86_64 architectures")

BUFFER_SIZE = 1024 * 1024 if os_name == 'nt' else 64 * 1024

LOGGING_FORMAT = '%(asctime)s - %(message)s'


class CachingService:
    """Provides cached web access."""

    def __init__(self,
                 cache_dir: str = CACHE_DIRECTORY,
                 mirror: str = DEFAULT_MIRROR,
                 debug: bool = False):
        """Initialize."""
        self.cache_dir = cache_dir
        self.mirror = mirror
        self._debug = debug

    def build_url(self, url_path: str) -> str:
        """Build a URL from a path and the mirror info in this service."""
        return urljoin(self.mirror, url_path)

    def get_url(self,
                url: str,
                expected_hash: str = "",
                expected_size: int = -1,
                force_download: bool = False) -> Optional[str]:
        """Fetch a URL, storing it in the cache."""
        url_hash = sha256(url.encode("utf-8")).hexdigest()

        split_url = urlsplit(url)
        file_name = posix_basename(split_url.path)

        local_path = path.join(self.cache_dir, url_hash, file_name)
        local_path = path.normpath(local_path)

        if self._debug:
            print(f"CachingService.get_url: {url} -> {local_path}", )

        if not path.isfile(local_path) or force_download:
            makedirs(path.dirname(local_path), exist_ok=True)
            try:
                request.urlretrieve(url, local_path)
                if self._debug:
                    print("CachingService.get_url: download complete")
            except HTTPError as error:
                if self._debug:
                    print("CachingService.get_url: error downloading file")
                    print(error)
                return None
        elif self._debug:
            print("CachingService.get_url: using cached file")

        if expected_size != -1:
            actual_size = path.getsize(local_path)

            if self._debug:
                print("Expected size", expected_size)
                print("Actual size", actual_size)

            if expected_size != actual_size:
                return None

        if expected_hash != "":
            with open(local_path, "rb") as file_to_hash:
                hasher = sha512()
                while True:
                    read_data = file_to_hash.read(BUFFER_SIZE)
                    if read_data == b"":
                        break
                    hasher.update(read_data)
                actual_hash = hasher.hexdigest()

            if self._debug:
                print("Expected hash", expected_hash)
                print("Actual hash", actual_hash)

            if expected_hash != actual_hash:
                return None

        return local_path


@dataclass
class PackageVersion:
    """Represents a particular version of a package."""

    version: str
    source: str
    install: Optional[str] = None
    depends2: List[str] = field(default_factory=list)
    build_depends: List[str] = field(default_factory=list)

    def get_install_url(self) -> Optional[Tuple[str, int, str]]:
        """Get the URL for the install files."""
        if self.install is None:
            return None
        install_path, bytes_count, hash_sum = self.install.split()
        url = urljoin(DEFAULT_MIRROR, install_path)
        return url, int(bytes_count), hash_sum

    @staticmethod
    def from_dict(version_dict: Dict[str, str]) -> PackageVersion:
        """Build a PackageVersion from a data dict."""
        return PackageVersion(
            version_dict["version"].strip(), version_dict["source"].strip(),
            version_dict["install"].strip() if "install" in version_dict else None,
            [x.strip()
             for x in version_dict["depends2"].split(",")] if "depends2" in version_dict else [],
            [x.strip() for x in version_dict["build-depends"].split(",")]
            if "build-depends" in version_dict else [])


@dataclass
class Package:
    """Represents a package."""

    package_name: str
    short_description: str
    long_description: str
    category: str
    requires: List[str]
    latest_version: str
    versions: Dict[str, PackageVersion]

    def get_latest_version(self) -> PackageVersion:
        """Get the latest PackageVersion of a package."""
        return self.versions[self.latest_version]

    @staticmethod
    def from_lines(lines: list[str]) -> Optional[Package]:
        """Build a package from it's lines in a setup file."""
        key: Optional[str] = None
        value_buffer = ""
        is_first_line: bool = True
        package_name: Optional[str] = None
        in_key = False
        properties: Dict[str, str] = {}
        versions: List[Dict[str, str]] = []
        in_version = True

        for number, line in enumerate(lines):
            if is_first_line:
                assert line.startswith("@")
                package_name = line[1:].strip()
            elif not in_key and line == "\n":
                continue
            elif line == "[prev]\n":
                if in_version:
                    versions.append(properties)
                properties = {}
                in_version = True
            elif line == "[test]\n":
                if in_version:
                    versions.append(properties)
                properties = {}
                in_version = False
            elif in_key:
                value_buffer += line
                in_key = value_buffer.count('"') % 2 != 0
            else:
                try:
                    key, value_buffer = line.split(":", 1)
                except ValueError:
                    print(number, line)
                    print(lines)
                    raise
                if value_buffer.count('"') % 2 != 0:
                    in_key = True
            if not in_key and not is_first_line:
                assert key is not None
                properties[key] = value_buffer
            is_first_line = False
        versions.append(properties)

        assert package_name is not None

        first, *_ = versions

        short_description = first["sdesc"].strip()[1:-1]
        del first["sdesc"]

        long_description = first["ldesc"].strip()[1:-1]
        del first["ldesc"]

        category = first["category"].strip()
        del first["category"]

        if "requires" in first:
            requires = [x.strip() for x in first["requires"].split()]
            del first["requires"]
        else:
            requires = []

        latest_version = first["version"].strip()

        return Package(
            package_name, short_description, long_description, category, requires, latest_version,
            {version["version"].strip(): PackageVersion.from_dict(version)
             for version in versions})


class PackageDatabase:
    """Provides querying of a package database."""

    def __init__(self, architecture: Architecture, cache: CachingService) -> None:
        self._arch = architecture
        self._cache = cache

        data_file = self._cache.get_url(self._cache.build_url(f"{architecture}/{SETUP_FILE}"),
                                        force_download=True)
        if data_file is None:
            raise Exception()

        with open(data_file, "r", encoding="utf-8") as file:
            self._data = file.readlines()

        self._package_bounds: Dict[str, Tuple[int, int]] = {}

        self._scan_packages()

    def _scan_packages(self) -> None:
        package_name: Optional[str] = None
        package_start = -1
        package_end = -1
        possibly_outside_of_package = True

        for line_number, line in enumerate(self._data):
            package_end = line_number
            if possibly_outside_of_package and line.startswith("@"):
                if package_name is not None:
                    self._package_bounds[package_name] = (package_start, package_end)
                package_name = line[1:].strip()
                package_start = line_number
                possibly_outside_of_package = False
            elif package_name is not None and line == "\n":
                possibly_outside_of_package = True
            elif package_name is not None:
                possibly_outside_of_package = False

        if package_name is not None:
            self._package_bounds[package_name] = (package_start, package_end + 1)

    def get_package(self, package_name: str) -> Optional[Package]:
        """Get package info."""
        if package_name not in self._package_bounds:
            return None

        start, end = self._package_bounds[package_name]
        return Package.from_lines(self._data[start:end])


T = TypeVar("T")


class OrderedSet(Generic[T]):
    """Ordered set for queuing tasks."""

    def __init__(self, iterable: Iterable[T] = None):
        """Initialise the ordered set."""
        self._set: Set[T] = set()
        self._list: List[T] = []

        if iterable is not None:
            self.add_all(iterable)

    def add_all(self, iterable: Iterable[T]):
        """Add all items from iterable."""
        for item in iterable:
            self.add(item)

    def add(self, item: T) -> None:
        """Add an item."""
        if item not in self._set:
            self._set.add(item)
            self._list.append(item)

    def pop(self) -> T:
        """Pop an item off the ordered set."""
        to_pop = self._list.pop()
        self._set.remove(to_pop)
        return to_pop

    def is_empty(self) -> bool:
        """Check if the ordered set is empty."""
        return not self._set


def decompress_file(source: str, destination: str, debug: bool = False):
    """Decompress a source file to a destination directory."""
    _, *extensions = source.split(".")

    if extensions[-2] == "tar" and extensions[-1] != "zst":
        if debug:
            print(f"Decompressing {source} -> {destination}")
        with tarfile.open(source, "r:*") as file:
            file.extractall(destination)

    if extensions[-2] == "tar" and extensions[-1] == "zst":
        if ZST_SUPPORTED:
            if debug:
                print(f"Decompressing {source} -> {destination}")
            with TemporaryFile() as intermediate_tar_file, open(source, "rb") as zst_file:
                zst_decompressor = zstandard.ZstdDecompressor()
                zst_reader = zst_decompressor.stream_reader(zst_file)
                copyfileobj(zst_reader, intermediate_tar_file)
                intermediate_tar_file.seek(0)
                with tarfile.TarFile(fileobj=intermediate_tar_file, ) as tar_file:
                    tar_file.extractall(destination)
        else:
            print(f"\r\n{path.basename(source)} skipped as no .zst support, install `zstandard`")


class Spinner:
    """Class to handle ASCII spinners."""

    ASCII_SPINNER = ("|", "/", "-", "\\")
    THIN_SPINNER = ("◜", "◝", "◞", "◟")
    DEFAULT_SPINNER = THIN_SPINNER

    def __init__(self,
                 topic: Optional[str] = None,
                 conclusion: Optional[str] = "Done",
                 spinner: Sequence[str] = DEFAULT_SPINNER,
                 debug: bool = False) -> None:
        """Initialise spinner."""
        self._spinner_index = 0
        self._last_length = 0
        self._spinner_states = spinner
        self._ended = False
        self._topic = topic
        self._conclusion = conclusion
        self._debug = debug

    def __enter__(self) -> Spinner:
        """Handle `with` enter."""
        self.start(self._topic)
        return self

    def __exit__(self, *_) -> None:
        """Handle `with` exit."""
        if not self._ended:
            self.stop(self._conclusion)

    def start(self, topic: Optional[str] = None) -> None:
        """Start the spinner."""
        if self._debug:
            print("Spinner started.")
            print(f"Spinner topic: {topic}")
            return

        if topic is None:
            topic = self._topic
        to_print = ""
        if topic is not None:
            to_print += topic + " "
        to_print += self._spinner_states[self._spinner_index]
        print(to_print, flush=True, end="")
        self._increment()

    def _increment(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_states)

    def step(self, message: Optional[str] = None) -> None:
        """Do a spinner step."""
        if self._debug:
            print(f"Spinner step message: {message}")
            return

        to_print = ("\b" * self._last_length + " " * self._last_length + "\b" +
                    "\b" * self._last_length)
        to_print += self._spinner_states[self._spinner_index]
        if message is not None:
            to_print += " " + message
            self._last_length = len(message) + 1
        print(to_print, flush=True, end="")
        self._increment()

    def stop(self, conclusion: Optional[str] = None) -> None:
        """Stop the spinner and finish it."""
        if self._debug:
            print("Spinner ended.")
            print(f"Spinner conclusion message: {conclusion}")
            return

        if conclusion is None:
            conclusion = self._conclusion
        self._ended = True
        to_print = ("\b" * self._last_length + " " * self._last_length + "\b" +
                    "\b" * self._last_length)
        if conclusion is not None:
            to_print += conclusion
            self._last_length = len(conclusion) + 1
        print(to_print, flush=True)


def find_requirements_for_package(target_package: str,
                                  package_database: PackageDatabase,
                                  debug: bool = False):
    """Find requirements for a target package."""
    packages_to_process: OrderedSet[str] = OrderedSet()
    packages_to_process.add(target_package)

    requirements: Set[str] = set()
    exclude_packages: Set[str] = set()
    processed_packages: Set[str] = set()

    requirements.add(target_package)

    with Spinner("Getting requirements...", debug=debug) as spinner:
        while not packages_to_process.is_empty():
            package_name = packages_to_process.pop()
            spinner.step(package_name[-40:])
            if package_name in exclude_packages and package_name != target_package:
                continue
            if package_name in processed_packages and package_name != target_package:
                continue
            processed_packages.add(package_name)
            if debug:
                print(f"Finding package dependencies: {package_name}")

            package = package_database.get_package(package_name)

            if package is None:
                if debug:
                    print(f"Skipping package and excluding: {package_name}")
                exclude_packages.add(package_name)
                continue

            depends = package.get_latest_version()

            packages_to_process.add_all(package.requires)
            packages_to_process.add_all(depends.depends2)
            requirements.update(package.requires)
            requirements.update(depends.depends2)

    requirements.difference_update(exclude_packages)

    return requirements


def download_and_extract_packages(target_package: str,
                                  destination_path: str,
                                  cyg_usr_bin_path: str,
                                  requirements: Iterable[str],
                                  package_database: PackageDatabase,
                                  cache: CachingService,
                                  debug: bool = False):
    """Download and extract a target package and it's requirements."""
    files_to_shim = set()

    with Spinner("Downloading and extracting...", debug=debug) as spinner:
        for requirement in requirements:
            spinner.step(requirement[-40:])

            if debug:
                print(f"Processing package dependencies: {requirement}")

            package = package_database.get_package(requirement)

            if package is None:
                continue

            downloaded_file = None

            for version_name, version in package.versions.items():
                if debug:
                    print(f"Trying version {version_name} for {requirement}.")

                version_info = version.get_install_url()
                if version_info is None:
                    continue

                download_url, download_size, download_hash = version_info

                downloaded_file = cache.get_url(download_url,
                                                expected_hash=download_hash,
                                                expected_size=download_size)
                if downloaded_file is not None:  # we found a working one
                    break

            if downloaded_file is None:  # we didn't find any working ones
                if debug:
                    print(f"Could not find working version of {requirement}!")
                continue

            if requirement == target_package:
                bin_before = set(listdir(cyg_usr_bin_path))

            decompress_file(downloaded_file, destination_path, debug=debug)

            if requirement == target_package:
                bin_after = set(listdir(cyg_usr_bin_path))
                files_to_shim = set(
                    filter(lambda x: x.endswith(EXEC_EXT), bin_after.difference(bin_before)))

    return files_to_shim


def fix_bin_and_lib_for_cygwin(cyg_bin_path: str,
                               cyg_usr_bin_path: str,
                               cyg_lib_path: str,
                               cyg_usr_lib_path: str,
                               debug: bool = False):
    """Replace the /bin and /lib path with their /usr version as a patch."""
    with Spinner("Fixing lib and bin location...", debug=debug) as spinner:
        for file_to_move in listdir(cyg_bin_path):
            spinner.step(f"moving {file_to_move[-33:]}")
            move(path.join(cyg_bin_path, file_to_move), path.join(cyg_usr_bin_path, file_to_move))

        for file_to_move in listdir(cyg_lib_path):
            spinner.step(f"moving {file_to_move[-33:]}")
            move(path.join(cyg_lib_path, file_to_move), path.join(cyg_usr_lib_path, file_to_move))

        spinner.step("removing /bin")
        rmtree(cyg_bin_path)

        spinner.step("removing /lib")
        rmtree(cyg_lib_path)

        spinner.step("moving /usr/bin to /bin")
        move(cyg_usr_bin_path, cyg_bin_path)

        spinner.step("moving /usr/lib to /lib")
        move(cyg_usr_lib_path, cyg_lib_path)


def generate_shim_files(files_to_shim: Iterable[str], destination_dir: str, debug: bool = False):
    """Generate cygwin shims for the specified binaries."""
    with Spinner("Generating shims...", debug=debug) as spinner:
        for file_to_shim in files_to_shim:
            spinner.step(file_to_shim[-40:])

            file_name, _ = path.splitext(file_to_shim)
            with open(path.join(destination_dir, f"{file_name}.bat"),
                      "w",
                      encoding="ascii",
                      newline="\r\n") as file:
                file.write("@setlocal\n")
                file.write("@set \"PATH=%~dp0bin;%~dp0usr\\bin\"\n")
                file.write("@set \"HOME=%~dp0home\\user\"\n")
                file.write("@set \"TERM=xterm\"\n")
                file.write(f"@%~dp0bin\\{file_to_shim} %*\n")
                file.write("@endlocal\n")


def pack_dir_to_tar_gz(source_dir: str,
                       destination_file_name: str,
                       destination_dir: str,
                       target_package: str,
                       debug: bool = False):
    """Package a directory to a .tar.gz file."""
    with Spinner("Packing...", debug=debug) as spinner:
        with TemporaryDirectory(prefix="cyg_build_app", suffix="compressing") as compressing_dir:
            with tarfile.open(path.join(compressing_dir, destination_file_name),
                              "w:gz") as result_tar:
                for item in listdir(source_dir):
                    spinner.step(item[-40:])
                    result_tar.add(path.join(source_dir, item),
                                   arcname=path.join(target_package, item),
                                   recursive=True)
            move(path.join(compressing_dir, destination_file_name),
                 path.join(destination_dir, destination_file_name))


def main() -> int:
    """Entry point for execution."""
    argument_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                                     description=__doc__)

    argument_parser.add_argument('--version',
                                 action='version',
                                 version=f'%(prog)s {__VERSION__} by {__AUTHOR__}')
    argument_parser.add_argument("TARGET_PACKAGE", type=str, help="target package")
    argument_parser.add_argument("--debug", action="store_true", help="enable debug messages")
    argument_parser.add_argument("-a",
                                 "--arch",
                                 choices=ARCHITECTURES,
                                 default=DEFAULT_ARCHITECTURE,
                                 help="architecture for packages")
    argument_parser.add_argument("-o", "--output", default="./", help="output location for package")
    argument_parser.add_argument("-m",
                                 "--mirror",
                                 default=DEFAULT_MIRROR,
                                 help="cygwin package mirror to use")
    argument_parser.add_argument("-d",
                                 "--list-dependencies",
                                 "--list-deps",
                                 action="store_true",
                                 help="list dependencies of package, then exit")

    parsed_arguments = argument_parser.parse_args()

    debug: bool = parsed_arguments.debug
    target_package: str = parsed_arguments.TARGET_PACKAGE
    target_architecture: Architecture = parsed_arguments.arch
    list_dependencies: bool = parsed_arguments.list_dependencies
    output_path: str = parsed_arguments.output

    output_file_name = f"{target_package}.tar.gz"
    output_directory_path = "./"

    if path.isdir(output_path):
        output_directory_path = output_path
    elif path.isdir(path.dirname(output_path)):
        output_directory_path = path.dirname(output_path)
        output_file_name = path.basename(output_path)
    else:
        print(f"Output directory {output_path} does not exist.")
        return 1

    print(f"Generating app package for '{target_package}'.")
    print(f"Using architecture '{target_architecture}'.")

    print("Initialising package database and caching services... ", end="", flush=True)

    cache = CachingService(debug=debug)
    package_database = PackageDatabase(target_architecture, cache)

    print("Done")

    if package_database.get_package(target_package) is None:
        print(f"Package '{target_package}' does not exist.")
        return 2

    requirements = find_requirements_for_package(target_package, package_database)

    if list_dependencies:
        print("Dependencies:")
        for dependency in sorted(requirements):
            print("  ->", dependency)
        return 0

    with TemporaryDirectory(prefix="cyg_build_app", suffix="working") as working_dir:
        cyg_bin_path = path.join(working_dir, "bin")
        cyg_lib_path = path.join(working_dir, "lib")
        cyg_usr_bin_path = path.join(working_dir, "usr", "bin")
        cyg_usr_lib_path = path.join(working_dir, "usr", "lib")
        cyg_home_path = path.join(working_dir, "home", "user")
        cyg_temp_path = path.join(working_dir, "tmp")

        makedirs(cyg_bin_path)
        makedirs(cyg_lib_path)
        makedirs(cyg_usr_bin_path)
        makedirs(cyg_usr_lib_path)
        makedirs(cyg_home_path)
        makedirs(cyg_temp_path)

        files_to_shim = download_and_extract_packages(target_package, working_dir, cyg_usr_bin_path,
                                                      requirements, package_database, cache, debug)

        fix_bin_and_lib_for_cygwin(cyg_bin_path, cyg_usr_bin_path, cyg_lib_path, cyg_usr_lib_path,
                                   debug)

        generate_shim_files(files_to_shim, working_dir, debug)

        pack_dir_to_tar_gz(working_dir, output_file_name, output_directory_path, target_package,
                           debug)

        print("Cleaning up... ", end="", flush=True)

        # exiting the with block cleans up temp files, so there will be a pause between this and the
        # next print below...

    print("Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
