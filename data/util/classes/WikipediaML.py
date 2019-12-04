import tensorflow as tf
from tensorflow_datasets.core import api_utils
import tensorflow_datasets as tfds
import xml.etree.cElementTree as etree
import apache_beam as beam
import mwparserfromhell
import urllib3
import shutil
import math
import json
import re
import os
import six
import time

# TF code produces warnings due to lazy implementation of urllib3 requests
# Can't fix this from the outside, so we'll ignore their mistakes
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

http = urllib3.PoolManager()

# Disables my print statements if you prefer minimal printing
ENABLE_CODE_FEEDBACK = False

# These constants are specified by using the WikipediaML class
LANGUAGE = None
DATE = None
DUMP_URL = None
STATUS_URL = None
STATUS_FILE = None
SPECIFIED_DOWNLOAD_DIRECTORY = None
VERSION_OVERRIDE = tfds.core.Version("0.0.0")


def feedback(message):
    if ENABLE_CODE_FEEDBACK:
        print(message)


def _parse_and_clean_wikicode(raw_content):
    # The parser from the original TF code
    wikicode = mwparserfromhell.parse(raw_content)

    # Filters for references, tables, and file/image links.
    re_rm_wikilink = re.compile("^(?:File|Image|Media):",
                                flags=re.IGNORECASE | re.UNICODE)

    def rm_wikilink(obj):
        return bool(re_rm_wikilink.match(six.text_type(obj.title)))

    def rm_tag(obj):
        return six.text_type(obj.tag) in {
            "ref",
            "table"
        }

    def rm_template(obj):
        return obj.name.lower() in {
            "reflist",
            "notelist",
            "notelist-ua",
            "notelist-lr",
            "notelist-ur",
            "notelist-lg"
        }

    def try_remove_obj(obj, section):
        try:
            section.remove(obj)
        except ValueError:
            # For unknown reasons, objects are sometimes not found.
            pass

    section_text = []

    # Filter individual sections to clean.
    for section in wikicode.get_sections(flat=True,
                                         include_lead=True,
                                         include_headings=True):
        for obj in section.ifilter_wikilinks(matches=rm_wikilink,
                                             recursive=True):
            try_remove_obj(obj, section)

        for obj in section.ifilter_templates(matches=rm_template,
                                             recursive=True):
            try_remove_obj(obj, section)

        for obj in section.ifilter_tags(matches=rm_tag,
                                        recursive=True):
            try_remove_obj(obj, section)

        section_text.append(section.strip_code().strip())

    return "\n\n".join(section_text)


class _CustomWikipediaConfig(tfds.core.BuilderConfig):
    @api_utils.disallow_positional_args
    def __init__(self, language=None, date=None, **kwargs):
        super(_CustomWikipediaConfig, self).__init__(
            name="{0}.{1}".format(date,
                                  language),
            description="Wikipedia dataset for {0}, parsed from {1} dump.".format(language,
                                                                                  date),
            **kwargs)

        self.date = date
        self.language = language


class CustomWikipedia(tfds.core.BeamBasedBuilder):
    BUILDER_CONFIG = _CustomWikipediaConfig(
        version=VERSION_OVERRIDE,
        language=LANGUAGE,
        date=DATE
    )

    VERSION = VERSION_OVERRIDE

    def __init__(self):
        # Force an override of the default download location
        super(CustomWikipedia, self).__init__(
            data_dir=SPECIFIED_DOWNLOAD_DIRECTORY)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "title": tfds.features.Text(),
                "text": tfds.features.Text(),
            }),
            supervised_keys=None,
            urls=["https://dumps.wikimedia.org"],
            citation="""\
                @ONLINE {wikidump,
                    author = "Wikimedia Foundation",
                    title  = "Wikimedia Downloads",
                    url    = "https://dumps.wikimedia.org"
                }""",
            redistribution_info={"license": (
                "This work is licensed under the Creative Commons Attribution-ShareAlike "
                "3.0 Unported License. To view a copy of this license, visit "
                "http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to "
                "Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.")})

    def _split_generators(self, dl_manager):
        xml_urls = []
        total_bytes = 0

        # Re-use the generated status.json
        with open(STATUS_FILE) as fh:
            dump_info = json.load(fh)

        multistream_dump_info = dump_info["jobs"]["articlesmultistreamdump"]

        assert multistream_dump_info["status"] == "done"

        for fname, info in multistream_dump_info["files"].items():
            if ".xml" not in fname:
                continue

            total_bytes += info["size"]
            xml_urls.append(DUMP_URL + fname)

        downloaded_files = dl_manager.download_and_extract({
            "xml": xml_urls
        })

        # Max 128MB
        max_bytes = int(math.ceil(total_bytes / (128 * 2**20)))

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=max_bytes,
                gen_kwargs={
                    "filepaths": downloaded_files["xml"],
                    "language": LANGUAGE
                })
        ]

    def _build_pcollection(self, pipeline, filepaths, language):
        def _extract_content(filepath):
            with tf.io.gfile.GFile(filepath) as f:
                for _, elem in etree.iterparse(f, events=("end",)):
                    if not elem.tag.endswith("page"):
                        continue

                    namespace = elem.tag[:-4]
                    title = elem.find("./{0}title".format(namespace)).text
                    ns = elem.find("./{0}ns".format(namespace)).text

                    # Filter pages that are not in the "main" namespace.
                    if ns != "0":
                        continue

                    raw_content = elem.find(
                        "./{0}revision/{0}text".format(namespace)
                    ).text

                    elem.clear()

                    # Filter redirects.
                    if raw_content is None or raw_content.lower().startswith("#redirect"):
                        beam.metrics.Metrics.counter(language,
                                                     "filtered-redirects").inc()
                        continue

                    beam.metrics.Metrics.counter(language,
                                                 "extracted-examples").inc()

                    yield (title, raw_content)

        def _clean_content(inputs):
            title, raw_content = inputs

            try:
                text = _parse_and_clean_wikicode(raw_content)
            except (mwparserfromhell.parser.ParserError) as e:
                beam.metrics.Metrics.counter(language, "parser-error").inc()
                return

            beam.metrics.Metrics.counter(language, "cleaned-examples").inc()

            yield {
                "title": title,
                "text": text
            }

        return (
            pipeline
            | beam.Create(filepaths)
            | beam.FlatMap(_extract_content)
            | beam.FlatMap(_clean_content)
        )


class WikipediaML():
    @api_utils.disallow_positional_args
    def __init__(self,
                 language=None,
                 date=None,
                 data_dir=None,
                 split=tfds.Split.TRAIN,
                 as_supervised=False,
                 batch_size=1,
                 shuffle_files=False,
                 code_messages=ENABLE_CODE_FEEDBACK):
        self._abs_dir = os.path.abspath(os.path.dirname(__file__))

        # Use preference of user
        global ENABLE_CODE_FEEDBACK
        ENABLE_CODE_FEEDBACK = code_messages

        self._language = language
        self._timestamp = date

        self._split = split
        self._as_supervised = as_supervised
        self._batch_size = batch_size
        self._shuffle_files = shuffle_files

        # Wikimedia urls using your.org mirror
        self._base_url = "https://dumps.wikimedia.your.org"
        self._dump_url = "{0}/{1}wiki/{2}/".format(self._base_url,
                                                   self._language,
                                                   self._timestamp)
        self._status_url = self._dump_url + "dumpstatus.json"

        self._download_path = os.path.join(self._abs_dir, data_dir)

        self._extract_path = os.path.join(self._download_path, "extracted")
        if not os.path.exists(self._extract_path):
            # Create relative extraction directory
            os.makedirs(self._extract_path)

        self._manual_path = os.path.join(self._download_path, "manual")
        if not os.path.exists(self._manual_path):
            # Create relative manual directory
            os.mkdir(self._manual_path)

        self._checksum_initial_file_path = os.path.join(self._download_path,
                                                        "_dump_manifest.txt")

        self._checksum_file_path = os.path.join(self._download_path,
                                                "custom_wikipedia.txt")

    def load(self, download=False, download_mode=tfds.GenerateMode.REUSE_DATASET_IF_EXISTS):
        # Show TF classes what dump is relevant
        # This code is disgusting...
        global LANGUAGE
        global DATE
        global DUMP_URL
        global STATUS_URL
        global SPECIFIED_DOWNLOAD_DIRECTORY

        LANGUAGE = self._language
        DATE = self._timestamp
        DUMP_URL = self._dump_url
        STATUS_URL = self._status_url
        SPECIFIED_DOWNLOAD_DIRECTORY = self._download_path

        self._builder = CustomWikipedia()

        self._pipeline_config = beam.options.pipeline_options.PipelineOptions()
        self._download_config = tfds.download.DownloadConfig(beam_options=self._pipeline_config,
                                                             extract_dir=self._extract_path,
                                                             manual_dir=self._manual_path,
                                                             download_mode=download_mode,
                                                             register_checksums=True)

        tfds.download.add_checksums_dir(self._download_path)

        if not os.path.exists(self._checksum_file_path):
            self._checksum_make()

            while not os.path.exists(self._checksum_file_path):
                # Wait on the checksum file generator...
                pass

        feedback("Loading {0} Wikipedia dump from {1}.\n".format(self._language,
                                                                 self._timestamp))
        feedback("This could take a while...\n")

        download_start = self._g_time()

        self._builder.download_and_prepare(download_dir=self._download_path,
                                           download_config=self._download_config)

        feedback("...done. Data prep took ~{0}mins.\n".format(
            self._g_minutes_elapsed(download_start)))

        feedback("Making TF Dataset...\n")

        dataset_start = self._g_time()

        self._tensorflow_dataset = self._builder.as_dataset(split=self._split,
                                                            batch_size=self._batch_size,
                                                            shuffle_files=self._shuffle_files,
                                                            as_supervised=self._as_supervised)

        feedback("...done. Dataset creation took ~{0}mins.\n".format(
            self._g_minutes_elapsed(dataset_start)))

        return self._tensorflow_dataset

    def _g_time(self):
        return time.time()

    def _g_minutes_elapsed(self, start):
        return int((self._g_time() - start) / 60)

    def _checksum_make(self):
        # This creates a checksum format file for the download to proceed
        # The complete file is moved to the main directory which then
        #  will trigger the download

        if os.path.exists(self._checksum_file_path):
            os.remove(self._checksum_file_path)

        if not os.path.exists(self._download_path):
            os.makedirs(self._download_path)

        global STATUS_FILE

        STATUS_FILE = os.path.join(self._download_path, "status.json")

        # Download the dumpstatus.json file from wikimedia
        with http.request('GET', self._status_url, preload_content=False) as r, open(STATUS_FILE, 'wb') as out_file:
            shutil.copyfileobj(r, out_file)

        try:
            with open(STATUS_FILE) as fh:
                dump_info = json.load(fh)
        except ValueError:
            print("Could not source the status JSON file.")
            exit(404)

        # Target the articles
        multistream_dump_info = dump_info["jobs"]["articlesmultistreamdump"]

        with open(self._checksum_initial_file_path, "w") as fh:
            for fname, info in multistream_dump_info["files"].items():
                if ".xml" not in fname:
                    continue

                # Write the checksum line
                fh.write("{0} {1} {2}\n".format(self._base_url + info["url"],
                                                info["size"],
                                                info["sha1"]))

        # Move the completed checksum
        os.rename(self._checksum_initial_file_path, self._checksum_file_path)


if __name__ == "__main__":
    # Example use case
    dataset = WikipediaML(language="en",
                          date=20190801,
                          data_dir="data/en_wikipedia").load()

    # Do dataset things...
    print(dataset)
