import json

import datasets
from datasets.tasks import LanguageModeling


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
"""

_DESCRIPTION = """\
SimplestCodeGeneration Dataset \
"""

_URL = "https://github.com/hzaskywalker/Concept"
_URLS = {
    "train": _URL + "train-v1.1.json",
    "dev": _URL + "dev-v1.1.json",
}


class SimpleConfig(datasets.BuilderConfig):
    """BuilderConfig for Simple."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SimpleConfig, self).__init__(**kwargs)


class Simple(datasets.GeneratorBasedBuilder):
    """SQUAD: The Stanford Question Answering Dataset. Version 1.1."""

    BUILDER_CONFIGS = [
        SimpleConfig(
            name="plain_code",
            version=datasets.Version("1.0.0", ""),
            description="Plain Code",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "input_ids": datasets.Value("string"),
                    "col_offsets": datasets.Value("string"),
                    "linenos": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/hzaskywalker/Concept",
            citation=_CITATION,
            task_templates=[
                LanguageModeling(
                    text_column="input_ids"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        # downloaded_files = dl_manager.download_and_extract(_URLS)
        from llm.pl.tester.integer import int_dsl as dsl
        from llm.pl.tester.integer import incr, decr, test
        from llm.learn.serialize import extract_code

        codes = [test, incr] #test, 
        data = [{k: ' '.join(v) for k, v in extract_code(i).serialize().items()} for i in codes]
        train = data[:2]
        valid = data[:1]

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"data": train}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"data": valid}),
        ]

    def _generate_examples(self, data):
        """This function returns the examples in the raw (text) form."""
        # logger.info("generating examples from = %s", filepath)
        for idx, val in enumerate(data):
            yield idx, val