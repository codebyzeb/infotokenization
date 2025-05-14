import copy
import json
from pathlib import Path

import pandas as pd
import tokenization_scorer
import typer
from scipy.stats import pearsonr
from tokenizers import models, normalizers, pre_tokenizers
from tqdm import tqdm
from transformers import AutoTokenizer

app = typer.Typer()


def filter_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "type" and value is not None}


class Token:
    def __init__(self, subword: str) -> None:
        self.value = subword


class BenchmarkTokenizer:
    def __init__(self, config_filepath: str | Path) -> None:
        self.config = self.load_tokenizer(config_filepath)
        self.normalizer = BenchmarkNormalizer(self.config["normalizer"])
        self.pre_tokenizer = BenchmarkPreTokenizer(self.config["pre_tokenizer"])
        self.model = BenchmarkModel(self.config["model"])

        model_config = self.config["model"]
        self.type = model_config["type"]
        if isinstance(model_config["vocab"], dict):
            self.vocab: dict[bytes, int] = model_config["vocab"]
        elif isinstance(model_config["vocab"], list):
            # this is the case of a unigram based vocab where each inner list is (token,likelihood)
            self.vocab = {ls[0]: idx for idx, ls in enumerate(model_config["vocab"])}
        self.inv_vocab: dict[int, bytes] = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def tokenize(self, text: str) -> list[str]:
        normalized_text = self.normalizer.normalize_str(text)
        pre_tokenized_text = self.pre_tokenizer.pre_tokenize_str(normalized_text)
        tokens = []
        for word, offset in pre_tokenized_text:
            tokens.extend(map(lambda tok: tok.value, self.model.tokenize(word)))
        if self.get_type() == "WP_equal_like":
            tokens = list(map(lambda tok: "##" + tok, tokens))
            tokens[0] = tokens[0][2:]
        return tokens

    def get_vocab(self) -> dict[bytes, int]:
        return self.vocab

    def is_byte_level(self) -> bool:
        return self.pre_tokenizer.byte_level

    def get_type(self) -> str:
        return self.model.type

    def load_tokenizer(self, config_filepath: str | Path) -> dict:
        """Read our hex formatted vocab file and return a list of bytes objects.

        Input file has one vocab word per line each hex encoded.
        """
        config_path = Path(config_filepath)
        if not config_path.exists():
            raise FileNotFoundError(f"Missing vocab file: {config_filepath}")

        with config_path.open() as config_file:
            tokenizer_config = json.load(config_file)
        return tokenizer_config


class BenchmarkNormalizer:
    def __init__(self, config_normalizer: dict) -> None:
        self.backend_normalizer = None
        if config_normalizer:
            if config_normalizer["type"] == "Sequence":
                hf_normalizers = [self.get_hf_normalizer(config) for config in config_normalizer["normalizers"]]
                self.backend_normalizer = normalizers.Sequence(hf_normalizers)  # type: ignore
            else:
                self.backend_normalizer = self.get_hf_normalizer(config_normalizer)

    def normalize_str(self, sequence: str) -> str:
        """
        Normalize the given string

        This method provides a way to visualize the effect of a Normalizer,
         but it does not keep track of the alignment information.
         If you need to get/convert offsets, you can use normalize()
        :param normalized:
        :return:
        """
        if self.backend_normalizer:
            return self.backend_normalizer.normalize_str(sequence)
        else:
            return sequence

    def get_hf_normalizer(self, normalizer_config: dict) -> normalizers.Normalizer | None:
        normalizer_map = {
            "BertNormalizer": normalizers.BertNormalizer,
            "Lowercase": normalizers.Lowercase,
            "NFC": normalizers.NFC,
            "NFD": normalizers.NFD,
            "NFKC": normalizers.NFKC,
            "NFKD": normalizers.NFKD,
            "Nmt": normalizers.Nmt,
            "Precompiled": lambda: normalizers.Precompiled(bytes(normalizer_config["precompiled_charsmap"], "utf-32")),
            "Replace": normalizers.Replace,
            "Strip": normalizers.Strip,
            "StripAccents": normalizers.StripAccents,
        }
        normalizer_class = normalizer_map.get(normalizer_config["type"])
        return normalizer_class(**filter_config(normalizer_config)) if normalizer_class else None


class BenchmarkPreTokenizer:
    def __init__(self, pretokenizer_config: dict) -> None:
        self.byte_level = False
        if pretokenizer_config:
            if pretokenizer_config["type"] == "Sequence":
                hf_pretokenizers = [self.get_hf_pretokenizer(config) for config in pretokenizer_config["pretokenizers"]]
                for config in pretokenizer_config["pretokenizers"]:
                    self.byte_level = self.byte_level or config["type"] == "ByteLevel"
                self.backend_pretokenizer = pre_tokenizers.Sequence(hf_pretokenizers)
            else:
                self.backend_pretokenizer = self.get_hf_pretokenizer(pretokenizer_config)
                self.byte_level = pretokenizer_config["type"] == "ByteLevel"
        else:
            self.backend_pretokenizer = None

    def pre_tokenize_str(self, sequence: str) -> list[str]:
        """Pre-tokenize the given string.

        This method provides a way to visualize the effect of a PreTokenizer, but it does not keep track
        of the alignment, nor does it provide all the capabilities of the PreTokenizedString. If you need
        some of these, you can use pre_tokenize().
        """
        if self.backend_pretokenizer:
            return self.backend_pretokenizer.pre_tokenize_str(sequence)
        else:
            return sequence.split()

    def get_hf_pretokenizer(self, pretokenizer_config: dict) -> pre_tokenizers.PreTokenizer | None:
        pretokenizer_map = {
            "BertPreTokenizer": lambda: pre_tokenizers.BertPreTokenizer(),
            "ByteLevel": lambda: pre_tokenizers.ByteLevel(**filter_config(pretokenizer_config)),
            "CharDelimiterSplit": lambda: pre_tokenizers.CharDelimiterSplit(),
            "Digits": lambda: pre_tokenizers.Digits(**filter_config(pretokenizer_config)),
            "Metaspace": lambda: pre_tokenizers.Metaspace(**filter_config(pretokenizer_config)),
            "Punctuation": lambda: pre_tokenizers.Punctuation(**filter_config(pretokenizer_config)),
            "Split": lambda: pre_tokenizers.Split(
                pattern=pretokenizer_config["pattern"]["Regex"],
                behavior=pretokenizer_config["behavior"].lower(),
                **filter_config(pretokenizer_config),
            ),
            "UnicodeScripts": lambda: pre_tokenizers.UnicodeScripts(),
            "Whitespace": lambda: pre_tokenizers.Whitespace(),
            "WhitespaceSplit": lambda: pre_tokenizers.WhitespaceSplit(),
        }
        pretokenizer_type = pretokenizer_config["type"]
        return pretokenizer_map[pretokenizer_type]() if pretokenizer_type in pretokenizer_map else None


class BenchmarkModel:
    def __init__(self, model_config: dict) -> None:
        self.type = model_config["type"]
        model_config = copy.deepcopy(model_config)
        match model_config.pop("type"):
            case "BPE":
                model_config["merges"] = tuple(tuple(s) for s in model_config["merges"])
                if not model_config["continuing_subword_prefix"]:
                    model_config["continuing_subword_prefix"] = ""
                if not model_config["end_of_word_suffix"]:
                    model_config["end_of_word_suffix"] = ""
                self.backend_model = models.BPE(**model_config)
            case "BPE_dropout":
                model_config["merges"] = tuple(tuple(s) for s in model_config["merges"])
                if not model_config["continuing_subword_prefix"]:
                    model_config["continuing_subword_prefix"] = ""
                if not model_config["end_of_word_suffix"]:
                    model_config["end_of_word_suffix"] = ""
                self.backend_model = models.BPE(**model_config)
            case "WordPiece":
                self.backend_model = models.WordPiece(**model_config)
            case "WordLevel":
                self.backend_model = models.WordLevel(**model_config)
            case "Sage":
                self.backend_model = models.WordPiece(**model_config)
            case "Greedy_Unigram":
                self.backend_model = models.WordPiece(**model_config)
            case "Greedy_BPE":
                self.backend_model = models.WordPiece(**model_config)
            case "Unigram":
                model_config["vocab"] = tuple(tuple(ls) for ls in model_config["vocab"])
                self.backend_model = models.Unigram(**model_config)
            case "SaGe_as_Unigram":
                model_config["vocab"] = tuple(tuple(ls) for ls in model_config["vocab"])
                self.backend_model = models.Unigram(**model_config)
            case "Unigram_equal_like":
                model_config["vocab"] = tuple(tuple(ls) for ls in model_config["vocab"])
                self.backend_model = models.Unigram(**model_config)
            case "BPE_equal_like":
                model_config["vocab"] = tuple(tuple(ls) for ls in model_config["vocab"])
                self.backend_model = models.Unigram(**model_config)
            case "SaGe_equal_like":
                model_config["vocab"] = tuple(tuple(ls) for ls in model_config["vocab"])
                self.backend_model = models.Unigram(**model_config)
            case "WP_equal_like":
                model_config["vocab"] = tuple(tuple(ls) for ls in model_config["vocab"])
                self.backend_model = models.Unigram(**model_config)
            case "flota":
                # unigram
                if isinstance(model_config["vocab"], list):
                    vocab = tuple(tuple(ls) for ls in model_config["vocab"])
                    model_config["vocab"] = {tok[0]: i for i, tok in enumerate(vocab)}
                self.backend_model = FlotaTokenizer(model_config["vocab"])
            case "WP_flota":
                # unigram
                if isinstance(model_config["vocab"], list):
                    vocab = tuple(tuple(ls) for ls in model_config["vocab"])
                    model_config["vocab"] = {tok[0]: i for i, tok in enumerate(vocab)}
                self.backend_model = FlotaTokenizer(model_config["vocab"], special="##")
            case "longest_suffix":
                if isinstance(model_config["vocab"], list):
                    vocab = tuple(tuple(ls) for ls in model_config["vocab"])
                    model_config["vocab"] = {tok[0]: i for i, tok in enumerate(vocab)}
                self.backend_model = LongestSuffix(model_config["vocab"])
            case "WP_longest_suffix":
                if isinstance(model_config["vocab"], list):
                    vocab = tuple(tuple(ls) for ls in model_config["vocab"])
                    model_config["vocab"] = {tok[0]: i for i, tok in enumerate(vocab)}
                self.backend_model = LongestSuffix(model_config["vocab"], special="##")

    def tokenize(self, sequence: str) -> list[Token]:
        return self.backend_model.tokenize(sequence)


class FlotaTokenizer:
    def __init__(self, vocab: dict, special: str = "Ġ") -> None:
        self.vocab = vocab
        self.special = special

    def max_subword_split(self, w: str) -> tuple:
        for l in range(len(w), 0, -1):
            for i in range(0, len(w) - l + 1):
                if w[i] == "\u2581":
                    continue
                subword = w[i : i + l]
                if self.special == "Ġ":
                    if subword in self.vocab:
                        return subword, w[:i] + l * "\u2581" + w[i + l :], i
                else:
                    if i == 0:
                        if subword in self.vocab:
                            return subword, w[:i] + l * "\u2581" + w[i + l :], i
                    else:
                        if (self.special + subword) in self.vocab:
                            return self.special + subword, w[:i] + l * "\u2581" + w[i + l :], i
        return None, None, None

    def get_flota_dict(self, w: str) -> dict:
        max_subword, rest, i = self.max_subword_split(w)
        if max_subword is None:
            return dict()
        if rest == len(rest) * "\u2581":
            flota_dict = {i: max_subword}
            return flota_dict
        flota_dict = self.get_flota_dict(rest)
        flota_dict[i] = max_subword
        return flota_dict

    def tokenize(self, w: str) -> list[Token]:
        flota_dict = self.get_flota_dict(w)
        return [Token(subword) for i, subword in sorted(flota_dict.items())]


class LongestSuffix:
    def __init__(self, vocab: dict, special: str = "Ġ") -> None:
        self.vocab = vocab
        self.special = special

    def tokenize(self, w: str) -> list[Token]:
        tokens = []
        i = 0
        while w and i < len(w):
            if self.special == "Ġ":
                if w[i:] in self.vocab:
                    tokens.insert(0, w[i:])
                    w = w[:i]
                    i = 0
                else:
                    i += 1
            else:
                if i == 0:
                    if w[i:] in self.vocab:
                        tokens.insert(0, w[i:])
                        w = w[:i]
                        i = 0
                    else:
                        i += 1
                else:
                    if self.special + w[i:] in self.vocab:
                        tokens.insert(0, self.special + w[i:])
                        w = w[:i]
                        i = 0
                    else:
                        i += 1
        return [Token(token) for token in tokens]


class Evaluator:
    def __init__(
        self,
        test_corpus_path: str | Path = "minipile.txt",
        test_combined_path: str | Path = "combined_resources.csv",
        test_cog_path: str | Path = "cog.csv",
    ) -> None:
        self.TEST_CORPUS = Path(test_corpus_path)
        self.COMBINED_CORPUS = Path(test_combined_path)
        self.COG_CORPUS = Path(test_cog_path)

        for path in [self.TEST_CORPUS, self.COMBINED_CORPUS, self.COG_CORPUS]:
            assert path.exists(), f"Missing file: {path}"

    def eval_tokenizer(self, tokenizer: BenchmarkTokenizer, special: str, all_tokenizers, compare) -> dict:
        metrics = {}
        metrics.update({"type": tokenizer.get_type()})

        # Static metrics
        corpus = self.corpus_to_list(self.TEST_CORPUS)
        metrics.update(self.tokenization_scorer(tokenizer, corpus))

        # # Linguistic metrics
        # metrics.update(self.combined_coverage(self.COMBINED_CORPUS, tokenizer, special))

        # # human metrics
        # metrics.update(self.eval_cog(self.COG_CORPUS, tokenizer))

        # # comparative measures
        # if compare and tokenizer == all_tokenizers[0]:
        #     # This function doesn't return a value and just prints the segmentation difference once
        #     # This counts on the fact that the vocabulary is the same for all tokenizers
        #     # And that the default inference is the first tokenizer
        #     self.segmentation_diff(all_tokenizers[0], all_tokenizers[1:], corpus, special)

        return metrics

    def corpus_to_list(self, file_path: str | Path, encoding: str = "utf-8") -> list[str]:
        file_path = Path(file_path)
        with file_path.open(encoding=encoding) as file:
            corpus = [line.strip() for line in file.readlines()]
        return corpus

    @staticmethod
    def tokenization_scorer(tokenizer, corpus: list[str]) -> dict[str, float]:
        """Calculate the fertility and entropy."""
        tokenized_corpus = [tokenizer.tokenize(text) for text in corpus]

        num_of_tokens = sum([len(tokenized_sentence) for tokenized_sentence in tokenized_corpus])
        num_of_words = sum(len(sentence.split(" ")) for sentence in corpus)

        out = {}
        for metric in [
            "renyi_efficiency"
            # "renyi_entropy",
            # "shannon_efficiency",
            # "shannon_entropy",
            # "bits",
            # "seq_len",
            # "doc_len",
            # "perc_freq",
        ]:
            kwargs = {"power": 2.5} if metric == "renyi_efficiency" else {}
            out[metric] = tokenization_scorer.score(tokenized_corpus, metric=metric, **kwargs)

        return {"fertility": num_of_tokens / num_of_words, **out}

    @staticmethod
    def combined_coverage(combined_path: str | Path, tokenizer, special) -> dict:
        """Evaluate the coverage of the tokenizer using the combined resources dataset."""
        df = pd.read_csv(combined_path, sep=",")

        def get_boundaries(tokenization: list[str]) -> list[int]:
            return [len("".join(tokenization[:i])) for i in range(1, len(tokenization))]

        def get_seg_coverage(x, tokenizer, key_in_df, special: str) -> dict[str, float]:
            tps = fps = fns = length = count = 0
            for _, row in x.iterrows():
                # Gold standard morphological segmentation from the dataset
                gstandard = row["Gold_standard_segmentation"]
                gstandard[0] = "Ġ" + gstandard[0]
                if "".join(gstandard) not in tokenizer.get_vocab():
                    if special == "##":
                        gstandard = list(map(lambda tok: "##" + tok, gstandard))
                        gstandard[0] = gstandard[0][2:]

                    if all(token in tokenizer.get_vocab() for token in gstandard):
                        count += 1
                        # Tokenise the compound with the given tokeniser
                        y = [x for x in tokenizer.tokenize(row[key_in_df])]
                        # Get the boundaries for the gold standard and the tokeniser
                        gstandard_boundaries = get_boundaries(gstandard)
                        y_boundaries = get_boundaries(y)
                        fn = 0
                        for i in y_boundaries:
                            if i in gstandard_boundaries:
                                # True positives are those appearing in both generated and reference
                                tps += 1
                            else:
                                # False positives are those appearing in the generated but not the reference
                                fps += 1
                        for i in gstandard_boundaries:
                            if i not in y_boundaries:
                                # False negatives are those appearing in the reference but not the generated
                                fn += 1
                        fns += fn
                        length += len(y)

            f1 = tps / (tps + 0.5 * (fps + fns))
            return {"f1": f1}

        datasets = ["Ladec", "MorphoLex", "MorphyNet", "Dago_Bert", "UniMorph", "UnBlend", "CompoundPiece"]
        coverage = {}
        avg_f1 = 0
        for dataset in datasets:
            curr_coverage = get_seg_coverage(df.loc[df["Origin"] == dataset], tokenizer, "Word", special)
            avg_f1 += curr_coverage["f1"]
            curr_coverage = {dataset + "_" + key: val for key, val in curr_coverage.items()}
            coverage.update(curr_coverage)
        coverage["avg_f1"] = avg_f1 / len(datasets)
        return coverage

    @staticmethod
    def eval_cog(cog_path: str | Path, tokenizer) -> dict:
        """Evaluate the cognitive scores of the tokenizer using the Cog dataset."""
        cog_data = pd.read_csv(cog_path)
        cog_data = cog_data.dropna()
        words = cog_data[cog_data["lexicality"] == "W"]
        nonwords = cog_data[cog_data["lexicality"] == "N"]

        datasets = {"words": words, "nonwords": nonwords}
        all_results = {}
        avg_corr = 0
        for category, dataset in datasets.items():
            # measurements
            words = list(dataset["spelling"])
            rts = list(dataset["rt"])
            accs = list(dataset["accuracy"])

            # splits in model output
            tokens = list([tokenizer.tokenize(word) for word in words])
            wordiness = [1 - (len(tokens[i]) / len(str(words[i]))) for i in range(len(dataset))]

            # correlation
            corr1 = pearsonr(wordiness, rts).correlation
            corr2 = pearsonr(wordiness, accs).correlation

            category_results = {category + "_chunkability_rts": corr1, category + "_chunkability_accs": corr2}
            avg_corr += abs(corr1)
            avg_corr += abs(corr2)
            all_results.update(category_results)

        all_results["cog_score"] = avg_corr / 4
        return all_results

    @staticmethod
    def segmentation_diff(default_tokenizer, others, corpus, special):
        corpus_str = "".join(corpus)
        splitted_text = corpus_str.split()
        default_tokenized_corpus = [default_tokenizer.tokenize(pre_token) for pre_token in splitted_text]
        tokenized_corpuses = [[tokenizer.tokenize(pre_token) for pre_token in splitted_text] for tokenizer in others]
        for i, tokenized_corpus in enumerate(tokenized_corpuses):
            tokenizer = others[i]
            diff = total = 0
            for default_tokenization, tokenization in zip(default_tokenized_corpus, tokenized_corpus, strict=False):
                if default_tokenization != tokenization:
                    if special == "##" and "equal" in tokenizer.get_type():
                        default_tokenization = list(
                            map(lambda tok: "##" + tok if not tok.startswith("##") else tok, default_tokenization)
                        )
                        default_tokenization[0] = default_tokenization[0][2:]
                    if default_tokenization != tokenization:
                        diff += 1
                total += 1
            print(f"for tokenizer {tokenizer.get_type()} the diff is {diff / total}")


@app.command()
def intrinsic(tok_list_path: str, tests_path: str = "./data/tok_eval", compare: bool = False) -> None:
    with Path(tok_list_path).open() as vocabs_file:
        paths = [Path(path.strip()) for path in vocabs_file.readlines()]

    tokenizers = [BenchmarkTokenizer(path) for path in paths]

    p = Path(tests_path)
    evaluator = Evaluator(p / "minipile.txt", p / "combined_resources.csv", p / "cog.csv")

    results = []
    iterable = list(zip(paths, tokenizers, strict=True))
    for path, tokenizer in tqdm(iterable, desc="Evaluating Tokenizers"):
        special = (
            "##"
            if any(path.name.startswith(prefix) for prefix in ["wordpiece", "flota_wordpiece", "suffix_wordpiece"])
            else "Ġ"
        )

        result = {
            "tokenizer": path.name.rstrip(".json"),
            **evaluator.eval_tokenizer(
                tokenizer=tokenizer, special=special, all_tokenizers=tokenizers, compare=compare
            ),
        }
        results.append(result)

    df = pd.DataFrame(results).round(4)
    df.to_csv("tok_eval_results.csv", index=False)


@app.command()
def intrinsic_other(tok_list_path: str) -> None:
    with Path(tok_list_path).open() as tok_list_file:
        toks = [
            AutoTokenizer.from_pretrained("InfoTokenizers/tokenizers", subfolder=tok)
            for tok in tok_list_file.readlines()
        ]
