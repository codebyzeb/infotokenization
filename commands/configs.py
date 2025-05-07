HF_USERNAME = "InfoTokenizers"

# tokenizer repository
TOK_REPO_ID = "tokenizers"
BYTELEVEL_TOK_FOLDER = "bytelevel"


# data repository
FINEWEBEDU_REPO_ID = "finewebedu-20B"
COMMONCORPUS_REPO_ID = "common-corpus"
BYTE_DATA_FOLDER = "bytelevel"
BYTE_DATA_NGRAM_TRAINING = "bytelevel-subset"
BYTE_DATA_NGRAM_EXTRACTION = "bytelevel-subset_1"
BYTE_DATA_TOKENIZER_EVALUATION = "bytelevel-subset_2"
BYTE_LLM_PREDICTION_DATA = "bytelevel-llm-data"

# ngram model repository
BYTE_MODELS_REPO_ID = "bytelevel-models"
NGRAM_MODEL_FOLDER = "ngram"
BYTE_LLM_MODEL_FOLDER = "llm"

# ngram training
MAX_NGRAM_LENGTH = 5

# subset size for training byte-level models
NUM_TRAIN_ROWS = 100_000

# configs for splitting common corpus
TOKENS_PER_LANGUAGE = 10_000_000
LANGUAGES = ['English', 'Russian', 'Hungarian', 'Chinese', 'Indonesian',
             'Finnish', 'Arabic', 'Turkish', 'Korean', 'Basque',
             'Cebuano', 'French', 'Spanish', 'Czech', 'German',
             'Japanese', 'Maltese', 'Hebrew', 'Hindi', 'Catalan',
             'Italian', 'Polish', 'Portuguese', 'Danish', 'Vietnamese']