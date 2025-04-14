# Script used to copy the trained byte-level LLM to our huggingface repo

git clone https://huggingface.co/InfoTokenizers/fw57M-tied_finewebedu-20B_bytelevel
cd fw57M-tied_finewebedu-20B_bytelevel/
git checkout step50000
huggingface-cli upload InfoTokenizers/byte-level-models . llm/fw57M-tied \
--exclude ".git/*" \
--exclude ".gitattributes" \
--exclude "version_*/*" \
--repo-type model \
--commit-message "Copying trained fw57M-tied model" \
