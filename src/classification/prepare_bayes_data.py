from src.data.loader import load_h2
from src.data.data_cleaning import clean_data

raw = load_h2()
cleaned, report = clean_data(
    raw,
    drop_zero_adr=False,
    drop_company=True,
    keep_zero_nights=True
)

print(report)

from src.data.data_split import split_data, save_splits, summarize_splits

train, val, test = split_data(cleaned)

print(summarize_splits(train, val, test))

save_splits(train, val, test, out_dir="data/bayes")