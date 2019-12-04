import os
import re
import json
import pickle

DIR_PATH = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":
    with open(os.path.join(DIR_PATH, "Dolphin18K.json"), encoding='utf-8-sig') as fh:
        dolphin = json.load(fh)

    dataset = []
    for question in dolphin:
        dq = dict(question)

        text = dq["text"]

        # Filter the data
        # Yes, I know this looks like a nightmare!!
        text = re.sub(r"\/?>.*", '', text)
        if not "http" in text:
            text = re.sub(r"(\"|')", '', text)
            text = re.sub(r"[^0-9]\.[^0-9]", " . ", text)
            text = re.sub(r"\?", " ? ", text)
            text = re.sub(r"!", " ! ", text)
            text = re.sub(r"-", " - ", text)
            text = re.sub(r"\+", " + ", text)
            text = re.sub(r"\\", " \ ", text)
            text = re.sub(r"\/", " / ", text)
            text = re.sub(r"\*", " * ", text)
            text = re.sub(r"=", " = ", text)
            text = re.sub(r"\$", " $ ", text)
            text = re.sub(r"%", " % ", text)
            text = re.sub(r";", " ; ", text)
            text = re.sub(r":", " : ", text)
            text = re.sub(r"'", " '", text)
            text = re.sub(r"\)", " ) ", text)
            text = re.sub(r"\(", " ( ", text)
            text = re.sub(r"\(", " ( ", text)
            text = re.sub(r"_", '', text)
            text = re.sub(r"[^0-9],[^0-9]", " , ", text)
            text = re.sub(r"&.*;", "", text)
            text = re.sub(r"([0-9]+(\.[0-9]+)?)", r" \1 ", text).strip()
            text = re.sub(r"\s+", ' ', text)
            text = re.sub(r"((\s)?$|^(\s)?)", '', text)

            if len(text) > 0:
                dataset.append(text.lower())

    # print(len(dataset), dataset[:10])

    # Save to a binary file
    with open(os.path.join(DIR_PATH, "dolphin.pretraining.p"), 'wb') as fh:
        pickle.dump(dataset, fh)

    print("Pretraining set created...")
    print(f"{len(dataset)} examples have been cleaned and saved.")
