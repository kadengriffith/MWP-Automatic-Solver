from argparse import ArgumentParser
import yaml
import os

DIR_PATH = os.path.abspath(os.path.dirname(__file__))


def get_args():
    parser = ArgumentParser(add_help=False)

    parser.add_argument("-c", "--config",
                        help="Reference to a yaml configuration file.",
                        type=str,
                        default="config.yaml")
    parser.add_argument("-e", "--epochs",
                        help="How many epochs to reach in MWP training.",
                        type=int,
                        default=300)
    parser.add_argument("-p", "--pretraining",
                        help="Adjust the pre-training corpus.",
                        type=str,
                        default=None)
    parser.add_argument("-t", "--tagging",
                        help="Adjust the tagging method.",
                        type=str,
                        default=None)
    parser.add_argument("-d", "--duplication",
                        help="Adjust the duplication of data",
                        type=int,
                        default=None)
    parser.add_argument("--stopwords",
                        help="Toggle stopwords in training.",
                        action='store_true',
                        default=False)
    parser.add_argument("--lemmas",
                        help="Toggle lemmatization in training.",
                        action='store_true',
                        default=False)
    parser.add_argument("--pos",
                        help="Toggle POS tagging in training.",
                        action='store_true',
                        default=False)
    parser.add_argument("--wpos",
                        help="Toggle POS tagging with words in training.",
                        action='store_true',
                        default=False)
    parser.add_argument("--reorder",
                        help="Toggle reordering of sentences in training.",
                        action='store_true',
                        default=False)
    parser.add_argument("-h", "--help",
                        help="The following options allow for automation for switching from pre-training to MWP training.\n \
                            -c, --config: A reference to a configuration file to alter before initiating training.\n \
                            -e, --epochs: The number of epochs to write to the config.\n \
                            -p, --pretraining: An option from 'wikipedia', 'dolphin', or 'imdb' to change pre-training to.\n \
                                If no option is provided, the training will be set for MWP data, with testing enabled.\n \
                            -t, --tagging: An option from 'lst', 'st', or 'et' to change tagging to.\n \
                            --stopwords, --lemmas, --pos, --wpos, --reorder: Flags to adjust pre-processing functions.\n")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # The config to alter
    config_file = args.config
    config_path = os.path.join(DIR_PATH, "../..", config_file)
    # Number of training iterations
    epochs = args.epochs
    pre_training = args.pretraining
    tagging = args.tagging
    duplication = args.duplication
    sw = args.stopwords
    l = args.lemmas
    pos = args.pos
    pos_words = args.wpos
    reorder = args.reorder

    updates = {
        "epochs": epochs,
        "remove_stopwords": sw,
        "as_lemmas": l,
        "pos": pos,
        "pos_words": pos_words,
        "reorder": reorder
    }

    if tagging is None:
        updates["tagging"] = False
    else:
        updates["tagging"] = tagging

    if duplication is None:
        updates["duplication"] = False
    else:
        updates["duplication"] = tagging

    with open(config_path, 'r', encoding='utf-8-sig') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

        if pre_training is None:
            updates["pretrain"] = False

            if "postfix" in config["dataset"]:
                updates["test"] = "postfix"
            elif "prefix" in config["dataset"]:
                updates["test"] = "prefix"
            elif "infix" in config["dataset"]:
                updates["test"] = "infix"
        else:
            updates["pretrain"] = pre_training

        config.update(updates)

    with open(config_path, 'w', encoding='utf-8-sig') as yaml_file:
        yaml.dump(config, yaml_file)
