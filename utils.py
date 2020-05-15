import logging, random, os, json, argparse

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s | %(message)s', datefmt='%Y-%b-%d %H:%M:%S',
                    filename="results.log", filemode='a') #logging level critical to avoid also logging the warning messages of various models
logger = logging.getLogger(__name__)

def get_training_testing(training_dir, split=1.0):
    filenames = sorted(os.listdir(training_dir))
    n = len(filenames)
    print("There are {} files in the training directory: {}".format(n, training_dir))
    #random.seed(53)  # if you want the same random split every time since working on colab seeding doesn't work - only way to ensure same order is alphabetical sorting
    #random.shuffle(filenames)
    index = int(n * split)
    trainingfiles = filenames[:index]
    heldoutfiles = filenames[index:]
    return trainingfiles, heldoutfiles


def log_results(model_name, total_correct, num_qs, correct_list, incorrect_list, **kwargs):
    kwstr = [f"| {k} = {v}" for k, v in kwargs.items()]
    kwstr = " ".join(kwstr)
    logger.critical(
        f"{model_name} | accuracy = {[(total_correct / num_qs)]} | right ids: {correct_list} | wrong ids: {incorrect_list} {kwstr}")
    return print(f"Done and logged! Accuracy was {(total_correct / num_qs):.3f}")


def load_json(filename):
    with open(filename) as js:
        data = json.load(js)
    return data


def get_default_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str,
                        help='Language model mode if more than one exist, ie unigram/bigram')
    parser.add_argument('-c', '--config', default='config.json', help='language model config json')
    parser.add_argument('-td', '--training_dir', type=str, default=r"cleaned_data/Holmes_Training_Data",
                        help='location of Holmes_Training_Data folder')
    parser.add_argument('-vb', "--verbose", type=bool, default=True,
                        help="Print out processed files alongside answers to question to debug")
    parser.add_argument('-mf', '--max_files', type=int, default=None, help='maximum number of files used in training. choose 0 in order to skip training')
    return parser
