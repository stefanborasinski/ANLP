import logging, random, os, json

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%Y-%b-%d %H:%M',
                    filename="results.log", filemode='a')
logger = logging.getLogger(__name__)

keys = ["a)", "b)", "c)", "d)", "e)"]


def get_training_testing(training_dir=r"data/Holmes_Training_Data", split=0.5):
    filenames = os.listdir(training_dir)
    n = len(filenames)
    print("There are {} files in the training directory: {}".format(n, training_dir))
    random.seed(53)  # if you want the same random split every time
    random.shuffle(filenames)
    index = int(n * split)
    trainingfiles = filenames[:index]
    heldoutfiles = filenames[index:]
    return trainingfiles, heldoutfiles


def log_results(model_name, total_correct, num_qs, correct_list, incorrect_list):
    logging.info(
        f"{model_name} | accuracy = {(total_correct / num_qs):.3f} | right ids: {correct_list} | wrong ids: {incorrect_list}")
    return print(f"Done and logged! Accuracy was {(total_correct / num_qs):.3f}")


def load_json(filename):
    with open(filename) as js:
        data = json.load(js)
    return data
