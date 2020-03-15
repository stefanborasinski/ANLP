import csv
from nltk import word_tokenize as tokenize

class question:

    def __init__(self, aline):
        self.fields = aline

    def get_field(self, field):
        return self.fields[question.colnames[field]]

    def add_answer(self, fields):
        self.fields += fields[1]

    def make_sentence(self, answer,highlight=False):
        q = self.get_field("question")
        if highlight:
            answer = "*"+answer+"*"
        return q.replace("_____", answer)


class scc_reader:

    def __init__(self, qs=r"data/testing_data.csv", ans=r"data/test_answer.csv"):
        self.qs = qs
        self.ans = ans
        self.read_files()

    def read_files(self):
        # read in the question file
        with open(self.qs) as instream:
            csvreader = csv.reader(instream)
            qlines = list(csvreader)

        # store the column names as a reverse index so they can be used to reference parts of the question
        question.colnames = {item: i for i, item in enumerate(qlines[0])}
        question.colnames["answer"] = len(question.colnames)

        # create a question instance for each line of the file (other than heading line)
        self.questions = [question(qline) for qline in qlines[1:]]

        # read in the answer file
        with open(self.ans) as instream:
            csvreader = csv.reader(instream)
            alines = list(csvreader)

        # add answers to questions so predictions can be checked
        for q, aline in zip(self.questions, alines[1:]):
            q.add_answer(aline)

    def get_field(self, field):
        return [q.get_field(field) for q in self.questions]