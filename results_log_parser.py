from datetime import datetime, date, time, timedelta
import re, copy


class ResultsLogParser:

    def __init__(self, log_path=r"results.log"):
        self.ops = 0
        self.log_path = log_path
        self.filtlist = self.get_all()
        self.history = []
        self.since_when_dict = {"today": datetime.combine(date.today(), time()),
                                "yesterday": datetime.combine(date.today() - timedelta(1), time()),
                                "start": datetime(2020, 3, 9, 0, 0),
                                "monday": self._get_weekday("monday"),
                                "wednesday": self._get_weekday("wednesday"),
                                "thursday": self._get_weekday("thursday"),
                                "friday": self._get_weekday("friday"),
                                "saturday": self._get_weekday("saturday"),
                                "sunday": self._get_weekday("sunday")}

    def get_all(self): #read all lines from log file
        with open(self.log_path, "r") as filtlist:
            all_results = [line for line in filtlist]
        self.filtlist = all_results

    def _manage_filtlist(self, filtlist=None):
        close = None
        if filtlist is None:
            filtlist = self.filtlist #look for existing log list
            if filtlist is not None:
                self.history.append(copy.deepcopy(filtlist)) #append current loglist to history in preparation for alteration
            else:
                filtlist = open(self.log_path, "r") #if no loglist found yet list hasn't been initiated, read from file
                close = True
        return filtlist, close

    def _split_line(self, line): #split line of log into sections based on | divider token
        messages = line.split(r' | ')
        return messages
    
    def get_as_list(self, keyword, filtlist=None): #get section of log as a list ie ids of correct sentences, out of vocabulary words
        results = []
        idx = self._find_index(keyword)
        filtlist, close = self._manage_filtlist(filtlist=filtlist)
        for line in filtlist:
            try:
                stringlist = self._split_line(line)[idx]
            except IndexError:
                continue
            reslist = self._dirtystring_to_list(stringlist)
            results += reslist
        if close:
            filtlist.close()
        try:
            return sorted(results)
        except TypeError:
            return results

    def _dirtystring_to_list(self, dirtystring): #turn a string containing a list (and potentially whitespace) as a list
        m = re.search(r'\[([^]]*)\]', dirtystring)
        cleanstring = m.group(0)
        cleanstring = cleanstring.replace("'", '')
        cleanlist = cleanstring.strip('][').split(', ')
        results = []
        for item in cleanlist:
            if item.isnumeric():
                item = int(item)
            elif item.replace('.', '', 1).isdigit():
                item = float(item)
            else:
                item = str(item)
            results.append(item)
        return results

    def filter_by_model(self, model, filtlist=None): #filter log by model name ie word2vec, fasttext, roberta, unigram, bigram etc

        results = []
        filtlist, close = self._manage_filtlist(filtlist=filtlist)

        for line in filtlist:
            messages = self._split_line(line)
            if model in messages[1]:
                results.append(line)
        if close:
            filtlist.close()

        self.filtlist = results
        self.ops += 1 #increment number of operations performed

    def undo_steps(self, steps): #undo filter history to restore previous state
        if type(steps) == int:
            if steps > 0:
                if steps <= self.ops:
                    self.history = self.history[:self.ops - steps]
                    self.ops -= steps
                else:
                    return print("Too many steps back")
            else:
                if abs(steps) < self.ops:
                    self.history = self.history[:abs(steps)]
                    self.ops = abs(steps)
                else:
                    return print("Too many steps forward from start")
        elif type(steps) == str:
            if steps in ["cl", "clear", "all"]:
                self.get_all()
                self.history = []
                self.ops = 0
                return

        self.filtlist = copy.deepcopy(self.history[-1])


    def _get_weekday(self, d): #filter log by number of days ago
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        twd = weekdays.index(d)  # get number of target weekday
        cwd = datetime.today().weekday()  # get number of current weekday
        td = abs(abs(cwd - twd) - 7)  # calculate time delta
        return datetime.combine(date.today() - timedelta(td), time())

    def _find_index(self,
                    search_terms):  # find index of message component, added to give more flexibility in log construction. ie correct ids may be index 3 when the log is divided into message sections
        idx = None
        if not isinstance(search_terms, list):
            search_terms = [search_terms]
        found = False
        with open(self.log_path, "r") as logfile:
            while not found:
                line = next(logfile)
                messages = self._split_line(line)
                if len(messages) > 2:
                    found = True
            for i, message in enumerate(messages):
                if any(substring in message for substring in search_terms):
                    idx = i
                    break
        return idx

    def filter_by_time(self, *args, hours_ago=None, since_when=None, filtlist=None): #filter logs by either a specific numbers if hours ago or since a point in time
        if len(args) == 1:
            compare_dt = datetime(
                *args[0])  # insert specific datetime in the format expected by datetime ie (2020, 3, 9, 21, 0)
        elif hours_ago is not None:
            compare_dt = (datetime.now() - timedelta(hours=hours_ago))
        elif since_when is not None:
            if since_when not in self.since_when_dict.keys():
                since_when = "start"
            compare_dt = self.since_when_dict[since_when]

        results = []
        filtlist, close = self._manage_filtlist(filtlist=filtlist)

        for line in filtlist:
            dt = self._split_line(line)[0]
            dt = datetime.strptime(dt, '%Y-%b-%d %H:%M:%S')
            if dt >= compare_dt:
                results.append(line)

        if close:
            filtlist.close()

        self.filtlist = results
        self.ops += 1
