import logging
import copy
from os.path import join
from util import history_logger


class LogHandler():
    def __init__(self, opt):
        self.log = logging.basicConfig(filename=join(opt['logdir'],'console_log.txt'), level=logging.INFO)
        # Logging and visualization
        se = opt['start_epoch'] if opt['start_epoch'] > 0 else -1
        self.hist_log = history_logger.HistorySaver(join(opt['logdir'], 'hist'), se)
        self.hist_log.get_meta().update(copy.deepcopy(opt))

    def i(self, msg):
        logging.info(msg)
        print(msg)

    def add_hist(self, epoch, logmap):
        self.hist_log.add(epoch, logmap)

    def save(self):
        self.hist_log.save()

    