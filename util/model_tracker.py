import torch
from os.path import join

# Experimental saver class
class ModelSaveTracker(object):
    def __init__(self, freq, track, prefix=""):
        self.freq = freq
        self.track = {prefix+k:v for k,v in track.items()}
        self.best_record_map = {}

    def load(self, filename, load_key='track'):
        perf = torch.load(filename)[load_key]
        for metric, value in perf.items():
            self.best_record_map[metric] = value
            print('Loaded the best {} = {}'.format(metric, value))
        
        untracked = set(perf) - set(self.track)
        if len(untracked) != 0:
            print('Untracked metrics', untracked)
            raise Exception('There are untracked metric. Update conf file track option')

        newtrack = set(self.track) - set(perf)
        if len(newtrack) != 0:
            print('Newly tracked metrics', newtrack)

    def update(self, epoch, model, model_root, filename='chk_{}.pth.tar', **perfs):
        if (epoch+1)%self.freq == 0:
            ModelSaveTracker.save(model, filename=join(model_root, filename.format(epoch)), perf=perfs, extras={'epoch':epoch})

        for metric, higher_is_better in self.track.items():
            if metric not in perfs: raise Exception('Unknown metric. Please update conf file track option', metric)
            higher_is_better = (higher_is_better == 1)
            value = perfs[metric]

            # Update best score map if there is no entry about the metric
            if metric not in self.best_record_map:
                self.best_record_map[metric] = value

            # Check the current score whether it is the best or not
            if higher_is_better and value >= self.best_record_map[metric]:
                self.best_record_map[metric] = value
                ModelSaveTracker.save(model, filename=join(model_root, filename.format('best_'+metric)), perf=perfs, extras={'epoch':epoch})
            elif not higher_is_better and value <= self.best_record_map[metric]:
                self.best_record_map[metric] = value
                ModelSaveTracker.save(model, filename=join(model_root, filename.format('best_'+metric)), perf=perfs, extras={'epoch':epoch})
    
    @staticmethod
    def save(netA, filename, perf={}, extras={}):
        state = {}
        state['model'] = netA.state_dict()
        state.update(extras)
        state['track'] = perf
        torch.save(state, filename)