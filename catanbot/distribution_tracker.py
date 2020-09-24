import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import numpy as np

class DistributionTracker:
    """
    Utility to track rolls and display them
    Important to keep track or order so we can watch distribution through time
    """
    def __init__(self):
        self.rolls = []

        self.show_avg = True

        self.show_up_to = 0

    def add_roll(self, roll):
        try:
            self.rolls.append(int(roll))
        except:
            print('Expected roll to be in [2, 12], got {}'.format(roll))

    def draw_dist(self, up_to = None):
        if up_to is None:
            data = self.rolls
        else:
            data = self.rolls[:up_to]

        self.ax.hist(data, density=True, align = 'left', rwidth=0.8, bins = range(2, 14), label = 'Actual')

    def draw_avg(self):
        """
        Superimpose the diagonal dist over the histogram
        """
        x = np.arange(2., 13.)
        y = (1/6) - ((abs(7-x))/36)
        self.ax.plot(x, y, label = 'Average', marker='x')

    def slider_update(self, val):
        self.ax.cla()
        self.ax.set_xlim(1, 13)
        self.ax.set_ylim(0, 1)
        self.draw_dist(up_to = int(val))
        self.draw_avg()

    def update(self):
        self.fig, self.ax = plt.subplots(figsize = (4, 6))
        self.fig.suptitle('Roll Distribution ({} turns)'.format(len(self.rolls)))        
        plt.subplots_adjust(bottom=0.25)
        self.s_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(self.s_ax, 'Freq', 0, 0, valinit=0, valstep=1)
        self.ax.set_xlim(1, 13)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks(range(1, 14))

        self.draw_dist()
        self.draw_avg()
        self.s_ax.cla()
        self.slider = Slider(self.s_ax, 'Turn Number', 0, len(self.rolls), valinit=len(self.rolls), valstep=1)
        self.slider.on_changed(self.slider_update)

        plt.legend()
        
if __name__ == '__main__':
    tracker = DistributionTracker()
    #import pdb;pdb.set_trace()
    while True:
        roll = input('Turn {}, roll = '.format(1 + len(tracker.rolls)))
        tracker.add_roll(roll)
        tracker.update()
        plt.show()
