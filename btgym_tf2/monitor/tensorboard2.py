###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
import os
from subprocess import PIPE
import psutil
import glob

import warnings

try:
    import tensorflow as tf

except:
    warnings.warn('BTgymMonitor2 requires Tensorflow')

    class BTgymMonitor2():
        pass

    quit(1)


class BTgymMonitor2():
    """
    Light tensorflow 'summaries' wrapper for convenient tensorboard logging.
    """
    def __init__(self,
                 scalars={},
                 images={},
                 histograms={},
                 text={},
                 logdir='./tb_log',
                 subdir='/',
                 purge_previous=True,
                 **kwargs):
        """
        Monitor parameters:
        Sets of names for every value category: scalars, images, histograms ant text.
        logdir - tensorboard log directory;
        subdir - this monitor log subdirectory;
        port - localhost webpage addr to look at;
        reload - web page refresh rate.
        purge_previous - delete previous logs in logdir/subdir if found.
        """

        self.tensorboard = Tensorboard(logdir=logdir, **kwargs)
        self.logdir = logdir+subdir
        self.purge_previous = purge_previous
        self.feed_holder = dict()
        self.summary = None

        # Remove previous log files if opted:
        if self.purge_previous:
            files = glob.glob(self.logdir + '/*')
            p = psutil.Popen(['rm', '-R', ] + files, stdout=PIPE, stderr=PIPE)

        # Prepare writer:
        self.writer = tf.compat.v1.summary.FileWriter(self.logdir, graph=tf.compat.v1.get_default_graph())


        # Create summary:
        summaries = []

        for entry in scalars:
            assert type(entry) == str
            self.feed_holder[entry] = tf.compat.v1.placeholder(tf.float32)
            summaries += [tf.compat.v1.summary.scalar(entry, self.feed_holder[entry],)]

        for entry in images:
            assert type(entry) == str
            self.feed_holder[entry] = tf.compat.v1.placeholder(tf.uint8, [None, None, None, 3])
            summaries += [tf.compat.v1.summary.image(entry, self.feed_holder[entry], )]

        for entry in histograms:
            assert type(entry) == str
            self.feed_holder[entry] = tf.compat.v1.placeholder(tf.float32,[None, None],)
            summaries += [tf.compat.v1.summary.histogram(entry, self.feed_holder[entry], )]

        for entry in text:
            assert type(entry) == str
            self.feed_holder[entry] = tf.compat.v1.placeholder(tf.string)
            summaries += [tf.compat.v1.summary.histogram(entry, self.feed_holder[entry], )]

        self.summary = tf.compat.v1.summary.merge(summaries)

    def write(self, sess, feed_dict, global_step):
        """
        Updates monitor with provided data.
        """
        feeder = dict()

        # Assert feed_dict is ok:
        try:
            for key in self.feed_holder.keys():
                assert key in feed_dict
                feeder.update({self.feed_holder[key]: feed_dict[key]})


        except:
            raise AssertionError('Inconsistent monitor feed:\nGot: {}\nExpected: {}\n'.
                                 format(feed_dict.keys(),self.feed_holder.keys())
                                )
        # Write down:
        evaluated = sess.run(self.summary, feed_dict=feeder)
        self.writer.add_summary(summary=evaluated, global_step=global_step)
        self.writer.flush()

    def close(self):
        self.writer.close()

class Tensorboard():
    """
    Utility class to start/stop tensorboard server.
    """
    def __init__(self, logdir='./btgym_log', port=6006, reload=30,):
        """____"""
        self.port = port
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.process = None
        self.pid = ''

        # Compose start command:
        self.start_string = ['tensorboard']

        assert type(logdir) == str
        self.start_string += ['--logdir={}'.format(logdir)]

        assert type(port) == int
        self.start_string += ['--port={}'.format(port)]

        assert type(reload) == int
        self.start_string += ['--reload_interval={}'.format(reload)]

        self.start_string += ['--purge_orphaned_data']

    def start(self):
        """Launches Tensorboard app."""

        # Kill everything on port-to-use:
        p = psutil.Popen(['lsof', '-i:{}'.format(self.port), '-t'], stdout=PIPE, stderr=PIPE)
        self.pid = p.communicate()[0].decode()[:-1]  # retrieving PID

        if self.pid is not '':
            p = psutil.Popen(['kill', self.pid])  # , stdout=PIPE, stderr=PIPE)

        # Start:
        self.process = psutil.Popen(self.start_string)  # , stdout=PIPE, stderr=PIPE)

    def stop(self):
        """Closes tensorboard server."""
        if self.process is not None:
            self.process.terminate()
