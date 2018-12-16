'''
writes mod files for all channels defined in ``neat.channels.channelcollection``
'''
import os

from neat.channels import channelcollection


for name, channel_class in channelcollection.__dict__.items():
    if isinstance(channel_class, type) and name != 'IonChannel':
        chan = channel_class()
        chan.writeModFile(os.path.join(os.path.dirname(__file__), 'mech'))
