import copy
import numpy as np
import matplotlib.pyplot as plt
import mne


def mne_arrays(
    bin_epochs: list,
    rate: int,
    ch_set: list,
):
    info = mne.create_info(
        ch_names=ch_set,
        ch_types=['eeg'] * len(ch_set),
        sfreq=rate
    )
    # Set the montage separately
    montage = mne.channels.make_standard_montage('easycap-M1')
    info.set_montage(montage)
    
    print(bin_epochs[0][0].shape, "oy ebitch")

    epochs = bin_epochs[0]
    events= np.array(bin_epochs[1])
    events = np.c_[np.arange(len(events)), events, events]

    print(events.shape, epochs.shape, "hgere at last")
    epochs_array = mne.EpochsArray(data=epochs, info=info, events=events, verbose=False)
    evoked_nontarget = epochs_array['0'].average()
    evoked_target_raw = copy.deepcopy(epochs_array['1']).average()

    epochs_delta = epochs_array['1'].subtract_evoked(evoked=evoked_nontarget)
    evoked_target_delta = copy.deepcopy(epochs_delta).average()

    return {
        'target': evoked_target_raw,
        'nontarget': evoked_nontarget,
        'delta': evoked_target_delta,
    }


def plot_evoked_electrodes(
    bin_epochs: list,
    rate: int,
    ch_set: list,
    title: str = '',
    figsize: tuple = (16, 10),
):
    '''Plots averaged evoked potentials from labeled binary epochs data

    Args:
        bin_epochs: data in format of `P300Dataset.binary_epochs`
        rate: rate of given signal
        ch_set: list of names of channels

    Returns:
        figure epochs plotted on
    '''
    colors = {
        'target': 'red',
        'nontarget': 'blue',
        'delta': 'black',
    }

    data = mne_arrays(bin_epochs, rate, ch_set)
    info = mne.create_info(
        ch_names=ch_set,
        ch_types=['eeg'] * len(ch_set),
        sfreq=rate
    )
    # Set the montage separately
    montage = mne.channels.make_standard_montage('easycap-M1')
    info.set_montage(montage)

    # plot data
    fig = plt.figure(figsize=figsize)
    topo_iterator = mne.viz.topo.iter_topography(info, fig=fig, layout_scale=0.945,
                                                 fig_facecolor='w', axis_facecolor='w', axis_spinecolor='w')
    for axis, channel in topo_iterator:
        for name, value in data.items():
            axis.plot(value.times, value.data[channel], color=colors[name], label=name)
        axis.axhline(0, color='black', linewidth=1)
        axis.grid()
        axis.set_title(ch_set[channel])

    fig.axes[0].legend(prop={'size': 10}, loc=(-0.4, 0.1))
    fig.suptitle(title, fontsize=20)


def plot_evoked_map(
    bin_epochs: list,
    rate: int,
    ch_set: list,
):
    target = mne_arrays(bin_epochs, rate, ch_set)['target']
    target.plot_topomap(times='peaks', title='Topomap of Target')


def plot_evoked_joint(
    bin_epochs: list,
    rate: int,
    ch_set: list,
):
    target = mne_arrays(bin_epochs, rate, ch_set)['target']
    target.plot_joint(title='Joint Plot of Target')
