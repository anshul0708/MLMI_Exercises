import scipy.io as sio
import matplotlib.pyplot as plt


mat = sio.loadmat('/home/nomad/Documents/Projects/MLMI/04_exercise/AutoencoderAssignment/DRIVEdata.mat')
labels = mat['Labels']
images = mat['DataMatrix']
size = labels.shape


def load_data():
    load_size = int(size[0]*.8)
    return images[:load_size], images[load_size:]

def visualize_weights(W1, panel_shape, tile_size):
    # for coef, ax in zip(W1[0].T, axes.ravel()):
    #     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    # ax.set_xticks(())
    # ax.set_yticks(())
    print 'vizz'

