import tensorflow as tf
import numpy as np
import os
from music21 import *

# import model1
# import model2


# model_name = input("choose modle:")
# if model_name == 1:
#     model = model1
# else:
#     model = model2
#
# checkpoint_dir = './training_checkpoints_model{}'.format(model_name)
# generator = model.make_generator_model()
# discriminator = model.make_discriminator_model()
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=model.generator_optimizer,
#                                  discriminator_optimizer=model.discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
#
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def predictions(generator):
    seed = tf.random.normal([1, 100])
    predictions = generator(seed, training=False)
    pre = np.reshape(predictions.numpy(),[4,200])
    # pre0 = np.trunc(np.abs(pre[0]))
    # pre1 = np.trunc(np.abs(pre[1]))
    # pre2 = np.around(np.abs(pre[2]),decimals =  2)
    # pre3 = np.around(np.abs(pre[3]),decimals =  2)
    # print(pre)
    # input()
    note0 = np.abs(pre[0]).tolist()
    velocity0 = np.abs(pre[1]).tolist()
    duration0 = np.abs(pre[2]).tolist()
    time0 = np.abs(pre[3]).tolist()
    print(note0,"\n",velocity0,"\n",duration0,"\n",time0)
    #
    big = stream.Stream()
    temp = 0
    TimeSignature0 = None

    for el in range(0, len(time0)):
        # temp = temp + 1
        if temp < len(time0) - 10:
            count = time0[temp: temp + 10].count(time0[temp])
        else:
            count = time0[temp: len(time0)].count(time0[temp])
        if temp != len(time0) - 1:
            if count == 1:
                note_geti = note.Note(note0[temp])
                note_geti.duration = duration.Duration(duration0[temp])
                note_geti.volume.velocity = velocity0[temp]
                big.insert(time0[temp], note_geti)
                # note_geti.offset = time0[temp]
                # measure.insert(time[temp],note_geti)
                # big.append(note_geti)
                temp = temp + 1
            if count != 1 and temp + count < len(time0):
                chord_geti = chord.Chord()
                for el in range(0, count):
                    note_on_chord = note.Note(note0[temp + el])
                    note_on_chord.volume.velocity = velocity0[temp + el]
                    chord_geti.add(note_on_chord)
                chord_geti.duration = duration.Duration(duration0[temp])
                big.insert(time0[temp], chord_geti)

                # chord_geti.offset = time0[temp]
                # print(chord_geti.offset)
                # measure.insert(time[temp], chord_geti)
                temp = temp + count

    # for el in range(0,len(tempo_time)):
    #     tp =tempo.MetronomeMark(number=tempo0[el])
    #     if el < len(tempo_time):
    #         big.insert(tempo_time[el], tp)

    # ts = meter.TimeSignature('{}'.format(TimeSignature0))

    # big.insert(0, 3/4)
    big.show('midi')
    # input()
def generate(generator,i):
    for el in range(0,i):
        predictions(generator)