from music21 import *
import numpy as np
import os
import glob
import tensorflow as tf
data = []

def get_data(path = 'data/beeth/beethoven_opus10_1.mid'):
# def get_data(path = 'generat_music/tmpabkizmwf.mid'):
    file = converter.parse(path)
    # file.show('text')
    # input()
    time0 =[]
    duration0 = []
    note0 = []
    big = stream.Stream()
    temp = 0
    velocity0 = []
    TimeSignature0 = None
    tempo0 = []
    tempo_time = []
    for el in file.recurse():

        if type(el) == note.Note:
            # print(el.offset, el.duration.quarterLength)
            velocity0.append(el.volume.velocity)
            # input()
            time0.append(el.offset)
            note0.append(el.pitch.midi)
            duration0.append(el.duration.quarterLength)
        # print(time,duration)
        if type(el) == chord.Chord:
            # print(el.offset, el.duration.quarterLength)
            el_ = el.pitches
            # print(len(el_))
            for a in range(0, len(el_)):
                velocity0.append(el.volume.velocity)
                time0.append(el.offset)
                note0.append(el_[a].midi)
                duration0.append(el.duration.quarterLength)
    # print(time0)
    # print(note0)
    # print(duration0)
        if type(el) == meter.TimeSignature:
            TimeSignature0 = el.ratioString

        if type(el) == tempo.MetronomeMark:
            tempo0.append(el.number)
            tempo_time.append(el.offset)



    # for el in range(0,len(time0)):
    #     # temp = temp + 1
    #     if temp < len(time0) - 10:
    #         count = time0[temp : temp + 10].count(time0[temp])
    #     else:
    #         count = time0[temp : len(time0)].count(time0[temp])
    #     if  temp != len(time0) - 1:
    #         if count == 1:
    #             note_geti = note.Note(note0[temp])
    #             note_geti.duration = duration.Duration(duration0[temp])
    #             note_geti.volume.velocity = velocity0[temp]
    #             big.insert(time0[temp], note_geti)
    #             # note_geti.offset = time0[temp]
    #             # measure.insert(time[temp],note_geti)
    #             # big.append(note_geti)
    #             temp = temp + 1
    #         if count != 1 and temp + count < len(time0):
    #             chord_geti = chord.Chord()
    #             for el in range(0,count):
    #                 note_on_chord = note.Note(note0[temp + el])
    #                 note_on_chord.volume.velocity = velocity0[temp + el]
    #                 chord_geti.add(note_on_chord)
    #             chord_geti.duration = duration.Duration(duration0[temp])
    #             big.insert(time0[temp], chord_geti)
    #
    #             # chord_geti.offset = time0[temp]
    #             # print(chord_geti.offset)
    #             # measure.insert(time[temp], chord_geti)
    #             temp = temp + count
    #
    # # for el in range(0,len(tempo_time)):
    # #     tp =tempo.MetronomeMark(number=tempo0[el])
    # #     if el < len(tempo_time):
    # #         big.insert(tempo_time[el], tp)
    #
    # ts = meter.TimeSignature('{}'.format(TimeSignature0))
    #
    # big.insert(0, ts)
    #
    # # print(len(time0),len(velocity0))
    # # print(file_MIDI)
    #
    # # big.show('midi')


    return np.array([note0,velocity0,duration0,time0])
print(get_data())
def get_more_data():
    path = r'E:\GAN学习\listen\data\beeth'
    dataset = []
    file = os.listdir(path)
    a=0
    for i in file:
        data0 = get_data(path + '/' + i)[:, 0:200]
        dataset.append(data0.astype(np.float32))
        a+=1
    dataset = np.array(dataset)
    my_data = tf.constant(dataset, shape=[a,4,200],dtype = tf.float32)  # int32类型

    return my_data

# get_more_data = get_more_data()
# # print(get_more_data)
# print(get_more_data)