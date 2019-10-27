from music21 import *
import numpy as np
import os
import glob

def get_data(path = 'data/beethoven_opus10_1.mid'):
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
# print(get_data())
def get_more_data():
    path = r'E:\GAN学习\listen\data\beeth'
    dataset = []
    file = os.listdir(path)
    for i in file:
        dataset.append(get_data(path + '/' + i ))
    return dataset

# a = get_more_data()
# print(a)

# a = get_data()
# print(len(a),len(b),len(c),len(d))
# # print(a)
# # print(b)
# # print(c)
# # print(d)
# print(np.array([a,b,c,d]))
# input()
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# def midi_to_data(file): #return time and note set
#     time_ = []
#     note_ = []
#     duration_ = []
#     velocity_ = []
#     for el in file.recurse():
#         if type(el) == note.Note:
#             print(el.offset, el.pitch.midi, el.duration.quarterLength)
#             time_.append(el.offset)
#             note_.append(el.pitch.midi)
#             duration_.append(el.duration.quarterLength)
#
#         if type(el) == chord.Chord:
#             print(el.offset, el.duration.quarterLength)
#
#             el_ = el.pitches
#             # chord0 = []
#             for a in range(0,len(el_)):
#             #     chord0.append(el_[a].midi)
#             # print(el.offset, chord0 , 2)
#             #     print(el.offset, el_[a].midi)
#                 time_.append(el.offset)
#                 note_.append(el_[a].midi)
#                 duration_.append(el.duration.quarterLength)
#
#
#     return time_,note_,duration_
#
# def date_to_midi(time_,note_,duration_,velocity_=None):# return type of value is music21.stream
#     temp = 0
#     get_voice = stream.Voice()
#     for el in range(0,len(time_)):
#         # temp = temp + 1
#         if temp < len(time_) - 10:
#             count = time_[temp : temp + 10].count(time_[temp])
#         else:
#             count = time_[temp : len(time_)].count(time_[temp])
#         if  temp != len(time_) - 1:
#             if count == 1:
#                 note_geti = note.Note(note_[temp])
#                 note_geti.duration = duration.Duration(duration_[temp])
#                 note_geti.offset = time_[temp] + duration_[temp]
#                 # measure.insert(time[temp],note_geti)
#                 get_voice.append(note_geti)
#                 temp = temp + 1
#             if count != 1 and temp + count < len(time_):
#                 chord_geti = chord.Chord()
#                 for el in range(0,count):
#                     chord_geti.add(note.Note(note_[temp + el]))
#                 chord_geti.duration = duration.Duration(duration_[temp])
#                 chord_geti.offset = time_[temp] + duration_[temp]
#                 # print(chord_geti.offset)
#                 # measure.insert(time[temp], chord_geti)
#                 temp = temp + count
#                 get_voice.append(chord_geti)
#     # biggerStream.show('midi')
#     # biggerStream.show('text')
#     return get_voice
#
# voice_ = []
# def get_voice(file):
#
#     biggerStream0 = stream.Part()
#     biggerStream1 = stream.Voice()
#     for el in file.recurse():
#         biggerStream = stream.Voice()
#         if type(el) == stream.Voice:
#             # print(el.elements)
#             # voice_.append(el.elements)
#             biggerStream.append(el.elements)
#             a, b, c = midi_to_data(biggerStream)
#             print(a[0:10])
#             print(b[0:10])
#             print(c[0:10])
#             # print(type(date_to_midi(a, b, c)))
#             biggerStream0.append(date_to_midi(a, b, c))
#             # biggerStream0.show("text")
#             input()
#             # biggerStream0.insert(biggerStream1)
#     # biggerStream0.show('text')
#
#
# get_voice(file)
#
#
# # !!!!!!!!!!!!!!test!test test test test test test tset test tset tset test test test test test test
# # a, b = midi_to_data(file)
# # date_to_midi(a, b)
# # time = []
# # note_ = []
# # for el in file.recurse():
# #     # if type(el) == note.Note:
# #     #     print(el.offset, el.pitch.midi, el.duration.quarterLength)
# #     if type(el) == chord.Chord:
# #
# #         el_ = el.pitches
# #         # chord0 = []
# #         for a in range(0,len(el_)):
# #         #     chord0.append(el_[a].midi)
# #         # print(el.offset, chord0 , 2)
# #             print(el.offset, el_[a].midi, el.duration.quarterLength, el.activeSite.elements)
# #             # time.append(el.offset)
# #             # note_.append(el_[a].midi)
# #             # duration_.append(el.duration.quarterLength)
# #         input()
#
#
# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111111
#
# # file_MIDI = midi.translate.streamToMidiFile(file)
# # # print(file_MIDI)
# # track_ = []
# # channel_ = []
# # pitch_ = []
# #
# # for el in file_MIDI.tracks:
# #     for element in el.events:
# #         if element.isNoteOn() is True:
# #             velocity.append(element.velocity)
# #         if element.isDeltaTime() is True:
# #
# # print(a.getBytes())