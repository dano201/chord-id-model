import os

## Input frets can only be in format ABCDEF if all notes are on a single digit fret; otherwise, each fret value must be space-delineated, e.g. X 8 10 10 10 8
def find_notes(fret_string):

    frets = ""
    if len(fret_string) == 6:
        frets = list(fret_string)
    else:
        frets = fret_string.split(" ")

    if len(frets) < 6 or len(frets) > 6:
        raise ValueError("You must provide one fret value per string.")
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    open_notes = ['E', 'A', 'D', 'G', 'B', 'E']
    open_octaves = [2, 2, 3, 3, 3, 4]

    output = ""

    for i in range(len(frets)):

        val = frets[i]
        if val == 'X' or val == 'x':
            continue

        else:
            open_note = open_notes[i]
            open_octave = open_octaves[i]

            fret = int(val)
            
            #find note
            start = notes.index(open_note)
            note = notes[(start + fret) % 12]

            #find octave
            octave = ((start + fret) // 12) + open_octave
            output += (note + str(octave) + " ")

    return output.rstrip(" ")

## Make filepaths absolute so scripts can be executed from any directory
def make_abs(path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(path):
        return os.path.join(root, path)
    else:
        return path
    
## Dictionaries mapping notes to array indexes, and vice versa
note_dict = {
    'E2': 0, 'F2': 1, 'F#2': 2, 'G2': 3, 'G#2': 4, 'A2': 5, 'A#2': 6, 'B2': 7,
    'C3': 8, 'C#3': 9, 'D3': 10, 'D#3': 11, 'E3': 12, 'F3': 13, 'F#3': 14,
    'G3': 15, 'G#3': 16, 'A3': 17, 'A#3': 18, 'B3': 19, 'C4': 20, 'C#4': 21,
    'D4': 22, 'D#4': 23, 'E4': 24, 'F4': 25, 'F#4': 26, 'G4': 27, 'G#4': 28,
    'A4': 29, 'A#4': 30, 'B4': 31, 'C5': 32, 'C#5': 33, 'D5': 34, 'D#5': 35,
    'E5': 36
}
idx_dict = {v: k for k, v in note_dict.items()}
        