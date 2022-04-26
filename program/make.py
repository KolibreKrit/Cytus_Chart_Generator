from musicprocessor import *
from learn import *
from synthesize import *
from bpm_detection import *
from random import randint
import sys
import math
import json

def get_bpm(filename):
    samps, fs = read_wav(filename)
    data = []
    correl = []
    bpm = 0
    n = 0
    nsamps = len(samps)
    window_samps = int(11*fs)
    samps_ndx = 0  # First sample in window_ndx
    max_window_ndx = math.floor(nsamps / window_samps)
    bpms = numpy.zeros(max_window_ndx)

    # Iterate through all windows
    for window_ndx in range(0, max_window_ndx):

        # Get a new set of samples
        # print(n,":",len(bpms),":",max_window_ndx_int,":",fs,":",nsamps,":",samps_ndx)
        data = samps[samps_ndx : samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))

        bpm, correl_temp = bpm_detector(data, fs)
        if bpm is None:
            continue
        bpms[window_ndx] = bpm
        correl = correl_temp

        # Iterate at the end of the loop
        samps_ndx = samps_ndx + window_samps

        # Counter for debug...
        n = n + 1

    bpm = numpy.median(bpms)
    print(bpm)
    return round(bpm)

def tja2cytus(title):
    FILE_PATH_1 = 'data/' + title + '-recombined.tja'
    FILE_PATH_2 = 'data/' + title + '-drums.tja'
    file_1 = open(FILE_PATH_1)
    file_2 = open(FILE_PATH_2)
    text_1 = file_1.read()
    text_2 = file_2.read()
    data_1 = text_1.splitlines()
    data_2 = text_2.splitlines()

    for i, line in enumerate(data_1):
        if 'BPM' in line:
            bpm = float(line.replace('BPM:', ''))
        if 'OFFSET' in line:
            music_offset = float(line.replace('OFFSET:', ''))
        
    measure_count = 0
    start = 0
    if len(data_1) > len(data_2):
        for i in range(len(data_1)):
            if '#END' in data_1[i]:
                break
            if ',' in data_1[i] and ':' not in data_1[i]:
                if start == 0:
                    start = i
                measure_count += 1
    else:
        for i in range(len(data_2)):
            if '#END' in data_1[i]:
                break
            if ',' in data_1[i] and ':' not in data_1[i]:
                if start == 0:
                    start = i
                measure_count += 1
    
    # Constants
    MSPM = 60000000 # microseconds per minute
    format_version = 1

    # Variables
    time_base = 480 # ticks per beat
    beats_per_page = 4

    mspb = math.ceil(MSPM / bpm) # microseconds per beat
    num_beats = measure_count * 4

    converted_data = {"format_version": format_version, 
         "time_base": time_base,
         "start_offset_time": music_offset,
         "page_list": get_page_list(time_base, num_beats, beats_per_page),
         "tempo_list": [{
             "tick": 0,
             "value": mspb
         }],
         "event_order_list": [],
         "note_list": get_note_list(data_1, data_2, start, time_base, beats_per_page)
        }
    return converted_data

def get_page_list(time_base, num_beats, beats_per_page):
    num_pages = math.ceil(num_beats / beats_per_page)
    page_list = list()
    for page in range(num_pages):
        page_list.append({"start_tick": page * time_base * beats_per_page,
                         "end_tick": (page+1) * time_base * beats_per_page,
                         "scan_line_direction": 1 if page % 2 == 0 else -1})
    return page_list    

def get_note_list(data_1, data_2, start, time_base, beats_per_page):
    note_list = list()
    note_id = 0
    count = start
    for i in range(start, len(data_1)):
        if '#END' in data_1[i]:
            break
        else:
            if ',' in data_1[i] and ':' not in data_1[i]:
                page_index = count-start
                page_tick = page_index*time_base*beats_per_page
                split = 0
                for j, note in enumerate(data_1[i]):
                    if note == ',':
                        split = j
                        break
                for j, note in enumerate(data_1[i]):
                    if note == ',':
                        break
                    # DON = tap on left
                    elif note == '1':
                        note_list.append({"page_index": page_index,
                                         "type": 0,
                                         "id": note_id,
                                         "tick": page_tick + math.ceil(j/split*time_base*beats_per_page),
                                         "x": 0.0 if page_index%2 == 0 else 0.2,
                                         "hold_tick": 0,
                                         "next_id": 0})
                        note_id += 1
                count += 1
    count = start
    for i in range(start, len(data_2)):
        if '#END' in data_2[i]:
            break
        else:
            if ',' in data_2[i] and ':' not in data_2[i]:
                page_index = count-start
                page_tick = page_index*time_base*beats_per_page
                split = 0
                for j, note in enumerate(data_2[i]):
                    if note == ',':
                        split = j
                        break
                for j, note in enumerate(data_2[i]):
                    if note == ',':
                        break
                    # DON = tap on right
                    elif note == '1':
                        note_list.append({"page_index": page_index,
                                         "type": 0,
                                         "id": note_id,
                                         "tick": page_tick + math.ceil(j/split*time_base*beats_per_page),
                                         "x": 1.0 if page_index%2 == 0 else 0.8,
                                         "hold_tick": 0,
                                         "next_id": 0})
                        note_id += 1
                count += 1
    return note_list

def adjust_page_list(page_list, bppc):
    num_pages = math.ceil(len(page_list) * bppc) + 5
    page_list_diff = page_list[0]['end_tick'] - page_list[0]['start_tick']
    new_page_list = list()
    for page in range(num_pages):
        new_page_list.append({"start_tick": page_list_diff * page,
                         "end_tick": page_list_diff * (page+1),
                         "scan_line_direction": 1 if page % 2 == 0 else -1})
    return new_page_list 

def double_page_list(page_list):
    num_pages = len(page_list) * 2 + 5
    new_ticks = math.ceil((page_list[0]['end_tick'] - page_list[0]['start_tick']) / 2)
    new_page_list = list()
    for page in range(num_pages):
        new_page_list.append({"start_tick": page * new_ticks,
                         "end_tick": (page+1) * new_ticks,
                         "scan_line_direction": 1 if page % 2 == 0 else -1})
    return new_page_list 

def adjust_note_list(note_list, bppc, page_list):
    for note in note_list:
        note['tick'] = math.ceil(note['tick'] * bppc)
        for index, page in enumerate(page_list):
            if note['tick'] < page['end_tick'] and note['tick'] >= page['start_tick']:
                note['page_index'] = index
                break
        if note['type'] == 1:
            if (note['tick'] + note['hold_tick']) > page_list[note['page_index']]['end_tick']:
                note['type'] = 2
        if note['x'] < 0.5:
            if note['page_index'] % 3 == 0:
                note['x'] = 0.0
            elif note['page_index'] % 3 == 1:
                note['x'] = 0.15
            else:
                note['x'] = 0.3
        else:
            if note['page_index'] % 3 == 0:
                note['x'] = 1.0
            elif note['page_index'] % 3 == 1:
                note['x'] = 0.85
            else:
                note['x'] = 0.7
    return note_list

def sync_note_list(note_list):
    for i, note_i in enumerate(note_list):
        for j, note_j in enumerate(note_list):
            if i != j and note_i['x'] < 0.5 and note_j['x'] > 0.5:
                if abs(note_i['tick'] - note_j['tick']) < 70:
                    note_j['tick'] = note_i['tick']
                    note_j['page_index'] = note_i['page_index']
    return note_list

def slide_note_list(note_list):
    left_list = list()
    right_list = list()
    for note in note_list:
        if note['x'] < 0.5:
            left_list.append(note)
        else:
            right_list.append(note)
    
    start = 0
    while start < len(left_list)-2:
        start, left_list = recursive_slide(start, left_list)
    start = 0
    while start < len(right_list)-2:
        start, right_list = recursive_slide(start, right_list)
    
    for note in note_list:
        if note['x'] < 0.5:
            for left_note in left_list:
                if left_note['id'] == note['id']:
                    note = left_note
        else:
            for right_note in right_list:
                if right_note['id'] == note['id']:
                    note = right_note
        
    return note_list

def recursive_slide(prev_note, side_list):
    # Return if slide ends
    if prev_note+1 >= len(side_list):
        return prev_note, side_list
    if (side_list[prev_note+1]['tick'] - side_list[prev_note]['tick'] > 210) or (prev_note+1 == len(side_list)):
        if side_list[prev_note]['type'] == 4:
            side_list[prev_note]['next_id'] = -1
        return prev_note+1, side_list
    
    else:
        # If previous note was not a slide, make it the head
        if side_list[prev_note]['type'] != 4:
            side_list[prev_note]['type'] = 3
        
        side_list[prev_note]['next_id'] = side_list[prev_note+1]['id']
        side_list[prev_note+1]['type'] = 4
        
        return recursive_slide(prev_note+1, side_list)

def snap_note_list(note_list, page_length, subdivision):
    page_sub = page_length / subdivision
    page_diff = page_sub / 2
    for note in note_list:
        if note['type'] != 3 and note['type'] != 4:
            if note['tick'] % page_sub < page_diff:
                note['tick'] = math.ceil(note['tick'] - (note['tick'] % page_sub))
            else:
                note['tick'] = math.ceil(note['tick'] + (page_sub - (note['tick'] % page_sub)))
    return note_list
        
def hold_note_list(note_list, page_list):
    left_list = list()
    right_list = list()
    for note in note_list:
        if note['x'] < 0.5:
            left_list.append(note)
        else:
            right_list.append(note)
            
    for i, note in enumerate(left_list):
        if (i == len(left_list)-1):
            break
        next_note_distance = left_list[i+1]['tick']-note['tick']
        page_end_distance = page_list[note['page_index']]['end_tick']-note['tick']
        if (next_note_distance >= 600 and next_note_distance < 1920):
            if (page_end_distance >= 600):
                if note['type'] == 0:
                    # if next_note_distance > page_end_distance:
                    #     note['type'] = 2
                    #     note['hold_tick'] = next_note_distance - 480
                    # else:
                    #     note['type'] = 1
                    #     note['hold_tick'] = next_note_distance - 240

                    note['type'] = 1
                    if page_end_distance > next_note_distance:
                        note['hold_tick'] = next_note_distance - 240
                    elif next_note_distance-page_end_distance >= 240:
                        note['hold_tick'] = page_end_distance
                    else:
                        note['hold_tick'] = page_end_distance - 240
                    
    for i, note in enumerate(right_list):
        if (i == len(right_list)-1):
            break
        next_note_distance = right_list[i+1]['tick']-note['tick']
        page_end_distance = page_list[note['page_index']]['end_tick']-note['tick']
        if (next_note_distance >= 600 and next_note_distance < 1920):
            if (page_end_distance >= 600):
                if note['type'] == 0:
                    # if next_note_distance > page_end_distance:
                    #     note['type'] = 2
                    #     note['hold_tick'] = next_note_distance - 480
                    # else:
                    #     note['type'] = 1
                    #     note['hold_tick'] = next_note_distance - 240

                    note['type'] = 1
                    note['hold_tick'] = min(next_note_distance, page_end_distance) - 240
        
    for note in note_list:
        if note['x'] < 0.5:
            for left_note in left_list:
                if left_note['id'] == note['id']:
                    note = left_note
        else:
            for right_note in right_list:
                if right_note['id'] == note['id']:
                    note = right_note
        
    return note_list

def procedural_note_list(note_list, page_list):
    # Reset positions
    for note in note_list:
        if note['x'] < 0.5:
            note['x'] = 0.0
        else:
            note['x'] = 1.0

    general_patterns = ['in', 'out', 'left', 'right']
    specific_patterns = ['in', 'out', 'left', 'right', 'jump']
    left_possibilities = [0.0, 0.1, 0.2, 0.3, 0.4]
    right_possibilities = [1.0, 0.9, 0.8, 0.7, 0.6]
    general_pattern = general_patterns[randint(0,len(general_patterns)-1)]
    specific_pattern = specific_patterns[randint(0,len(specific_patterns)-1)]
    ban_ticks = list()
    ban_x = list()
    for i, page in enumerate(page_list):
        start_tick = page['start_tick']
        if i == 0:
            for note in note_list:
                if note['page_index'] == i:
                    ban_ticks.append(note['tick']-start_tick)
                    ban_x.append(note['x'])

        else:
            for note in note_list:
                if note['page_index'] == i:
                    if note['x'] < 0.5 and note['type'] != 3:
                        note['x'] = choose_x((1920-(note['tick']-start_tick)), ban_ticks, ban_x, left_possibilities)
                    elif note['x'] > 0.5 and note['type'] !=3:
                        note['x'] = choose_x((1920-(note['tick']-start_tick)), ban_ticks, ban_x, right_possibilities)
            ban_ticks = list()
            ban_x = list()
            for note in note_list:
                if note['page_index'] == i:
                    ban_ticks.append(note['tick']-start_tick)
                    ban_x.append(note['x'])
    return note_list


def choose_x(tick, ban_ticks, ban_x, possibilities):
    results = list()
    for i, ban_tick in enumerate(ban_ticks):
        # If notes are close
        if abs(tick-ban_tick) < 240:
            for possibility in possibilities:
                if abs(possibility-ban_x[i]) <= 0.05:
                    results.append(possibility)

    for possibility in possibilities:
        if possibility not in results:
            return possibility
    

def smart_note_list(note_list, page_list):
    left_list = list()
    right_list = list()
    for note in note_list:
        if note['x'] < 0.5:
            left_list.append(note)
        else:
            right_list.append(note)
    
    # Change slides
    for i, note in enumerate(left_list):
        if note['type'] == 3:
            next_note = 0
            left_list[i+next_note]['x'] += 0.05 if (i+next_note) % 2 else 0.0
            # Zig zag pattern
            while left_list[i+next_note]['next_id'] != -1 and left_list[i+next_note+1]['id'] == left_list[i+next_note]['next_id']:
                next_note += 1
                left_list[i+next_note]['x'] += 0.05 if (i+next_note) % 2 else 0.0
                if i+next_note+1 >= len(left_list):
                    break
    for i, note in enumerate(right_list):
        if note['type'] == 3:
            next_note = 0
            right_list[i+next_note]['x'] -= 0.05 if (i+next_note) % 2 else 0.0
            # Zig zag pattern
            while right_list[i+next_note]['next_id'] != -1 and right_list[i+next_note+1]['id'] == right_list[i+next_note]['next_id']:
                next_note += 1
                right_list[i+next_note]['x'] -= 0.05 if (i+next_note) % 2 else 0.0
                if i+next_note+1 >= len(right_list):
                    break

    # Change long holds
    for i, note in enumerate(left_list):
        if note['type'] == 2:
            x_bans = list()
            if i > 0:
                if left_list[i-1]['page_index'] == note['page_index']:
                    x_bans.append(note['x'])
            if i < (len(left_list)-1):
                x_bans.append(left_list[i+1]['x'])
            possibilities = [0.0, 0.15, 0.30, 0.40]  
            for possibility in possibilities:
                possible = True
                for ban in x_bans:
                    if abs(possibility-ban) <= 0.05:
                        possible = False
                        break
                if possible:
                    note['x'] = possibility
    for i, note in enumerate(right_list):
        if note['type'] == 2:
            x_bans = list()
            if i > 0:
                if left_list[i-1]['page_index'] == note['page_index']:
                    x_bans.append(note['x'])
            if i < (len(left_list)-1):
                x_bans.append(left_list[i+1]['x'])
            possibilities = [1.0, 0.85, 0.70, 0.60]  
            for possibility in possibilities:
                possible = True
                for ban in x_bans:
                    if abs(possibility-ban) <= 0.05:
                        possible = False
                        break
                if possible:
                    note['x'] = possibility
    # Cement changes
    for note in note_list:
        if note['x'] < 0.5:
            for left_note in left_list:
                if left_note['id'] == note['id']:
                    note = left_note
        else:
            for right_note in right_list:
                if right_note['id'] == note['id']:
                    note = right_note

    # Change pages of certain notes
    for note in note_list:
        if note['tick'] == page_list[note['page_index']]['end_tick']:
            note['page_index'] += 1
        
    return note_list
    

def trim_page_list(note_list, page_list):
    max_tick = 0
    for note in note_list:
        if note['tick'] > max_tick:
            max_tick = note['tick']
    new_page_list = list()
    for page in page_list:
        if page['start_tick'] <= max_tick:
            new_page_list.append(page)
        else:
            break
    return new_page_list


title = sys.argv[1]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
net = convNet()
print(net)
net = net.to(device)
print(os.listdir('data//pickles'))

# Get timings for left side
serv = "data/songs/" + title + "/" + title + "-recombined"
musicfortest(serv)

with open('data//pickles//testdata.pickle', mode='rb') as f:
    song = pickle.load(f)
if torch.cuda.is_available():
    net.load_state_dict(torch.load('model//convmodelonset.pth'))
else:
    net.load_state_dict(torch.load('model//convmodelonset.pth',map_location='cpu'))
inference = net.infer(song.feats, device, minibatch=4192)
inference = np.reshape(inference, (-1))
by_librosa_detection2(inference, song)
create_tja("data//" + title + "-recombined.tja", song, song.timestampboth)

# Get timings for right side
serv = "data/songs/" + title + "/" + title + "-drums"
musicfortest(serv)

with open('data//pickles//testdata.pickle', mode='rb') as f:
    song = pickle.load(f)
inference = net.infer(song.feats, device, minibatch=4192)
inference = np.reshape(inference, (-1))
by_librosa_detection2(inference, song)
create_tja("data//" + title + "-drums.tja", song, song.timestampboth)

# Convert timings to rudimentary chart
data = tja2cytus(title)

# Change bpm over to correct bpm
if (len(sys.argv) > 2):
    bpm = int(sys.argv[2])
else: 
    bpm = get_bpm("data/songs/" + title + "/" + title + ".wav")
bppc = bpm / 240
mspb = math.ceil(60000000 / bpm)
data['page_list'] = adjust_page_list(data['page_list'], bppc)
data['note_list'] = adjust_note_list(data['note_list'], bppc, data['page_list'])
data['tempo_list'][0]['value'] = mspb

# Add slide notes
data['note_list'] = slide_note_list(data['note_list'])

if bpm < 90:
    data['page_list'] = adjust_page_list(data['page_list'], 2)
    data['note_list'] = adjust_note_list(data['note_list'], 2, data['page_list'])
    bpm = bpm * 2
    mspb = math.ceil(60000000 / bpm)
    data['tempo_list'][0]['value'] = mspb

# Sync double notes
data['note_list'] = sync_note_list(data['note_list'])

# Double beats per page
data['page_list'] = double_page_list(data['page_list'])
data['note_list'] = adjust_note_list(data['note_list'], 1, data['page_list'])

# Snap notes to bpm
subdivision = 4
page_length = data['page_list'][0]['end_tick']
data['note_list'] = snap_note_list(data['note_list'], page_length, subdivision)

# Add holds
data['note_list'] = hold_note_list(data['note_list'], data['page_list'])

# # Intelligently change positions
data['note_list'] = smart_note_list(data['note_list'], data['page_list'])
# data['note_list'] = procedural_note_list(data['note_list'], data['page_list'])

# Trim page list
data['page_list'] = trim_page_list(data['note_list'], data['page_list'])


filename = "Final_" + title + '.json'
with open(filename, 'w') as f:
    json.dump(data, f)
