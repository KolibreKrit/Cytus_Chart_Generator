import json
import os
from random import randint
from random import shuffle
import numpy as np
import math
import sys

def page_zones(note_list, page_list, page):
    zones = np.zeros((17, 21), dtype=np.int8)
    start_tick = page_list[page]['start_tick']
    for note in note_list:
        if note['page_index'] == page:
            tick = math.floor((note['tick']-start_tick)/60)
            x_value = math.floor(note['x']*20)
            if note['type'] != 4:
                if tick-1 >= 0:
                    zones[tick-1][max((x_value-1, 0)):min((x_value+2, 20))] = 1
                zones[tick][max((x_value-1, 0)):min((x_value+2, 20))] = 1
                if tick+1 <= 16:
                    zones[tick+1][max((x_value-1, 0)):min((x_value+2, 20))] = 1
            else:
                zones[tick][x_value] = 1
            if note['type'] == 1:
                length = math.floor((note['hold_tick'])/60)
                for i in range(length):
                    zones[tick+i][max((x_value-1, 0)):min((x_value+2, 20))] = 1
    if page % 2 == 0:
        zones = np.flip(zones, 0)
    return zones

def execute_idea(note_list, possibility, page, prev_page, diff):
    # Move notes inward
    if possibility == 'in':
        # Middle is not empty
        if np.sum(prev_page[:,7:14]) > 0:
            return note_list, -1
        # Not enough space to move inwards
        left = 0
        right = 20
        for x in range(len(prev_page[0])):
            if x < 10 and np.sum(prev_page[:, x]) > 0:
                left = x
            elif x > 10 and np.sum(prev_page[:, x] > 0):
                right = x
                break
        if right-left <= 8:
            return note_list, -1
        # Execute idea:
        for note in note_list:
            if note['page_index'] == page:
                if note['x'] < 0.5:
                    note['x'] = left*0.05 + diff
                else:
                    note['x'] = right*0.05 - diff
                    
        return note_list, 1
    
    # Move notes outward
    if possibility == 'out':
        # Outsides are filled
        if np.sum(prev_page[:,0:3]) > 0 or np.sum(prev_page[:,18:21]) > 0:
            return note_list, -1
        # Execute idea:
        if np.sum(prev_page) == 0:
            for note in note_list:
                if note['page_index'] == page:
                    if note['x'] < 0.5:
                        note['x'] = 0.0
                    else:
                        note['x'] = 1.0
            return note_list, 1
        # Find outer-most x
        left = 9
        right = 11
#         for x in range(10):
#             if np.sum(prev_page[:, 9-x]) > 0:
#                 left = 9-x
#             if np.sum(prev_page[:, 11+x]) > 0:
#                 right = 11+x
        
        baseline = min(left, 20-right)
        
        for note in note_list:
            if note['page_index'] == page:
                    if note['x'] < 0.5:
                        note['x'] = baseline*0.05 - diff
                    else:
                        note['x'] = baseline*0.05 + diff
        return note_list, 1
    
    return note_list, -1

def execute_slide(note_list, possibility, page, prev_page):
    if possibility == 'left':
        for note in note_list:
            if note['page_index'] == page and note['type'] == 4:
                for other_note in note_list:
                    if other_note['next_id'] == note['id']:
                        if other_note['page_index'] != note['page_index']:
                            note['x'] = get_position(other_note, 'left', 0.15)
                        else:
                            note['x'] = get_position(other_note, 'left', 0.05)
        return note_list, 1
    if possibility == 'right':
        for note in note_list:
            if note['page_index'] == page and note['type'] == 4:
                for other_note in note_list:
                    if other_note['next_id'] == note['id']:
                        if other_note['page_index'] != note['page_index']:
                            note['x'] = get_position(other_note, 'right', 0.15)
                        else:
                            note['x'] = get_position(other_note, 'right', 0.05)
        return note_list, 1
    if possibility == 'in':
        for note in note_list:
            if note['page_index'] == page and note['type'] == 4:
                for other_note in note_list:
                    if other_note['next_id'] == note['id']:
                        if other_note['page_index'] != note['page_index']:
                            if other_note['x'] < 0.5:
                                note['x'] = get_position(other_note, 'right', 0.15)
                            else:
                                note['x'] = get_position(other_note, 'left', 0.15)
                        else:
                            if other_note['x'] < 0.5:
                                note['x'] = get_position(other_note, 'right', 0.05)
                            else:
                                note['x'] = get_position(other_note, 'left', 0.05)
        return note_list, 1
    if possibility == 'out':
        for note in note_list:
            if note['page_index'] == page and note['type'] == 4:
                for other_note in note_list:
                    if other_note['next_id'] == note['id']:
                        if other_note['page_index'] != note['page_index']:
                            if other_note['x'] < 0.5:
                                note['x'] = get_position(other_note, 'left', 0.15)
                            else:
                                note['x'] = get_position(other_note, 'right', 0.15)
                        else:
                            if other_note['x'] < 0.5:
                                note['x'] = get_position(other_note, 'left', 0.05)
                            else:
                                note['x'] = get_position(other_note, 'right', 0.05)
        return note_list, 1

    if possibility == 'zig-zag':
        for i, note in enumerate(note_list):
            if note['page_index'] == page and note['type'] == 4:
                for other_note in note_list:
                    if other_note['next_id'] == note['id']:
                        if other_note['page_index'] != note['page_index']:
                            if i % 2 == 0:
                                note['x'] = get_position(other_note, 'left', 0.15)
                            else:
                                note['x'] = get_position(other_note, 'right', 0.15)
                        else:
                            if i % 2 == 0:
                                note['x'] = get_position(other_note, 'left', 0.05)
                            else:
                                note['x'] = get_position(other_note, 'right', 0.05)
        return note_list, 1
        
    return note_list, -1
    
def get_position(prev_note, direction, diff):
    if direction == 'left':
        # If note is too close to 0.0 or 0.5, move right instead
        if (prev_note['x']-0.0) < diff or (prev_note['x'] > 0.5 and (prev_note['x']-0.55) < diff):
            return prev_note['x'] + diff
        else:
            return prev_note['x'] - diff
    else:
        # If note is too close to 1.0 or 0.5, move left instead
        if (1.0-prev_note['x']) < diff or (prev_note['x'] < 0.5 and (0.45-prev_note['x']) < diff):
            return prev_note['x'] - diff
        else:
            return prev_note['x'] + diff
        
    
def procedural_2(note_list, page_list):
    # Reset positions
    for note in note_list:
        if note['x'] < 0.5:
            note['x'] = 0.0
        else:
            note['x'] = 1.0
    
    move_ideas = ['in', 'out', 'left', 'right']
    slide_ideas = ['in', 'out', 'left', 'right', 'zig-zag']
    pages = list()
    for page in range(len(page_list)):
        pages.append(page_zones(note_list, page_list, page))
    for i, page in enumerate(page_list):
        shuffle(move_ideas)
        shuffle(slide_ideas)
        if i != 0:
            for idea in move_ideas:
                diff = 0.10
                temp_note_list, result = execute_idea(note_list, idea, i, pages[i-1], diff)
                if result == 1:
                    page_zone = page_zones(temp_note_list, page_list, i)
                    if np.sum(page_zone*pages[i-1]) == 0:
                        note_list = temp_note_list
                        pages[i] = page_zone
                        break
                    elif np.sum(page_zone*pages[i-1]) < np.sum(pages[i]*pages[i-1]):
                        note_list = temp_note_list
                        pages[i] = page_zone
                        
                diff = 0.15
                temp_note_list, result = execute_idea(note_list, idea, i, pages[i-1], diff)
                if result == 1:
                    page_zone = page_zones(temp_note_list, page_list, i)
                    if np.sum(page_zone*pages[i-1]) == 0:
                        note_list = temp_note_list
                        pages[i] = page_zone
                        break
                    elif np.sum(page_zone*pages[i-1]) < np.sum(pages[i]*pages[i-1]):
                        note_list = temp_note_list
                        pages[i] = page_zone
                        
            for j, idea in enumerate(slide_ideas):
                if j == 0:
                    note_list, result = execute_slide(note_list, idea, i, pages[i-1])
                    pages[i] = page_zones(temp_note_list, page_list, i)
                else:
                    temp_note_list, result = execute_slide(note_list, idea, i, pages[i-1])
                    if result == 1:
                        page_zone = page_zones(temp_note_list, page_list, i)
                        if np.sum(page_zone*pages[i-1]) == 0:
                            note_list = temp_note_list
                            pages[i] = page_zone
                            break
                        elif np.sum(page_zone*pages[i-1]) <= np.sum(pages[i]*pages[i-1]):
                            note_list = temp_note_list
                            pages[i] = page_zone
    return note_list

title = sys.argv[1]

file_path = 'Final_' + title + '.json'
file = open(file_path)
data = json.load(file)


data['note_list'] = procedural_2(data['note_list'], data['page_list'])
filename = 'Procedural_' + title + '.json'
with open(filename, 'w') as f:
    json.dump(data, f)