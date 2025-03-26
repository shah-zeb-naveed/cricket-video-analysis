import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def merge_times(times):
    # Sort the times list to ensure we process them in order
    #times.sort()
    
    # Result list to store merged times
    merged_times = []
    
    # Start by adding the first time to the merged list
    current_time = times[0]
    
    # Iterate over the remaining times
    for time in times[1:]:
        # If the time is within 1 second of the current time, merge it
        if time - current_time <= 1:
            current_time = min(current_time, time)  # Merge the times by taking the earlier one
        else:
            # If it's not within 1 second, add the current group to the result
            merged_times.append(current_time)
            current_time = time  # Start a new group
    
    # Append the last merged time
    merged_times.append(current_time)
    
    return merged_times



def get_release_times(y, sr, percentile=99.50):
    # Compute Short-Time Fourier Transform (STFT)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Identify sudden peaks in amplitude
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks, _ = find_peaks(onset_env, height=np.percentile(onset_env, percentile))  # Detect strong peaks

    # Convert peak indices to time
    release_times = librosa.frames_to_time(peaks, sr=sr)
    return release_times

def write_list_to_file(filename, data):
    """
    Write a list of elements to a text file, each element on a new line.

    :param filename: The name of the file to write to.
    :param data: The list of data to write to the file.
    """
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")

def read_list_from_file(filename):
    """
    Read a list from a text file, assuming each element is on a new line.

    :param filename: The name of the file to read from.
    :return: A list of elements read from the file.
    """
    with open(filename, 'r') as file:
        return [float(line.strip()) for line in file.readlines()]
    

def pair_bowler_faces_with_batting(peaks, bowler_faces):
    """
    Pair bowler face timestamps with the immediately prior batting peaks.
    
    :param peaks: List of batting peak times (sorted)
    :param bowler_faces: List of bowler face timestamps (sorted)
    :return: List of paired batting peak times for each bowler face timestamp
    """
    paired_times = {}
    peak_idx = 0  # Pointer to the last unpaired batting peak
    
    # Iterate through each bowler face frame
    for bowler_face in bowler_faces:
        # Move the peak_idx forward to find the last batting peak that occurred before the bowler face time
        while peak_idx < len(peaks) and peaks[peak_idx] <= bowler_face:
            peak_idx += 1
        
        # The previous peak (i.e., peak_idx - 1) is the last peak before the bowler face timestamp
        # If peak_idx > 0, it means there was at least one peak before this bowler face frame
        if peak_idx > 0:
            paired_times[bowler_face] = peaks[peak_idx - 1]
        else:
            paired_times[bowler_face] = None  # If no peaks before the bowler face, append None (or handle as needed)
    
    return paired_times