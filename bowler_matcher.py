import cv2
import dlib
import os
import numpy as np
import matplotlib.pyplot as plt

def crop_to_face_and_save(face_ref, reference_img):
    # Extract the face region from the reference image
    x, y, w, h = face_ref.left(), face_ref.top(), face_ref.width(), face_ref.height()
    face_img = reference_img[y:y+h, x:x+w]

    # Save the extracted face region
    cv2.imwrite('reference_face.jpg', face_img)

def preprocess_frame(frame, width=2500):

    #print('Channels: ', len(frame.shape))
    # if len(frame.shape) == 3:  # If image is color (has 3 channels)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height = int(frame.shape[0] * (width / frame.shape[1]))
    frame = cv2.resize(frame, (width, height))
    return frame


def match_bowler(reference_image_path, frames_folder, match_threshold=0.45):

    # Initialize face detector and face recognition model
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Shape predictor model
    face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')  # Face recognition model

    # Load the reference image and detect faces
    reference_img = cv2.imread(reference_image_path)
    reference_img = preprocess_frame(reference_img)
    if reference_img is None:
        print(f"Could not load the image {reference_image_path}. Please check the file.")
        return []
    
    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    faces_ref = detector(gray_ref)

    
    #print('Number of faces detected in the reference image: ', len(faces_ref))
    
    if len(faces_ref) == 0:
        return []
    
    # Assume there's only one face in the reference image (the bowler's face)
    face_ref = faces_ref[0]
    crop_to_face_and_save(face_ref, reference_img)
    
    shape_ref = sp(reference_img, face_ref)
    
    # Get the face descriptor (embedding) for the reference face
    face_descriptor_ref = face_rec_model.compute_face_descriptor(reference_img, shape_ref)

    matched_frames = []
    distances = []
    
    # Iterate through frames in the folder
    for frame_file in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to load frame: {frame_path}")
            continue
        frame = preprocess_frame(frame)
        
        if frame is None:
            continue
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces_frame = detector(gray_frame)
        print('Number of faces detected: ', len(faces_frame), 'in frame: ', frame_file, end=' ')
        for face in faces_frame:

            shape_frame = sp(frame, face)
            
            # Get the face descriptor (embedding) for the detected face in the frame
            face_descriptor_frame = face_rec_model.compute_face_descriptor(frame, shape_frame)
            
            # Compare the descriptors using Euclidean distance
            distance = np.linalg.norm(np.array(face_descriptor_ref) - np.array(face_descriptor_frame))
            distances.append(distance)

            print('Distance: ', distance) 
            # If the distance is below the threshold, it's considered a match
            if distance < match_threshold:
                
                frame_index = int(frame_file.split('_')[1].split('.')[0])
                matched_frames.append(frame_index)
                break  # Stop after finding the first matching face
    # print distasnce statistics aveage, median, min, max

    # save plot of distribution of distances
    
    plt.hist(distances, bins=100)
    plt.savefig('distance_distribution.png')
    plt.title(f'Distance Distribution\nAvg: {np.mean(distances):.3f}, Med: {np.median(distances):.3f}\nMin: {np.min(distances):.3f}, Max: {np.max(distances):.3f}')
    matched_frames.sort()
    return matched_frames

if __name__ == "__main__":
    import sys
    reference_image_path = sys.argv[1]
    frames_folder = sys.argv[2]

    matched_frames = match_bowler(reference_image_path, frames_folder)
    print(matched_frames)  # Print indices where the bowler appears

    # [10780, 13935, 13940, 13945, 17605, 3710, 6175]
