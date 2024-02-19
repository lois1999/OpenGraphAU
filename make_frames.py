import cv2
import argparse
import os 

def FrameCapture(path): 
  
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(".mp4"):
                # Path to video file 
                video_path = os.path.join(root, name)
                vidObj = cv2.VideoCapture(video_path) 

                video_name = video_path.split("/")[-1].strip(".mp4")
                save_path = ("/").join(video_path.split("/")[:-1])

                # Used as counter variable 
                count = 0
            
                # checks whether frames were extracted 
                success = 1
            
                while success: 
            
                    # vidObj object calls read 
                    # function extract frames 
                    success, image = vidObj.read() 
                    if success == 0:
                        break 

                    frame_name = "{}_frame{}.jpg".format(video_name, count)

                    # Saves the frames with frame-count 
                    cv2.imwrite(save_path + "/" + frame_name, image) 

                    count += 1

# ---------------------------------------------------------------------------------

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", "--input_dir", help="Path to the input directory")
    args = parser.parse_args()
    FrameCapture(args.input_dir)

