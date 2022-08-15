
from PIL import Image
import numpy as np
import cv2
from PIL import Image
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
from keras.models import model_from_json


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    # Get image ready for feeding into model
    pic = Image.open('image')
    pic.size[0]=80
    pic.size[1]=160
    small_img = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3)
    small_img = small_img[None, :, :, :]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 50:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image

    imgray = cv2.cvtColor(lane_drawn, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray.astype(np.uint8), 20, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(lane_drawn, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)
        # draw the book contour (in green)
        cv2.rectangle(lane_drawn, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pic = Image.open("image")
        pic.size[0] = 80
        pic.size[1] = 160
       
    lane_image = lane_drawn.reshape(80, 160, 3)
    # cv2.polylines(lane_image,average_position_left_lower,False,(255,0,0))

    average_position_left_lower = (0, 0)
    average_position_left_upper = (0, 0)
    average_position_right_lower = (0, 0)
    average_position_right_upper = (0, 0)
    # print("R", lane_image[:, :, 0])

    # cv2.imshow("laneImage",lane_image)
    # print("G",lane_image[x, y, 1])
    # print("B",lane_image[x, y, 2])

    # print(x,y)
    # cv2.waitKey(1)

    # Merge the lane drawing onto the original image
    cv2.imshow("laneImage", lane_image)
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    cv2.imshow("Live", result)
    cv2.waitKey(1)
    return result





if __name__ == '__main__':
    
    # Load Keras model
    json_file = open('C:\Users\Administrator\Desktop\MY_PROJECT_LANE-20220813T041556Z-001\MY_PROJECT_LANE\full_CNN_model.json', 'r')
    json_model = json_file.read()
    json_file.close()
    model = model_from_json(json_model)
    model.load_weights(r'C:\Users\Administrator\Desktop\MY_PROJECT_LANE-20220813T041556Z-001\MY_PROJECT_LANE\full_CNN_model.h5')
    # Create lanes object
    lanes = Lanes()
    # Where to save the output video
    vid_output = 'proj_reg_vid.mp4'
    # Location of the input video
    clip1 = VideoFileClip(r'C:\Users\Administrator\Desktop\MY_PROJECT_LANE-20220813T041556Z-001\MY_PROJECT_LANE\project_video.MP4')
    # Create the clip
    vid_clip = clip1.fl_image(road_lines)
    vid_clip.write_videofile(vid_output, audio=False)


