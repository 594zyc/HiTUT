import glob, cv2
import numpy as np
import subprocess
from gen.constants import *


class VideoSaver(object):

    def __init__(self, frame_rate=VIDEO_FRAME_RATE):
        self.frame_rate = frame_rate

    def save(self, image_path, save_path):
        subprocess.call(["ffmpeg -r %d -pattern_type glob -y -i '%s' -c:v libx264 -pix_fmt yuv420p '%s'" %
                         (self.frame_rate, image_path, save_path)], shell=True)



def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image.
    """
    if mask is None:
        return image
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def save_frame(frame, visualize_info, repeat_num, draw_mask=True):
    img = np.array(frame)
    if draw_mask and visualize_info['mask'] is not None:
        img = apply_mask(img, visualize_info['mask'], color=(0, 1, 0))
        img = np.array(img)
        xmin, ymin, xmax, ymax = visualize_info['bbox']
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        cv2.putText(img, visualize_info['class'], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    sg_type = caption_map.get(visualize_info['high_action'], visualize_info['high_action'])
    sg = '%s(%s)'%(sg_type, visualize_info['high_arg'])
    sg_disp = 'SG%d: %s'%(visualize_info['high_idx']+1, sg)
    cv2.putText(img, sg_disp, (5,20), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,0,255), 1)

    action_type = caption_map.get(visualize_info['low_action'], visualize_info['low_action'])
    if visualize_info['low_arg'] not in [None, 'None']:
        a_str = '%s(%s)'%(action_type, visualize_info['low_arg'])
    else:
        a_str = '%s'%(action_type)
    a_disp = 'A%d: %s'%(visualize_info['global_step']+1, a_str)
    cv2.putText(img, a_disp, (5,40), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,0,255), 1)

    for i in range(repeat_num):
        im_ind = len(glob.glob(visualize_info['save_path'] + '/*.png'))
        cv2.imwrite(visualize_info['save_path'] + '/%05d.png' %im_ind, img)

    # cv2.namedWindow('full', cv2.WINDOW_NORMAL)
    # cv2.moveWindow('full', 20, 20)
    # cv2.imshow('full', img)
    # cv2.waitKey(300)


def save_frame_failure(frame, visualize_info):
    img = np.array(frame)

    sg_type = caption_map.get(visualize_info['high_action'], visualize_info['high_action'])
    sg = '%s(%s)'%(sg_type, visualize_info['high_arg'])
    sg_disp = 'SG%d: %s'%(visualize_info['high_idx']+1, sg)
    cv2.putText(img, sg_disp, (5,20), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,0,255), 1)

    action_type = caption_map.get(visualize_info['low_action'], visualize_info['low_action'])
    if visualize_info['low_arg'] not in [None, 'None']:
        a_str = '%s(%s)'%(action_type, visualize_info['low_arg'])
    else:
        a_str = '%s'%(action_type)
    a_disp = 'A%d: %s'%(visualize_info['global_step']+1, a_str)
    cv2.putText(img, a_disp, (5,40), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,0,255), 1)


    cv2.putText(img, 'Failed!', (5,60), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,0,255), 1)
    for i in range(5):
        im_ind = len(glob.glob(visualize_info['save_path'] + '/*.png'))
        cv2.imwrite(visualize_info['save_path'] + '/%05d.png' %im_ind, img)

    cv2.putText(img, 'Backtrack to SG%d'%(visualize_info['high_idx']), (5,80), cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,0,255), 1)
    for i in range(10):
        im_ind = len(glob.glob(visualize_info['save_path'] + '/*.png'))
        cv2.imwrite(visualize_info['save_path'] + '/%05d.png' %im_ind, img)

    # cv2.imshow('full', img)
    # cv2.waitKey(300)


def save_frames_before_action(frame, action, visualize_info):
    a = action['action']
    if 'MoveAhead' in a or 'Rotate' in a or 'Look' in a:
        repeat_num = 1
    else:
        repeat_num = SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[a][BEFORE]
    save_frame(frame, visualize_info, repeat_num)


def save_frames_after_action(events, action, visualize_info):
    a = action['action']
    if 'MoveAhead' in a or 'Rotate' in a or 'Look' in a:
        for event in events:
            # im_ind = len(glob.glob(save_path + '/*.png'))
            # cv2.imwrite(save_path + '/%09d.png' % im_ind, event.frame[:, :, ::-1])
            save_frame(event.frame[:, :, ::-1], visualize_info, 1, draw_mask=False)

    else:
        repeat_num = SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[a][AFTER]
        assert len(events) == 1
        frame = events[0].frame[:, :, ::-1]
        save_frame(frame, visualize_info, repeat_num, draw_mask=False)


caption_map = {
    'NoOp': 'End',
    'GotoLocation': 'Goto',
    'PickupObject': 'Pickup',
    'PutObject': 'Put',
    'SliceObject': 'Slice',
    'CoolObject': "Cool",
    'HeatObject': "Heat",
    'CleanObject': 'Clean',
    'ToggleObject': 'Toggle',
    'OpenObject': 'Open',
    'CloseObject': 'Close',
    'ToggleObjectOn': 'TurnOn',
    'ToggleObjectOff': 'TurnOff',
}