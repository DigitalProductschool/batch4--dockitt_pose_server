import logging
import math

import slidingwindow as sw

import cv2
import numpy as np
import tensorflow as tf
import time

from tf_pose import common
from tf_pose.common import CocoPart
from tf_pose.tensblur.smoother import Smoother

try:
    from tf_pose.pafprocess import pafprocess
except ModuleNotFoundError as e:
    print(e)
    print(
        'you need to build c++ library for pafprocess. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess')
    exit(-1)

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def _round(v):
    return int(round(v))


def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_face_box(self, img_w, img_h, mode=0):
        """
        Get Face box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :param mode:
        :return:
        """
        # SEE : https://github.com/ildoonet/tf-pose-estimation/blob/master/tf_pose/common.py#L13
        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _REye = CocoPart.REye.value
        _LEye = CocoPart.LEye.value
        _REar = CocoPart.REar.value
        _LEar = CocoPart.LEar.value

        _THRESHOLD_PART_CONFIDENCE = 0.2
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]

        is_nose, part_nose = _include_part(parts, _NOSE)
        if not is_nose:
            return None

        size = 0
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_neck:
            size = max(size, img_h * (part_neck.y - part_nose.y) * 0.8)

        is_reye, part_reye = _include_part(parts, _REye)
        is_leye, part_leye = _include_part(parts, _LEye)
        if is_reye and is_leye:
            size = max(size, img_w * (part_reye.x - part_leye.x) * 2.0)
            size = max(size,
                       img_w * math.sqrt((part_reye.x - part_leye.x) ** 2 + (part_reye.y - part_leye.y) ** 2) * 2.0)

        if mode == 1:
            if not is_reye and not is_leye:
                return None

        is_rear, part_rear = _include_part(parts, _REar)
        is_lear, part_lear = _include_part(parts, _LEar)
        if is_rear and is_lear:
            size = max(size, img_w * (part_rear.x - part_lear.x) * 1.6)

        if size <= 0:
            return None

        if not is_reye and is_leye:
            x = part_nose.x * img_w - (size // 3 * 2)
        elif is_reye and not is_leye:
            x = part_nose.x * img_w - (size // 3)
        else:  # is_reye and is_leye:
            x = part_nose.x * img_w - size // 2

        x2 = x + size
        if mode == 0:
            y = part_nose.y * img_h - size // 3
        else:
            y = part_nose.y * img_h - _round(size / 2 * 1.2)
        y2 = y + size

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        if mode == 0:
            return {"x": _round((x + x2) / 2),
                    "y": _round((y + y2) / 2),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}
        else:
            return {"x": _round(x),
                    "y": _round(y),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}

    def get_upper_body_box(self, img_w, img_h):
        """
        Get Upper body box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :return:
        """

        if not (img_w > 0 and img_h > 0):
            raise Exception("img size should be positive")

        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _RSHOULDER = CocoPart.RShoulder.value
        _LSHOULDER = CocoPart.LShoulder.value
        _THRESHOLD_PART_CONFIDENCE = 0.3
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]
        part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                       part.part_idx in [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]]

        if len(part_coords) < 5:
            return None

        # Initial Bounding Box
        x = min([part[0] for part in part_coords])
        y = min([part[1] for part in part_coords])
        x2 = max([part[0] for part in part_coords])
        y2 = max([part[1] for part in part_coords])

        # # ------ Adjust heuristically +
        # if face points are detcted, adjust y value

        is_nose, part_nose = _include_part(parts, _NOSE)
        is_neck, part_neck = _include_part(parts, _NECK)
        torso_height = 0
        if is_nose and is_neck:
            y -= (part_neck.y * img_h - y) * 0.8
            torso_height = max(0, (part_neck.y - part_nose.y) * img_h * 2.5)
        #
        # # by using shoulder position, adjust width
        is_rshoulder, part_rshoulder = _include_part(parts, _RSHOULDER)
        is_lshoulder, part_lshoulder = _include_part(parts, _LSHOULDER)
        if is_rshoulder and is_lshoulder:
            half_w = x2 - x
            dx = half_w * 0.15
            x -= dx
            x2 += dx
        elif is_neck:
            if is_lshoulder and not is_rshoulder:
                half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)
            elif not is_lshoulder and is_rshoulder:
                half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)

        # ------ Adjust heuristically -

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        return {"x": _round((x + x2) / 2),
                "y": _round((y + y2) / 2),
                "w": _round(x2 - x),
                "h": _round(y2 - y)}

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_mat, paf_mat):
        pafprocess.process_paf(peaks, heat_mat, paf_mat)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                    pafprocess.get_part_score(c_idx)
                )

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans


class TfPoseEstimator:
    # TODO : multi-scale

    def __init__(self, graph_path, target_size=(320, 240), tf_config=None):
        self.target_size = target_size

        # load graph
        logger.info('loading graph from %s(default size=%dx%d)' % (graph_path, target_size[0], target_size[1]))
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph, config=tf_config)

        # for op in self.graph.get_operations():
        #     print(op.name)
        # for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
        #     print(ts)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        self.tensor_heatMat = self.tensor_output[:, :, :, :19]
        self.tensor_pafMat = self.tensor_output[:, :, :, 19:]
        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')
        self.tensor_heatMat_up = tf.image.resize_area(self.tensor_output[:, :, :, :19], self.upsample_size,
                                                      align_corners=False, name='upsample_heatmat')
        self.tensor_pafMat_up = tf.image.resize_area(self.tensor_output[:, :, :, 19:], self.upsample_size,
                                                     align_corners=False, name='upsample_pafmat')
        smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0)
        gaussian_heatMat = smoother.get_output()

        max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        self.tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                     tf.zeros_like(gaussian_heatMat))

        self.heatMat = self.pafMat = None

        # warm-up
        self.persistent_sess.run(tf.variables_initializer(
            [v for v in tf.global_variables() if
             v.name.split(':')[0] in [x.decode('utf-8') for x in
                                      self.persistent_sess.run(tf.report_uninitialized_variables())]
             ])
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1], target_size[0]]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 2, target_size[0] // 2]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 4, target_size[0] // 4]
            }
        )

    def __del__(self):
        # self.persistent_sess.close()
        pass

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2 ** 8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    @staticmethod
    def draw_humans(npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        for human in humans:
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue

                # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
                cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

        return npimg

    # @staticmethod
    # def evaluate_flexion(humans):
    #     for human in humans:
    #
    #         l_shoulder = False
    #         r_shoulder = False
    #         l_elbow = False
    #         r_elbow = False
    #         l_wrist = False
    #         r_wrist = False
    #
    #         for i in range(common.CocoPart.Background.value):
    #
    #             if i not in human.body_parts.keys():
    #                 continue
    #
    #             if i in [1, 2, 3, 4, 5, 6, 7, 8, 11]:
    #                 if i is 2:
    #                     l_shoulder = human.body_parts[i]
    #                 if i is 3:
    #                     l_elbow = human.body_parts[i]
    #                 if i is 4:
    #                     l_wrist = human.body_parts[i]
    #                 if i is 5:
    #                     r_shoulder = human.body_parts[i]
    #                 if i is 6:
    #                     r_elbow = human.body_parts[i]
    #                 if i is 7:
    #                     r_wrist = human.body_parts[i]
    #
    #         if all([l_shoulder, l_elbow, l_wrist]):
    #             logger.info("We see full left arm.")
    #             logger.info(l_shoulder.x)
    #             logger.info(l_shoulder.y)
    #             logger.info(l_elbow.x)
    #             logger.info(l_elbow.y)
    #             logger.info(l_wrist.x)
    #             logger.info(l_wrist.y)
    #             logger.info("---------------------")
    #
    #         if all([r_shoulder, r_elbow, r_wrist]):
    #             logger.info("We see full right arm.")
    #             logger.info(r_shoulder.x)
    #             logger.info(r_shoulder.y)
    #             logger.info(r_elbow.x)
    #             logger.info(r_elbow.y)
    #             logger.info(r_wrist.x)
    #             logger.info(r_wrist.y)
    #             logger.info("---------------------")
    #
    #     return True

    ### Atomic Kittens ###
    #@staticmethod
    #def calculate_angle(upper,lower):
    #    norm_angle = np.dot(upper,lower) / (np.linalg.norm(upper) * np.linalg.norm(lower))
    #    angle = np.degrees(np.arccos(norm_angle))
    #    return angle

    @staticmethod
    def evaluate_flexion(npimg, humans, imgcopy=False):

        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}

        for human in humans:

            nose = False
            throat = False
            r_shoulder = False
            r_elbow = False
            r_wrist = False
            l_shoulder = False
            l_elbow = False
            l_wrist = False
            r_hip = False
            r_knee = False
            r_foot = False
            l_hip = False
            l_knee = False
            l_foot = False
            r_eye = False
            l_eye = False
            l_ear = False
            l_ear = False
            r_elbow_angle = 0
            l_elbow_angle = 0
            r_abduction_angle = 0
            l_abduction_angle = 0
            r_wrist_height = False
            l_wrist_height = False

            for i in range(common.CocoPart.Background.value):

                if i not in human.body_parts.keys():
                    continue

                if i in range(17):
                    if i is 0:
                        nose       = human.body_parts[i]
                    if i is 1:
                        throat     = human.body_parts[i]
                    if i is 2:
                        r_shoulder = human.body_parts[i]
                    if i is 3:
                        r_elbow    = human.body_parts[i]
                    if i is 4:
                        r_wrist    = human.body_parts[i]
                    if i is 5:
                        l_shoulder = human.body_parts[i]
                    if i is 6:
                        l_elbow    = human.body_parts[i]
                    if i is 7:
                        l_wrist    = human.body_parts[i]
                    if i is 8:
                        r_hip      = human.body_parts[i]
                    if i is 9:
                        r_knee     = human.body_parts[i]
                    if i is 10:
                        r_foot     = human.body_parts[i]
                    if i is 11:
                        l_hip      = human.body_parts[i]
                    if i is 12:
                        l_knee     = human.body_parts[i]
                    if i is 13:
                        l_foot     = human.body_parts[i]
                    if i is 14:
                        r_eye      = human.body_parts[i]
                    if i is 15:
                        l_eye      = human.body_parts[i]
                    if i is 16:
                        r_ear      = human.body_parts[i]
                    if i is 17:
                        l_ear      = human.body_parts[i]

            # Left Elbow Angle
            if all([l_shoulder, l_elbow, l_wrist]):
                l_shoulder_xy = np.array([l_shoulder.x, l_shoulder.y])
                l_elbow_xy = np.array([l_elbow.x, l_elbow.y])
                l_wrist_xy = np.array([l_wrist.x, l_wrist.y])
                l_upper_arm = l_shoulder_xy - l_elbow_xy
                l_lower_arm = l_wrist_xy - l_elbow_xy
                #print(calculate_angle(l_upper_arm,l_lower_arm))
                l_cos_angle = np.dot(l_upper_arm, l_lower_arm) / (
                    np.linalg.norm(l_upper_arm) * np.linalg.norm(l_lower_arm))
                l_angle = np.arccos(l_cos_angle)
                l_elbow_angle = np.degrees(l_angle)

            # Right Elbow Angle
            if all([r_shoulder, r_elbow, r_wrist]):
                r_shoulder_xy = np.array([r_shoulder.x, r_shoulder.y])
                r_elbow_xy = np.array([r_elbow.x, r_elbow.y])
                r_wrist_xy = np.array([r_wrist.x, r_wrist.y])
                r_upper_arm = r_shoulder_xy - r_elbow_xy
                r_lower_arm = r_wrist_xy - r_elbow_xy
                r_cos_angle = np.dot(r_upper_arm, r_lower_arm) / (
                    np.linalg.norm(r_upper_arm) * np.linalg.norm(r_lower_arm))
                r_angle = np.arccos(r_cos_angle)
                r_elbow_angle = np.degrees(r_angle)

            # Abduction Right
                # Body Core Calculated
            #if all([l_hip, r_hip, throat, r_shoulder, r_elbow, r_wrist]) and r_elbow_angle > 150:
                # Body Core
                #throat_xy = np.array([throat.x, throat.y])
                #l_hip_xy  = np.array([l_hip.x,  l_hip.y])
                #r_hip_xy  = np.array([r_hip.x,  r_hip.y])
                #m_hip_xy  = (l_hip_xy + r_hip_xy) / 2
                #body_core = m_hip_xy - throat_xy # or reverse: throat_xy - m_hip_xy
                # Body Core Fake (Vertical Line)
            if all([r_shoulder, r_elbow, r_wrist]) and r_elbow_angle > 150:
                body_core = np.array([0,-1])
                # Right Arm
                r_shoulder_xy = np.array([r_shoulder.x, r_shoulder.y])
                r_wrist_xy    = np.array([r_wrist.x, r_wrist.y])
                r_arm         = r_shoulder_xy - r_wrist_xy
                r_cos_angle = np.dot(r_arm, body_core) / (
                    np.linalg.norm(r_arm) * np.linalg.norm(body_core))
                r_angle = np.arccos(r_cos_angle)
                r_abduction_angle = np.degrees(r_angle)
                #print("Right Abduction: "+str(r_abduction_angle))

            # Abduction Left
                # Body Core Calculated
            #if all([l_hip, r_hip, throat, r_shoulder, r_elbow, r_wrist]) and r_elbow_angle > 150:
                # Body Core
                #throat_xy = np.array([throat.x, throat.y])
                #l_hip_xy  = np.array([l_hip.x,  l_hip.y])
                #r_hip_xy  = np.array([r_hip.x,  r_hip.y])
                #m_hip_xy  = (l_hip_xy + r_hip_xy) / 2
                #body_core = throat_xy - m_hip_xy
                # Body Core Fake (Vertical Line)
            if all([l_shoulder, l_elbow, l_wrist]) and l_elbow_angle > 150:
                body_core = np.array([0,-1])
                # Right Arm
                l_shoulder_xy = np.array([l_shoulder.x, l_shoulder.y])
                l_wrist_xy    = np.array([l_wrist.x, l_wrist.y])
                l_arm         = l_shoulder_xy - l_wrist_xy
                l_cos_angle = np.dot(l_arm, body_core) / (
                    np.linalg.norm(l_arm) * np.linalg.norm(body_core))
                l_angle = np.arccos(l_cos_angle)
                l_abduction_angle = np.degrees(l_angle)
                #print("Left Abduction: "+str(l_abduction_angle))




            ## Abduction Left
            #if all([l_hip, r_hip, throat, l_shoulder, l_elbow, l_wrist]) and l_elbow_angle > 150:
            #    # Body Core
            #    throat_xy = np.array([throat.x, throat.y])
            #    l_hip_xy  = np.array([l_hip.x,  l_hip.y])
            #    r_hip_xy  = np.array([r_hip.x,  r_hip.y])
            #    m_hip_xy  = (l_hip_xy + r_hip_xy) / 2
            #    body_core = throat_xy - m_hip_xy
            #    # Left Arm
            #    l_shoulder_xy = np.array([l_shoulder.x, l_shoulder.y])
            #    l_wrist_xy    = np.array([l_wrist.x, l_wrist.y])
            #    l_arm         = l_shoulder_xy - l_wrist_xy
            #    l_cos_angle = np.dot(l_arm, body_core) / (
            #        np.linalg.norm(l_arm) * np.linalg.norm(body_core))
            #    l_angle = np.arccos(l_cos_angle)
            #    l_abduction_angle = np.degrees(l_angle)
            #    #print("Left  Abduction: "+str(l_abduction_angle))

            # Position Hand
            if all([l_hip, r_hip, throat, r_wrist]):
                # Body Core
                throat_xy = np.array([throat.x, throat.y])
                l_hip_xy  = np.array([l_hip.x,  l_hip.y])
                r_hip_xy  = np.array([r_hip.x,  r_hip.y])
                m_hip_xy  = (l_hip_xy + r_hip_xy) / 2
                m_hip_w   = abs(l_hip.x - r_hip.x) # Hip Width
                torso_xy = m_hip_xy - throat_xy
                # Arm
                r_wrist_xy = np.array([r_wrist.x, r_wrist.y])
                # Distance between wrist and hip normalised by width of the hip in percent
                if m_hip_w == 0:
                    pass
                else:
                    r_wrist_dx = (r_wrist.x - r_hip.x) * 100 / m_hip_w 
                    if abs(r_wrist_dx) < 30:
                        # Height of the hand normalised to length of the upper body
                        r_wrist_height = (m_hip_xy[1] - r_wrist.y) * 100 / torso_xy[1]
                    else:
                        r_wrist_height = False
                        l_wrist_height = False

            # Position Right Hand
            if all([l_hip, r_hip, throat, l_wrist]):
                # Body Core
                throat_xy = np.array([throat.x, throat.y])
                l_hip_xy  = np.array([l_hip.x,  l_hip.y])
                r_hip_xy  = np.array([r_hip.x,  r_hip.y])
                m_hip_xy  = (l_hip_xy + r_hip_xy) / 2
                m_hip_w   = abs(l_hip.x - r_hip.x) # Hip Width
                torso_xy = m_hip_xy - throat_xy
                # Left Arm
                l_wrist_xy = np.array([l_wrist.x, l_wrist.y])
                # Distance between wrist and hip normalised by width of the hip in percent
                if m_hip_w == 0:
                    pass
                else:
                    l_wrist_dx = (l_wrist.x - l_hip.x) * 100 / m_hip_w 
                    if abs(l_wrist_dx) < 30:
                        # Height of the hand normalised to length of the upper body
                        l_wrist_height = (m_hip_xy[1] - l_wrist.y) * 100 / torso_xy[1]
                    else:
                        l_wrist_height = 0


            if r_elbow_angle:
                cv2.putText(npimg, "{0:.2f}".format(r_elbow_angle),
                            (int(r_elbow.x * image_w + 0.5), int(r_elbow.y * image_h + 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                # logger.info("Right ellbow: "+str(r_elbow_angle))
            if l_elbow_angle:
                cv2.putText(npimg, "{0:.2f}".format(l_elbow_angle),
                            (int(l_elbow.x * image_w + 0.5), int(l_elbow.y * image_h + 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            if r_abduction_angle:
                cv2.putText(npimg, "{0:.2f}".format(r_abduction_angle),
                            (int(r_shoulder.x * image_w + 0.5), int(r_shoulder.y * image_h + 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            if l_abduction_angle:
                cv2.putText(npimg, "{0:.2f}".format(l_abduction_angle),
                            (int(l_shoulder.x * image_w + 0.5), int(l_shoulder.y * image_h + 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                # logger.info("Left ellbow: "+str(l_elbow_angle))

            if r_wrist_height:
                cv2.putText(npimg, "{0:.2f}".format(r_wrist_height),
                            (int(r_wrist.x * image_w + 0.5), int(r_wrist.y * image_h + 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            if l_wrist_height:
                cv2.putText(npimg, "{0:.2f}".format(l_wrist_height),
                            (int(l_wrist.x * image_w + 0.5), int(l_wrist.y * image_h + 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)


        return npimg

        ### Atomic Kittens ###

    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(h), self.target_size[1] / float(w)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size, interpolation=cv2.INTER_CUBIC)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)

            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1], 0.2)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            window_step = scale[1]

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1],
                                  window_step)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped

    def inference(self, npimg, resize_to_default=True, upsample_size=1.0):
        if npimg is None:
            raise Exception('The image is not valid. Please check your image exists.')

        if resize_to_default:
            upsample_size = [int(self.target_size[1] / 8 * upsample_size), int(self.target_size[0] / 8 * upsample_size)]
        else:
            upsample_size = [int(npimg.shape[0] / 8 * upsample_size), int(npimg.shape[1] / 8 * upsample_size)]

        if self.tensor_image.dtype == tf.quint8:
            # quantize input image
            npimg = TfPoseEstimator._quantize_img(npimg)
            pass

        logger.debug('inference+ original shape=%dx%d' % (npimg.shape[1], npimg.shape[0]))
        img = npimg
        if resize_to_default:
            img = self._get_scaled_img(npimg, None)[0][0]
        peaks, heatMat_up, pafMat_up = self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up], feed_dict={
                self.tensor_image: [img], self.upsample_size: upsample_size
            })
        peaks = peaks[0]
        self.heatMat = heatMat_up[0]
        self.pafMat = pafMat_up[0]
        logger.debug('inference- heatMat=%dx%d pafMat=%dx%d' % (
            self.heatMat.shape[1], self.heatMat.shape[0], self.pafMat.shape[1], self.pafMat.shape[0]))

        t = time.time()
        humans = PoseEstimator.estimate_paf(peaks, self.heatMat, self.pafMat)
        logger.debug('estimate time=%.5f' % (time.time() - t))
        return humans


if __name__ == '__main__':
    import pickle

    f = open('./etcs/heatpaf1.pkl', 'rb')
    data = pickle.load(f)
    logger.info('size={}'.format(data['heatMat'].shape))
    f.close()

    t = time.time()
    humans = PoseEstimator.estimate_paf(data['peaks'], data['heatMat'], data['pafMat'])
    dt = time.time() - t;
    t = time.time()
    logger.info('elapsed #humans=%d time=%.8f' % (len(humans), dt))
