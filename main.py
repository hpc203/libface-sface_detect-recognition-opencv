import numpy as np
import cv2
import argparse
from itertools import product
from _testcapi import FLT_MIN

class YuNet:
    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, keepTopK=750):
        self._model = cv2.dnn.readNet(modelPath)
        self._inputNames = ''
        self._outputNames = ['loc', 'conf', 'iou']
        self._inputSize = inputSize # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._keepTopK = keepTopK
        self._min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self._steps = [8, 16, 32, 64]
        self._variance = [0.1, 0.2]

        # Generate priors
        self._priorGen()

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backend):
        self._model.setPreferableBackend(backend)

    def setTarget(self, target):
        self._model.setPreferableTarget(target)

    def setInputSize(self, input_size):
        self._inputSize = input_size # [w, h]
        # Regenerate priors
        self._priorGen()

    def infer(self, image):
        assert image.shape[0] == self._inputSize[1], '{} (height of input image) != {} (preset height)'.format(image.shape[0], self._inputSize[1])
        assert image.shape[1] == self._inputSize[0], '{} (width of input image) != {} (preset width)'.format(image.shape[1], self._inputSize[0])

        # Preprocess
        inputBlob = cv2.dnn.blobFromImage(image)

        # Forward
        self._model.setInput(inputBlob, self._inputNames)
        outputBlob = self._model.forward(self._outputNames)

        # Postprocess
        results = self._postprocess(outputBlob)

        return results

    def _postprocess(self, outputBlob):
        # Decode
        dets = self._decode(outputBlob)

        # NMS
        keepIdx = cv2.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK
        ) # box_num x class_num
        if len(keepIdx) > 0:
            dets = dets[keepIdx]
            # dets = np.squeeze(dets, axis=1)
            return dets[:self._keepTopK]
        else:
            return np.empty(shape=(0, 15))

    def _priorGen(self):
        w, h = self._inputSize
        feature_map_2th = [int(int((h + 1) / 2) / 2),
                           int(int((w + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2),
                           int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2),
                           int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2),
                           int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2),
                           int(feature_map_5th[1] / 2)]

        feature_maps = [feature_map_3th, feature_map_4th,
                        feature_map_5th, feature_map_6th]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self._min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])): # i->h, j->w
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self._steps[k] / w
                    cy = (i + 0.5) * self._steps[k] / h

                    priors.append([cx, cy, s_kx, s_ky])
        self.priors = np.array(priors, dtype=np.float32)

    def _decode(self, outputBlob):
        loc, conf, iou = outputBlob
        # get score
        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]
        # clamp
        _idx = np.where(iou_scores < 0.)
        iou_scores[_idx] = 0.
        _idx = np.where(iou_scores > 1.)
        iou_scores[_idx] = 1.
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = np.array(self._inputSize)

        # get bboxes
        bboxes = np.hstack((
            (self.priors[:, 0:2] + loc[:, 0:2] * self._variance[0] * self.priors[:, 2:4]) * scale,
            (self.priors[:, 2:4] * np.exp(loc[:, 2:4] * self._variance)) * scale
        ))
        # (x_c, y_c, w, h) -> (x1, y1, w, h)
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        # get landmarks
        landmarks = np.hstack((
            (self.priors[:, 0:2] + loc[:,  4: 6] * self._variance[0] * self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2] + loc[:,  6: 8] * self._variance[0] * self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2] + loc[:,  8:10] * self._variance[0] * self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2] + loc[:, 10:12] * self._variance[0] * self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2] + loc[:, 12:14] * self._variance[0] * self.priors[:, 2:4]) * scale
        ))

        dets = np.hstack((bboxes, landmarks, scores))
        return dets

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in results:
        bbox = det[0:4].astype(np.int32)
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv2.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv2.circle(output, landmark, 2, landmark_color[idx], 2)
    return output

class SFace:
    def __init__(self, modelPath):
        self._model = cv2.dnn.readNet(modelPath)
        self._input_size = [112, 112]
        self._dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        self._dst_mean = np.array([56.0262, 71.9008], dtype=np.float32)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backend_id):
        self._model.setPreferableBackend(backend_id)

    def setTarget(self, target_id):
        self._model.setPreferableTarget(target_id)

    def _preprocess(self, image, bbox):
        aligned_image = self._alignCrop(image, bbox)
        return cv2.dnn.blobFromImage(aligned_image)

    def infer(self, image, bbox):
        # Preprocess
        # inputBlob = self._preprocess(image, bbox)
        inputBlob = cv2.dnn.blobFromImage(self._alignCrop(image, bbox))
        # Forward
        self._model.setInput(inputBlob)
        outputBlob = self._model.forward()

        # Postprocess
        results = outputBlob / cv2.norm(outputBlob)
        return results

    def match(self, image1, face1, image2, face2, dis_type=0):
        feature1 = self.infer(image1, face1)
        feature2 = self.infer(image2, face2)

        if dis_type == 0: # COSINE
            return np.sum(feature1 * feature2)
        elif dis_type == 1: # NORM_L2
            return cv2.norm(feature1, feature2)
        else:
            raise NotImplementedError()

    def _alignCrop(self, image, face):
        # Retrieve landmarks
        if face.shape[-1] == (4 + 5 * 2):
            landmarks = face[4:].reshape(5, 2)
        else:
            raise NotImplementedError()
        warp_mat = self._getSimilarityTransformMatrix(landmarks)
        aligned_image = cv2.warpAffine(image, warp_mat, self._input_size, flags=cv2.INTER_LINEAR)
        return aligned_image

    def _getSimilarityTransformMatrix(self, src):
        # compute the mean of src and dst
        src_mean = np.array([np.mean(src[:, 0]), np.mean(src[:, 1])], dtype=np.float32)
        dst_mean = np.array([56.0262, 71.9008], dtype=np.float32)
        # subtract the means from src and dst
        src_demean = src.copy()
        src_demean[:, 0] = src_demean[:, 0] - src_mean[0]
        src_demean[:, 1] = src_demean[:, 1] - src_mean[1]
        dst_demean = self._dst.copy()
        dst_demean[:, 0] = dst_demean[:, 0] - dst_mean[0]
        dst_demean[:, 1] = dst_demean[:, 1] - dst_mean[1]

        A = np.array([[0., 0.], [0., 0.]], dtype=np.float64)
        for i in range(5):
            A[0][0] += dst_demean[i][0] * src_demean[i][0]
            A[0][1] += dst_demean[i][0] * src_demean[i][1]
            A[1][0] += dst_demean[i][1] * src_demean[i][0]
            A[1][1] += dst_demean[i][1] * src_demean[i][1]
        A = A / 5

        d = np.array([1.0, 1.0], dtype=np.float64)
        if A[0][0] * A[1][1] - A[0][1] * A[1][0] < 0:
            d[1] = -1

        T = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        s, u, vt = cv2.SVDecomp(A)
        smax = s[0][0] if s[0][0] > s[1][0] else s[1][0]
        tol = smax * 2 * FLT_MIN
        rank = int(0)
        if s[0][0] > tol:
            rank += 1
        if s[1][0] > tol:
            rank += 1
        det_u = u[0][0] * u[1][1] - u[0][1] * u[1][0]
        det_vt = vt[0][0] * vt[1][1] - vt[0][1] * vt[1][0]
        if rank == 1:
            if det_u * det_vt > 0:
                uvt = np.matmul(u, vt)
                T[0][0] = uvt[0][0]
                T[0][1] = uvt[0][1]
                T[1][0] = uvt[1][0]
                T[1][1] = uvt[1][1]
            else:
                temp = d[1]
                d[1] = -1
                D = np.array([[d[0], 0.0], [0.0, d[1]]], dtype=np.float64)
                Dvt = np.matmul(D, vt)
                uDvt = np.matmul(u, Dvt)
                T[0][0] = uDvt[0][0]
                T[0][1] = uDvt[0][1]
                T[1][0] = uDvt[1][0]
                T[1][1] = uDvt[1][1]
                d[1] = temp
        else:
            D = np.array([[d[0], 0.0], [0.0, d[1]]], dtype=np.float64)
            Dvt = np.matmul(D, vt)
            uDvt = np.matmul(u, Dvt)
            T[0][0] = uDvt[0][0]
            T[0][1] = uDvt[0][1]
            T[1][0] = uDvt[1][0]
            T[1][1] = uDvt[1][1]

        var1 = 0.0
        var2 = 0.0
        for i in range(5):
            var1 += src_demean[i][0] * src_demean[i][0]
            var2 += src_demean[i][1] * src_demean[i][1]
        var1 /= 5
        var2 /= 5

        scale = 1.0 / (var1 + var2) * (s[0][0] * d[0] + s[1][0] * d[1])
        TS = [
            T[0][0] * src_mean[0] + T[0][1] * src_mean[1],
            T[1][0] * src_mean[0] + T[1][1] * src_mean[1]
        ]
        T[0][2] = dst_mean[0] - scale * TS[0]
        T[1][2] = dst_mean[1] - scale * TS[1]
        T[0][0] *= scale
        T[0][1] *= scale
        T[1][0] *= scale
        T[1][1] *= scale
        return np.array([
            [T[0][0], T[0][1], T[0][2]],
            [T[1][0], T[1][1], T[1][2]]
        ], dtype=np.float64)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
    parser.add_argument('--imgpath', type=str, default='selfie.jpg', help='Path to the input image')
    parser.add_argument('--detect_modelpath', type=str, default='weights/face_detection_yunet.onnx', help='Path to the face detect model.')
    parser.add_argument('--conf_threshold', type=float, default=0.9,
                        help='Filter out faces of confidence < conf_threshold.')
    parser.add_argument('--nms_threshold', type=float, default=0.3,
                        help='Suppress bounding boxes of iou >= nms_threshold.')
    parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
    parser.add_argument('--keep_top_k', type=int, default=750, help='Keep keep_top_k bounding boxes after NMS.')
    parser.add_argument('--rec_modelpath', type=str, default='weights/face_detection_yunet.onnx',
                        help='Path to the face detect model.')
    parser.add_argument('--img1path', type=str, default='telangpu.png', help='Path to the input image1 to match')
    parser.add_argument('--img2path', type=str, default='telangpu2.png', help='Path to the input image2 to match')
    parser.add_argument('--dis_type', type=int, choices=[0, 1], default=0, help='Distance type. 0: cosine, 1: norm_l1.')
    args = parser.parse_args()

    detector = YuNet(args.detect_modelpath, confThreshold=args.conf_threshold, nmsThreshold=args.nms_threshold, topK=args.top_k, keepTopK=args.keep_top_k)
    srcimg = cv2.imread(args.imgpath)
    detector.setInputSize([srcimg.shape[1], srcimg.shape[0]])
    results = detector.infer(srcimg)
    srcimg = visualize(srcimg, results)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    recognizer = SFace(args.rec_modelpath)
    img1 = cv2.imread(args.img1path)
    img2 = cv2.imread(args.img2path)

    detector.setInputSize([img1.shape[1], img1.shape[0]])
    face1 = detector.infer(img1)
    assert face1.shape[0] > 0, 'Cannot find a face in {}'.format(args.input1)
    detector.setInputSize([img2.shape[1], img2.shape[0]])
    face2 = detector.infer(img2)
    assert face2.shape[0] > 0, 'Cannot find a face in {}'.format(args.input2)

    distance = recognizer.match(img1, face1[0][:-1], img2, face2[0][:-1], args.dis_type)
    if args.dis_type == 0:
        dis_type = 'Cosine'
        threshold = 0.363
        result = 'same identity' if distance >= threshold else 'different identity'
    else:
        dis_type = 'Norm-L2'
        threshold = 1.128
        result = 'same identity' if distance <= threshold else 'different identity'
    print('Using {} distance, threshold {}: {}.'.format(dis_type, threshold, result))