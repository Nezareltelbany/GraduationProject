import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimizations for TensorFlow
import cv2
import numpy as np
import glob
import os.path as osp
from insightface.model_zoo import model_zoo

class LandmarkModel():
    def __init__(self, name, root='./checkpoints'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))  # Find ONNX model files
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_') > 0:  # Skip self-generated model files
                continue
            model = model_zoo.get_model(onnx_file)  # Load model from ONNX file
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model  # Store model in dictionary
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models  # Ensure detection model is present
        self.det_model = self.models['detection']  # Set detection model

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size)  # Prepare detection model
            else:
                model.prepare(ctx_id)  # Prepare other models

    def get(self, img, max_num=0):
        if img is None:
            print("Error: Input image is None")
            return None

        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        if bboxes is None or kpss is None:
            print("Error: Detection failed, bounding boxes or key points are None")
            return None
        
        if bboxes.shape[0] == 0:
            print("No faces detected")
            return None

        det_score = bboxes[..., 4]
        best_index = np.argmax(det_score)

        kps = None
        if kpss is not None:
            kps = kpss[best_index]

        return kps

    def gets(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        return kpss
