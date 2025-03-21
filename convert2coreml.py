import argparse
import json
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow as tf

from coremltools.converters.keras import convert as convert_keras_to_coreml
from coremltools.models.utils import _convert_neural_network_weights_to_fp16 as convert_neural_network_weights_to_fp16
from coremltools.models.utils import save_spec
from coremltools.proto import NeuralNetwork_pb2

from maskrcnn.model import Config
from maskrcnn.model import MaskRCNNModel

def export_models(config,
                  mask_rcnn_model,
                  classifier_model,
                  mask_model,
                  export_main_path,
                  export_mask_path,
                  export_classifier_path):
    license = "MIT"
    author = "Édouard Lavery-Plante"

    def convert_proposal(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "ProposalLayer"
        params.description = "Proposes regions of interests and performs NMS."

        param1 = params.parameters.add()
        param1.key = "bboxStdDev_count"
        param1.value.intValue = len(layer.bounding_box_std_dev)

        for idx, value in enumerate(layer.bounding_box_std_dev):
            param1 = params.parameters.add()
            param1.key = "bboxStdDev_" + str(idx)
            param1.value.doubleValue = value

        param1 = params.parameters.add()
        param1.key = "preNMSMaxProposals"
        param1.value.intValue = layer.pre_nms_max_proposals

        param1 = params.parameters.add()
        param1.key = "maxProposals"
        param1.value.intValue = layer.max_proposals
        
        param1 = params.parameters.add()
        param1.key = "nmsIOUThreshold"
        param1.value.doubleValue = layer.nms_threshold

        # params.parameters["bboxStdDev_count"].intValue = len(layer.bounding_box_std_dev)
        # for idx, value in enumerate(layer.bounding_box_std_dev):
        #     params.parameters["bboxStdDev_" + str(idx)].doubleValue = value
        # params.parameters["preNMSMaxProposals"].intValue = layer.pre_nms_max_proposals
        # params.parameters["maxProposals"].intValue = layer.max_proposals
        # params.parameters["nmsIOUThreshold"].doubleValue = layer.nms_threshold
        return params

    def convert_pyramid(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "PyramidROIAlignLayer"
        # params.parameters["poolSize"].intValue = layer.pool_shape[0]
        # params.parameters["imageWidth"].intValue = layer.image_shape[0]
        # params.parameters["imageHeight"].intValue = layer.image_shape[1]
        params.description = "Extracts feature maps based on the regions of interest."

        param1 = params.parameters.add()
        param1.key = "poolSize"
        param1.value.intValue = layer.pool_shape[0]

        param1 = params.parameters.add()
        param1.key = "imageWidth"
        param1.value.intValue = layer.image_shape[0]

        param1 = params.parameters.add()
        param1.key = "imageHeight"
        param1.value.intValue = layer.image_shape[1]

        return params

    def convert_time_distributed_classifier(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "TimeDistributedClassifierLayer"
        return params

    def convert_time_distributed(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "TimeDistributedMaskLayer"
        params.description = "Applies the Mask graph to each detections along the time dimension."
        return params

    def convert_detection(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "DetectionLayer"
        # params.parameters["bboxStdDev_count"].intValue = len(layer.bounding_box_std_dev)
        # for idx, value in enumerate(layer.bounding_box_std_dev):
        #     params.parameters["bboxStdDev_" + str(idx)].doubleValue = value
        # params.parameters["maxDetections"].intValue = layer.max_detections
        # params.parameters["scoreThreshold"].doubleValue = layer.detection_min_confidence
        # params.parameters["nmsIOUThreshold"].doubleValue = layer.detection_nms_threshold
        params.description = "Outputs detections based on confidence and performs NMS."

        param1 = params.parameters.add()
        param1.key = "bboxStdDev_count"
        param1.value.intValue = len(layer.bounding_box_std_dev)

        for idx, value in enumerate(layer.bounding_box_std_dev):
            param1 = params.parameters.add()
            param1.key = "bboxStdDev_" + str(idx)
            param1.value.doubleValue = value

        param1 = params.parameters.add()
        param1.key = "maxDetections"
        param1.value.intValue = layer.max_detections

        param1 = params.parameters.add()
        param1.key = "scoreThreshold"
        param1.value.doubleValue = layer.detection_min_confidence

        param1 = params.parameters.add()
        param1.key = "nmsIOUThreshold"
        param1.value.doubleValue = layer.detection_nms_threshold

        return params

    mask_rcnn_model = convert_keras_to_coreml(mask_rcnn_model,
                                              input_names=["image"],
                                              image_input_names=['image'],
                                              output_names=["detections", "mask"],
                                              red_bias=-123.7,
                                              green_bias=-116.8,
                                              blue_bias=-103.9,
                                              add_custom_layers=True,
                                              custom_conversion_functions={
                                                  "ProposalLayer": convert_proposal,
                                                  "PyramidROIAlign": convert_pyramid,
                                                  "TimeDistributedClassifier": convert_time_distributed_classifier,
                                                  "TimeDistributedMask": convert_time_distributed,
                                                  "DetectionLayer": convert_detection})

    mask_rcnn_model.author = author
    mask_rcnn_model.license = license
    mask_rcnn_model.short_description = "Mask-RCNN"
    mask_rcnn_model.input_description["image"] = "Input image"
    mask_rcnn_model.output_description["detections"] = "Detections (y1,x1,y2,x2,classId,score)"
    mask_rcnn_model.output_description["mask"] = "Masks for the detections"
    half_model = convert_neural_network_weights_to_fp16(mask_rcnn_model)
    half_spec = half_model.get_spec()
    save_spec(half_spec, export_main_path)

    mask_model_coreml = convert_keras_to_coreml(mask_model,
                                                input_names=["feature_map"],
                                                output_names=["masks"])
    mask_model_coreml.author = author
    mask_model_coreml.license = license
    mask_model_coreml.short_description = "Generates a mask for each class for a given feature map"
    mask_model_coreml.input_description["feature_map"] = "Fully processed feature map, ready for mask generation."
    mask_model_coreml.output_description["masks"] = "Masks corresponding to each class"
    half_mask_model = convert_neural_network_weights_to_fp16(mask_model_coreml)
    half_mask_spec = half_mask_model.get_spec()
    save_spec(half_mask_spec, export_mask_path)

    classifier_model_coreml = convert_keras_to_coreml(classifier_model,
                                                      input_names=["feature_map"],
                                                      output_names=["probabilities", "bounding_boxes"])

    classifier_model_coreml.author = author
    classifier_model_coreml.license = license
    classifier_model_coreml.short_description = ""
    classifier_model_coreml.input_description["feature_map"] = "Fully processed feature map, ready for classification."
    half_classifier_model = convert_neural_network_weights_to_fp16(classifier_model_coreml)
    half_classifier_spec = half_classifier_model.get_spec()
    save_spec(half_classifier_spec, export_classifier_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_path',
        help='Path to config file',
        default="maskrcnn_model/floorplan/model/config.json",
        required=False
    )

    parser.add_argument(
        '--weights_path',
        help='Path to weights file',
        default="maskrcnn_model/floorplan/model/mask_rcnn_floorplan_0013.h5",
        required=False
    )

    parser.add_argument(
        '--export_path',
        help='Path to export main file',
        default="",
        required=False
    )

    args = parser.parse_args()
    params = args.__dict__
    
    config_path = params.pop('config_path')
    weights_path = params.pop('weights_path')
    export_path = params.pop('export_path')
    if export_path == '':
        export_path =  os.path.join( os.path.dirname(weights_path), os.path.basename(weights_path)+'_coreml')
        os.makedirs(export_path, exist_ok=True)

    export_main_path = os.path.join(export_path, 'MaskRCNN.mlmodel')
    export_mask_path = os.path.join(export_path, 'Mask.mlmodel')
    export_classifier_path = os.path.join(export_path, 'Classifier.mlmodel')
    #TODO: remove and generate instead
    export_anchors_path = os.path.join(export_path, 'anchors.bin')

    
    config = Config()
    with open(config_path) as file:
        config_dict = json.load(file)
        config.__dict__.update(config_dict)

    print('weights_path: ', weights_path)

    model = MaskRCNNModel(config_path, initial_keras_weights=weights_path)

    mask_rcnn_model, classifier_model, mask_model, anchors = model.get_trained_keras_models()
    export_models(config, mask_rcnn_model, classifier_model, mask_model, export_main_path, export_mask_path,
                  export_classifier_path)
    anchors.tofile(export_anchors_path)


