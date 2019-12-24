﻿using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgprocModule;
using System.Collections.Generic;
using UnityEngine;


//  model https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus
public class OpenCVDnnEmotionFerPlusExample : OpenCVDnnModuleExampleBase
{

    protected override Mat PreProcess(Mat img)
    {
        // Input
        // The model expects input of the shape(Nx1x64x64), where N is the batch size.

        // Preprocessing
        // Given a path image_path to the image you would like to score:

        //import numpy as np
        //from PIL import Image

        //def preprocess(image_path):
        //  input_shape = (1, 1, 64, 64)
        //  img = Image.open(image_path)
        //  img = img.resize((64, 64), Image.ANTIALIAS)
        //  img_data = np.array(img)
        //  img_data = np.resize(img_data, input_shape)
        //  return img_data

        //return base.PreProcess(img);

        Mat grayImg = new Mat();
        Imgproc.cvtColor(img, grayImg, Imgproc.COLOR_RGB2GRAY);

        // Create a 4D blob from a frame.
        Size inpSize = new Size(inpWidth > 0 ? inpWidth : img.cols(),
                           inpHeight > 0 ? inpHeight : img.rows());
        Mat blob = Dnn.blobFromImage(grayImg, scale, inpSize, mean, swapRB, false, CvType.CV_32F);

        //Debug.Log(ToStringHighDimsMat(blob));

        grayImg.Dispose();

        return blob;
    }

    protected override void PostProcess(Mat img, List<Mat> outputBlobs)
    {
        // Output
        // The model outputs a(1x8) array of scores corresponding to the 8 emotion classes, where the labels map as follows:
        // emotion_table = { 'neutral':0, 'happiness':1, 'surprise':2, 'sadness':3, 'anger':4, 'disgust':5, 'fear':6, 'contempt':7}

        // Postprocessing
        // Route the model output through a softmax function to map the aggregated activations across the network to probabilities across the 8 classes.

        //import numpy as np

        //def softmax(scores):
        //  # your softmax function

        //def postprocess(scores):
        //  ''' 
        //  This function takes the scores generated by the network and returns the class IDs in decreasing
        //  order of probability.
        //  '''
        //  prob = softmax(scores)
        //  prob = np.squeeze(prob)
        //  classes = np.argsort(prob)[::- 1]
        //  return classes


        //Debug.Log(outputBlobs[0].dump());

        Mat probabilities = outputBlobs[0];

        Mat sorted = new Mat(probabilities.rows(), probabilities.cols(), CvType.CV_32FC1);
        Core.sortIdx(probabilities, sorted, Core.SORT_EVERY_ROW | Core.SORT_DESCENDING);

        Mat top5 = new Mat(sorted, new OpenCVForUnity.CoreModule.Rect(0, 0, 5, 1));

        for (int i = 0; i < top5.cols(); i++)
        {
            int index = (int)top5.get(0, i)[0];
            Debug.Log((i + 1) + " : class=" + classNames[index] + " ; probability=" + probabilities.get(0, index)[0]);
            console.text += (i + 1) + " : class=" + classNames[index] + " ; probability=" + probabilities.get(0, index)[0] + "\n";
        }
    }
}