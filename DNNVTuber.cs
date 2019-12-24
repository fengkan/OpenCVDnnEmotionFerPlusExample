using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DNNVTuber : MonoBehaviour
{
    [Header("Dnn")]

    [TooltipAttribute("Path to a binary file of model contains trained weights. It could be a file with extensions .caffemodel (Caffe), .pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet).")]
    public string model = "/emotion_ferplus/model.onnx";

    [TooltipAttribute("Optional list of classes to label detected objects.")]
    List<string> classesList = new List<string>(new string[] { "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt" });

    [TooltipAttribute("Confidence threshold.")]
    public float confThreshold = 0;

    [TooltipAttribute("Non-maximum suppression threshold.")]
    public float nmsThreshold = 0;

    [TooltipAttribute("Preprocess input image by multiplying on a scale factor.")]
    public float scale = 1;

    [TooltipAttribute("Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces.")]
    public Scalar mean = new Scalar(0);

    [TooltipAttribute("Indicate that model works with RGB input images instead BGR ones.")]
    public bool swapRB;

    [TooltipAttribute("Preprocess input image by resizing to a specific width.")]
    public int inpWidth = 64;

    [TooltipAttribute("Preprocess input image by resizing to a specific height.")]
    public int inpHeight = 64;

    protected Net net;

    protected List<string> classNames;
    protected List<string> outBlobNames;
    protected List<string> outBlobTypes;

    protected string model_filepath;

    protected Texture2D texture;

    [Header("UI")]
    public Text text;

    int currentClass;

    IEnumerator getFilePath_Coroutine;


    // Use this for initialization
    protected IEnumerator Start()
    {
        getFilePath_Coroutine = Utils.getMultipleFilePathsAsync(new string[] { model },
            (paths) =>
            {
                model_filepath = paths[0];
            });
        yield return getFilePath_Coroutine;
        getFilePath_Coroutine = null;

        //if true, The error log of the Native side OpenCV will be displayed on the Unity Editor Console.
        Utils.setDebugMode(true);

        SetupDnn();
    }

    public void ProcessImage(Mat img)
    {
        if (net != null)
            Process(img);
    }


    void OnDisable()
    {
        if (getFilePath_Coroutine != null)
        {
            StopCoroutine(getFilePath_Coroutine);
            ((IDisposable)getFilePath_Coroutine).Dispose();
        }

        if (net != null)
            net.Dispose();

        if (texture != null)
        {
            Texture.Destroy(texture);
            texture = null;
        }
    }
    protected virtual void SetupDnn()
    {
        if (classesList.Count > 0)
        {
            classNames = classesList;
        }

        if (!string.IsNullOrEmpty(model_filepath))
        {
            net = Dnn.readNet(model_filepath);
        }


        outBlobNames = getOutputsNames(net);
        //for (int i = 0; i < outBlobNames.Count; i++)
        //{
        //    Debug.Log("names [" + i + "] " + outBlobNames[i]);
        //}

        outBlobTypes = getOutputsTypes(net);
        //for (int i = 0; i < outBlobTypes.Count; i++)
        //{
        //    Debug.Log("types [" + i + "] " + outBlobTypes[i]);
        //}

    }

    protected virtual void Process(Mat img)
    {
        Mat inputBlob = PreProcess(img);


        TickMeter tm = new TickMeter();
        tm.start();


        List<Mat> outputBlobs = Predict(inputBlob);

        tm.stop();


        PostProcess(img, outputBlobs);

        for (int i = 0; i < outputBlobs.Count; i++)
        {
            outputBlobs[i].Dispose();
        }
        inputBlob.Dispose();
    }

    protected virtual Mat PreProcess(Mat img)
    {
        Mat grayImg = new Mat();
        Imgproc.cvtColor(img, grayImg, Imgproc.COLOR_RGB2GRAY);

        // Create a 4D blob from a frame.
        Size inpSize = new Size(inpWidth > 0 ? inpWidth : img.cols(),
                           inpHeight > 0 ? inpHeight : img.rows());
        Mat blob = Dnn.blobFromImage(grayImg, scale, inpSize, mean, swapRB, false, CvType.CV_32F);

        grayImg.Dispose();

        return blob;
    }

    protected virtual List<Mat> Predict(Mat inputBlob)
    {
        net.setInput(inputBlob);

        List<Mat> outs = new List<Mat>();
        net.forward(outs, outBlobNames);

        return outs;
    }

    protected virtual void PostProcess(Mat img, List<Mat> outputBlobs)
    {
        Mat probabilities = outputBlobs[0];

        Mat sorted = new Mat(probabilities.rows(), probabilities.cols(), CvType.CV_32FC1);
        Core.sortIdx(probabilities, sorted, Core.SORT_EVERY_ROW | Core.SORT_DESCENDING);

        Mat top5 = new Mat(sorted, new OpenCVForUnity.CoreModule.Rect(0, 0, 5, 1));

        currentClass = (int)top5.get(0, 0)[0];
        text.text = classNames[currentClass];
    }

    private List<string> getOutputsNames(Net net)
    {
        List<string> names = new List<string>();


        MatOfInt outLayers = net.getUnconnectedOutLayers();
        for (int i = 0; i < outLayers.total(); ++i)
        {
            names.Add(net.getLayer(new DictValue((int)outLayers.get(i, 0)[0])).get_name());
        }
        outLayers.Dispose();

        return names;
    }

    private List<string> getOutputsTypes(Net net)
    {
        List<string> types = new List<string>();


        MatOfInt outLayers = net.getUnconnectedOutLayers();
        for (int i = 0; i < outLayers.total(); ++i)
        {
            types.Add(net.getLayer(new DictValue((int)outLayers.get(i, 0)[0])).get_type());
        }
        outLayers.Dispose();

        return types;
    }

    public int getCurrentClass()
    {
        return currentClass;
    }
}