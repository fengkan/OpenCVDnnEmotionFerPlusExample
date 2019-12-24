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

public class OpenCVDnnModuleExampleBase : MonoBehaviour
{
    [Header("Dnn")]

    [TooltipAttribute("Path to input image.")]
    public string input;

    [TooltipAttribute("Path to a binary file of model contains trained weights. It could be a file with extensions .caffemodel (Caffe), .pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet).")]
    public string model;

    [TooltipAttribute("Path to a text file of model contains network configuration. It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet).")]
    public string config;

    [TooltipAttribute("Optional path to a text file with names of classes to label detected objects.")]
    public string classes;

    [TooltipAttribute("Optional list of classes to label detected objects.")]
    public List<string> classesList;

    [TooltipAttribute("Confidence threshold.")]
    public float confThreshold;

    [TooltipAttribute("Non-maximum suppression threshold.")]
    public float nmsThreshold;

    [TooltipAttribute("Preprocess input image by multiplying on a scale factor.")]
    public float scale;

    [TooltipAttribute("Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces.")]
    public Scalar mean;

    [TooltipAttribute("Indicate that model works with RGB input images instead BGR ones.")]
    public bool swapRB;

    [TooltipAttribute("Preprocess input image by resizing to a specific width.")]
    public int inpWidth;

    [TooltipAttribute("Preprocess input image by resizing to a specific height.")]
    public int inpHeight;

    protected Net net;

    protected List<string> classNames;
    protected List<string> outBlobNames;
    protected List<string> outBlobTypes;

    protected string classes_filepath;
    protected string input_filepath;
    protected string config_filepath;
    protected string model_filepath;

    protected Texture2D texture;

    [Header("UI")]
    public RawImage rawImage;
    public AspectRatioFitter aspectFitter;
    public InputField console;

    IEnumerator getFilePath_Coroutine;


    // Use this for initialization
    protected IEnumerator Start()
    {
        getFilePath_Coroutine = Utils.getMultipleFilePathsAsync(new string[] { classes, input, config, model },
            (paths) =>
            {
                classes_filepath = paths[0];
                input_filepath = paths[1];
                config_filepath = paths[2];
                model_filepath = paths[3];
            });
        yield return getFilePath_Coroutine;
        getFilePath_Coroutine = null;


        //if true, The error log of the Native side OpenCV will be displayed on the Unity Editor Console.
        Utils.setDebugMode(true);

        Mat img = LoadImage(input_filepath);

        SetupDnn();
        Process(img);

        Utils.setDebugMode(false);

        DisplayImage(img);

        img.Dispose();
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

    protected virtual Mat LoadImage(string imageFilepath)
    {
        Mat img = Imgcodecs.imread(imageFilepath);
        if (img.empty())
        {
            Debug.LogError(imageFilepath + " is not loaded.");
            img = new Mat(480, 640, CvType.CV_8UC3, new Scalar(0, 0, 0));
        }

        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);

        return img;
    }

    protected virtual void DisplayImage(Mat img)
    {
        if (texture == null) texture = new Texture2D(img.cols(), img.rows(), TextureFormat.RGBA32, false);
        Utils.matToTexture2D(img, texture);
        rawImage.texture = texture;
        // Scale the panel to match aspect ratios
        aspectFitter.aspectRatio = texture.width / (float)texture.height;
    }


    protected virtual void SetupDnn()
    {
        if (!string.IsNullOrEmpty(classes))
        {
            classNames = readClassNames(classes_filepath);
            if (classNames == null)
            {
                Debug.LogError(classes_filepath + " is not loaded.");
            }
        }
        else if (classesList.Count > 0)
        {
            classNames = classesList;
        }

        if (!string.IsNullOrEmpty(config_filepath) && !string.IsNullOrEmpty(model_filepath))
        {
            net = Dnn.readNet(model_filepath, config_filepath);
        }
        else if (!string.IsNullOrEmpty(model_filepath))
        {
            net = Dnn.readNet(model_filepath);
        }
        else
        {
            Debug.LogError(config_filepath + " or " + model_filepath + " is not loaded.");
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
        Debug.Log("Inference time, ms: " + tm.getTimeMilli());
        console.text += "Inference time, ms: " + tm.getTimeMilli() + "\n";


        PostProcess(img, outputBlobs);

        for (int i = 0; i < outputBlobs.Count; i++)
        {
            outputBlobs[i].Dispose();
        }
        inputBlob.Dispose();
    }

    protected virtual Mat PreProcess(Mat img)
    {
        // Create a 4D blob from a frame.
        Size inpSize = new Size(inpWidth > 0 ? inpWidth : img.cols(),
                           inpHeight > 0 ? inpHeight : img.rows());
        Mat blob = Dnn.blobFromImage(img, scale, inpSize, mean, swapRB, false);

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

    }

    protected virtual List<string> readClassNames(string filename)
    {
        List<string> classNames = new List<string>();

        System.IO.StreamReader cReader = null;
        try
        {
            cReader = new System.IO.StreamReader(filename, System.Text.Encoding.Default);

            while (cReader.Peek() >= 0)
            {
                string name = cReader.ReadLine();
                classNames.Add(name);
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError(ex.Message);
            return null;
        }
        finally
        {
            if (cReader != null)
                cReader.Close();
        }

        return classNames;
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

    protected string ToStringHighDimsMat(Mat mat)
    {
        string size = "";
        for (int i = 0; i < mat.dims(); ++i)
        {
            size += mat.size(i) + "*";
        }

        return "Mat [ " + size + CvType.typeToString(mat.type()) + ", isCont=" + mat.isContinuous() + ", isSubmat=" + mat.isSubmatrix()
             + ", nativeObj=" + mat.nativeObj + ", dataAddr=" + mat.dataAddr() + " ]";
    }
}