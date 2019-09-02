package com.example.opencvrecognition.activities

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.Window
import android.view.WindowManager
import android.widget.Toast
import com.example.opencvrecognition.R
import com.example.opencvrecognition.utilities.CNNOCR
import com.example.opencvrecognition.utilities.MSEROperations
import com.example.opencvrecognition.utilities.TesseractOCR
import kotlinx.android.synthetic.main.activity_mser_camera.*
import org.opencv.android.*
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.concurrent.thread


class MSERCameraActivity : BaseActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private val TAG = "MSERCameraActivity"
    private lateinit var mRgba: Mat
    private lateinit var mOpenCvCameraView: CameraBridgeViewBase
    private lateinit var mserOperations: MSEROperations
    private var finishedProcessing = true
    private lateinit var mTessOCR: TesseractOCR
    private val CNNOperations = CNNOCR()

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    mOpenCvCameraView.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(R.layout.activity_mser_camera)
        mserOperations = MSEROperations()
        mTessOCR = TesseractOCR()
        mTessOCR.initialize(this)

        mOpenCvCameraView = findViewById(R.id.mserVideoCamera)
        mOpenCvCameraView.setMaxFrameSize(1280, 720)
        mOpenCvCameraView.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView.setCvCameraViewListener(this)

        thread(start = true) {
            CNNOperations.initialize(this, contentResolver)
        }
    }

    override fun onPause() {
        super.onPause()
        mOpenCvCameraView.disableView()
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            //OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mOpenCvCameraView.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        mRgba = Mat(height, width, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        mRgba.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        mRgba = inputFrame.rgba()

        if (finishedProcessing) {
            val tempMat = Mat()
            mRgba.copyTo(tempMat)
            thread(start = true) {
                finishedProcessing = false
                mserOperations.initialize()
                mserOperations.unprocessedMat = tempMat
                mserOperations.detectTextRegionsVideo()
                if (noneRadioButton.isChecked) {
                    runOnUiThread {
                        Toast.makeText(this, "Detection completed", Toast.LENGTH_SHORT).show()
                    }
                } else if (tesseractRadioButton.isChecked) {
                    val text = doTesseract(mserOperations.textRegions)
                    runOnUiThread {
                        Toast.makeText(this, text, Toast.LENGTH_LONG).show()
                    }
                } else if (cnnRadioButton.isChecked) {
                    CNNOperations.textRegions = mserOperations.textRegions
                    CNNOperations.doRecognition()
                }
                finishedProcessing = true
            }
        }
        mRgba = mserOperations.showContours(mRgba)
        return mRgba
    }

    fun doTesseract(regions: ArrayList<Mat>): String {
        val milis = System.currentTimeMillis()
        var detectedText = ""
        for (region in regions.reversed()) {
            val bitmap =
                Bitmap.createBitmap(region.width(), region.height(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(region, bitmap)
            detectedText += mTessOCR.getOCRResult(bitmap) + "\n"
        }
        Log.i(TAG, "Recognition time:" + (System.currentTimeMillis() - milis))
        return detectedText
    }

}
