package com.example.opencvrecognition.activities

import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.Window
import android.view.WindowManager
import com.example.opencvrecognition.R
import com.example.opencvrecognition.utilities.MSEROperations
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat


class MSERCameraActivity : BaseActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private val TAG = "MSERCameraActivity"
    private lateinit var mRgba: Mat
    private lateinit var mOpenCvCameraView: CameraBridgeViewBase
    private lateinit var mserOperations: MSEROperations
    private var finishedProcessing = true

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

        mOpenCvCameraView = findViewById(R.id.mserVideoCamera)
        mOpenCvCameraView.setMaxFrameSize(640, 480)
        mOpenCvCameraView.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView.setCvCameraViewListener(this)
    }

    override fun onPause() {
        super.onPause()
        mOpenCvCameraView.disableView()
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback)
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

//        var bitmap = originalBitmap.copy(originalBitmap.config, true)
//        if (bitmap.height > 2000 || bitmap.width > 1000)
//            bitmap = Bitmap.createScaledBitmap(
//                bitmap,
//                (bitmap.width * 0.25).toInt(),
//                (bitmap.height * 0.25).toInt(),
//                false
//            )
        if (finishedProcessing) {
            finishedProcessing = false
            mserOperations.initialize()
            mserOperations.unprocessedMat = mRgba
            mserOperations.detectTextRegionsVideo()
            mRgba = mserOperations.mRgba
            finishedProcessing = true
        }

        return mRgba
    }

}
