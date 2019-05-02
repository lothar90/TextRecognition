package com.example.opencvrecognition.utilities

import java.util.ArrayList

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

class ColorBlobDetector {
    // Lower and Upper bounds for range checking in HSV color space
    private val mLowerBound = Scalar(0.0)
    private val mUpperBound = Scalar(0.0)
    // Color radius for range checking in HSV color space
    private var mColorRadius = Scalar(25.0, 50.0, 50.0, 0.0)
    val spectrum = Mat()
    private val mContours = ArrayList<MatOfPoint>()

    // Cache
    internal var mPyrDownMat = Mat()
    internal var mHsvMat = Mat()
    internal var mMask = Mat()
    internal var mDilatedMask = Mat()
    internal var mHierarchy = Mat()

    val contours: List<MatOfPoint>
        get() = mContours

    fun setColorRadius(radius: Scalar) {
        mColorRadius = radius
    }

    fun setHsvColor(hsvColor: Scalar) {
        var minH = 0.0
        if (hsvColor.`val`[0] >= mColorRadius.`val`[0])
            minH = hsvColor.`val`[0] - mColorRadius.`val`[0]

        var maxH = 255.0
        if (hsvColor.`val`[0] + mColorRadius.`val`[0] <= 255)
            maxH = hsvColor.`val`[0] + mColorRadius.`val`[0]

        mLowerBound.`val`[0] = minH
        mUpperBound.`val`[0] = maxH

        mLowerBound.`val`[1] = hsvColor.`val`[1] - mColorRadius.`val`[1]
        mUpperBound.`val`[1] = hsvColor.`val`[1] + mColorRadius.`val`[1]

        mLowerBound.`val`[2] = hsvColor.`val`[2] - mColorRadius.`val`[2]
        mUpperBound.`val`[2] = hsvColor.`val`[2] + mColorRadius.`val`[2]

        mLowerBound.`val`[3] = 0.0
        mUpperBound.`val`[3] = 255.0

        val spectrumHsv = Mat(1, (maxH - minH).toInt(), CvType.CV_8UC3)

        var j = 0
        while (j < maxH - minH) {
            val tmp = byteArrayOf((minH + j).toByte(), 255.toByte(), 255.toByte())
            spectrumHsv.put(0, j, tmp)
            j++
        }

        Imgproc.cvtColor(spectrumHsv, spectrum, Imgproc.COLOR_HSV2RGB_FULL, 4)
    }

    fun setMinContourArea(area: Double) {
        mMinContourArea = area
    }

    fun process(rgbaImage: Mat) {
        Imgproc.pyrDown(rgbaImage, mPyrDownMat)
        Imgproc.pyrDown(mPyrDownMat, mPyrDownMat)

        Imgproc.cvtColor(mPyrDownMat, mHsvMat, Imgproc.COLOR_RGB2HSV_FULL)

        Core.inRange(mHsvMat, mLowerBound, mUpperBound, mMask)
        Imgproc.dilate(mMask, mDilatedMask, Mat())

        val contours = ArrayList<MatOfPoint>()

        Imgproc.findContours(mDilatedMask, contours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        // Find max contour area
        var maxArea = 0.0
        var each = contours.iterator()
        while (each.hasNext()) {
            val wrapper = each.next()
            val area = Imgproc.contourArea(wrapper)
            if (area > maxArea)
                maxArea = area
        }

        // Filter contours by area and resize to fit the original image size
        mContours.clear()
        each = contours.iterator()
        while (each.hasNext()) {
            val contour = each.next()
            if (Imgproc.contourArea(contour) > mMinContourArea * maxArea) {
                Core.multiply(contour, Scalar(4.0, 4.0), contour)
                mContours.add(contour)
            }
        }
    }

    companion object {
        // Minimum contour area in percent for contours filtering
        private var mMinContourArea = 0.1
    }
}