package com.example.opencvrecognition.utilities

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.features2d.MSER
import org.opencv.imgproc.Imgproc


class MSEROperations {
    val CONTOUR_COLOR = Scalar(255.0, 0.0, 0.0, 0.0)
    lateinit var unprocessedMat: Mat
    lateinit var mRgba: Mat
    private lateinit var mGray: Mat
    lateinit var bitmap: Bitmap
    private var aspectRatio = 0.0
    private var extent = 0.0
    var textRegions = ArrayList<Mat>()
    private var contours = ArrayList<MatOfPoint>()

    private val TAG = "MSEROperations"

    fun initialize() {
        mRgba = Mat()
        mGray = Mat()
    }

    fun detectMSERRegions() {
        val listOfRegions = ArrayList<MatOfPoint>()
        val boundingBoxes = MatOfRect()

        Utils.bitmapToMat(bitmap, mRgba)

        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGB2GRAY)

        val detector = MSER.create(
            5, 60, 14400,
            0.2, 0.15, 200,
            1.01, 0.003, 5
        )

        var milis = System.currentTimeMillis()
        detector.detectRegions(mGray, listOfRegions, boundingBoxes)
        Log.i(TAG, "Detection time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        val rectangles = boundingBoxes.toList()
        for (i in 0 until rectangles.size) {
            val width = rectangles[i].width
            val height = rectangles[i].height
            aspectRatio = width.toDouble() / height.toDouble()
            extent = listOfRegions[i].toArray().size.toDouble() / (width * height)
            if (aspectRatio > 0.3 && aspectRatio < 2 && extent > 0.3 && extent < 0.9)
                Imgproc.rectangle(mRgba, rectangles[i].br(), rectangles[i].tl(), CONTOUR_COLOR, 2)
        }
        Log.i(TAG, "Masking time:" + (System.currentTimeMillis() - milis))

        bitmap = bitmap.copy(bitmap.config, true)
        Utils.matToBitmap(mRgba, bitmap)
    }

    fun createMSERMask() {
        val listOfRegions = ArrayList<MatOfPoint>()
        val boundingBoxes = MatOfRect()

        Utils.bitmapToMat(bitmap, mRgba)
        val mask = Mat.zeros(mRgba.size(), CvType.CV_8UC1)

        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGB2GRAY)

        val detector = MSER.create(
            5, 60, 14400,
            0.2, 0.15, 200,
            1.01, 0.003, 5
        )

        var milis = System.currentTimeMillis()
        detector.detectRegions(mGray, listOfRegions, boundingBoxes)
        Log.i(TAG, "Detection time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        val rectangles = boundingBoxes.toList()
        for (i in 0 until rectangles.size) {
            val width = rectangles[i].width
            val height = rectangles[i].height
            aspectRatio = width.toDouble() / height.toDouble()
            extent = listOfRegions[i].toArray().size.toDouble() / (width * height)
            if (aspectRatio > 0.3 && aspectRatio < 2 && extent > 0.3 && extent < 0.9) {
                val roi = Mat(mask, rectangles[i])
                roi.setTo(CONTOUR_COLOR)
            }
        }
        Log.i(TAG, "Masking time:" + (System.currentTimeMillis() - milis))

        bitmap = bitmap.copy(bitmap.config, true)
        Utils.matToBitmap(mask, bitmap)
    }

    fun maskAfterMorphology() {
        val listOfRegions = ArrayList<MatOfPoint>()
        val boundingBoxes = MatOfRect()

        Utils.bitmapToMat(bitmap, mRgba)
        val mask = Mat.zeros(mRgba.size(), CvType.CV_8UC1)

        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGB2GRAY)

        val detector = MSER.create(
            5, 60, 14400,
            0.2, 0.15, 200,
            1.01, 0.003, 5
        )

        var milis = System.currentTimeMillis()
        detector.detectRegions(mGray, listOfRegions, boundingBoxes)
        Log.i(TAG, "Detection time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        val rectangles = boundingBoxes.toList()
        for (i in 0 until rectangles.size) {
            val width = rectangles[i].width
            val height = rectangles[i].height
            aspectRatio = width.toDouble() / height.toDouble()
            extent = listOfRegions[i].toArray().size.toDouble() / (width * height)
            if (aspectRatio > 0.3 && aspectRatio < 2 && extent > 0.3 && extent < 0.9) {
                val roi = Mat(mask, rectangles[i])
                roi.setTo(CONTOUR_COLOR)
            }
        }
        Log.i(TAG, "Masking time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        Imgproc.morphologyEx(
            mask,
            mask,
            Imgproc.MORPH_DILATE,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 1.0))
        )
        Log.i(TAG, "Morpho time:" + (System.currentTimeMillis() - milis))

        bitmap = bitmap.copy(bitmap.config, true)
        Utils.matToBitmap(mask, bitmap)
    }

    fun drawAllFoundContours() {
        val listOfRegions = ArrayList<MatOfPoint>()
        val boundingBoxes = MatOfRect()
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var boundingBox: Rect

        Utils.bitmapToMat(bitmap, mRgba)
        val mask = Mat.zeros(mRgba.size(), CvType.CV_8UC1)

        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGB2GRAY)

        val detector = MSER.create(
            5, 60, 14400,
            0.2, 0.15, 200,
            1.01, 0.003, 5
        )

        var milis = System.currentTimeMillis()
        detector.detectRegions(mGray, listOfRegions, boundingBoxes)
        Log.i(TAG, "Detection time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        val rectangles = boundingBoxes.toList()
        for (i in 0 until rectangles.size) {
            val width = rectangles[i].width
            val height = rectangles[i].height
            aspectRatio = width.toDouble() / height.toDouble()
            extent = listOfRegions[i].toArray().size.toDouble() / (width * height)
            if (aspectRatio > 0.3 && aspectRatio < 2 && extent > 0.3 && extent < 0.9) {
                val roi = Mat(mask, rectangles[i])
                roi.setTo(CONTOUR_COLOR)
            }
        }
        Log.i(TAG, "Masking time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        Imgproc.morphologyEx(
            mask,
            mask,
            Imgproc.MORPH_DILATE,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 1.0))
        )
        Log.i(TAG, "Morpho time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        Log.i(TAG, "Finding contours time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        for (i in 0 until contours.size) {
            boundingBox = Imgproc.boundingRect(contours[i])
            Imgproc.rectangle(mRgba, boundingBox.br(), boundingBox.tl(), CONTOUR_COLOR, 2)
        }
        Log.i(TAG, "Drawing time:" + (System.currentTimeMillis() - milis))

        bitmap = bitmap.copy(bitmap.config, true)
        Utils.matToBitmap(mRgba, bitmap)
    }

    fun detectTextRegions() {
        val listOfRegions = ArrayList<MatOfPoint>()
        val boundingBoxes = MatOfRect()
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var boundingBox: Rect
        val zeros = Scalar(0.0, 0.0, 0.0)
        textRegions.clear()

        Utils.bitmapToMat(bitmap, mRgba)
        val mask = Mat.zeros(mRgba.size(), CvType.CV_8UC1)

        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGB2GRAY)

        val detector = MSER.create(
            5, 60, 14400,
            0.2, 0.15, 200,
            1.01, 0.003, 5
        )

        var milis = System.currentTimeMillis()
        detector.detectRegions(mGray, listOfRegions, boundingBoxes)
        Log.i(TAG, "Detection time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        val rectangles = boundingBoxes.toList()
        for (i in 0 until rectangles.size) {
            val width = rectangles[i].width
            val height = rectangles[i].height
            aspectRatio = width.toDouble() / height.toDouble()
            extent = listOfRegions[i].toArray().size.toDouble() / (width * height)
            if (aspectRatio > 0.3 && aspectRatio < 2 && extent > 0.3 && extent < 0.9) {
                val roi = Mat(mask, rectangles[i])
                roi.setTo(CONTOUR_COLOR)
            }
        }
        Log.i(TAG, "Masking time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        Imgproc.morphologyEx(
            mask,
            mask,
            Imgproc.MORPH_DILATE,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 1.0))
        )
        Log.i(TAG, "Morpho time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        Log.i(TAG, "Finding contours time:" + (System.currentTimeMillis() - milis))

        val copy = mRgba.clone()
        milis = System.currentTimeMillis()
        for (i in 0 until contours.size) {
            boundingBox = Imgproc.boundingRect(contours[i])
            if (boundingBox.area() > 100 && boundingBox.width / boundingBox.height > 1.5) {
                val roi = Mat(copy, boundingBox)
                textRegions.add(roi)
                Imgproc.rectangle(mRgba, boundingBox.br(), boundingBox.tl(), CONTOUR_COLOR, 1)
                continue
            }
            if (boundingBox.area() > 100 && boundingBox.height / boundingBox.width > 1.5) {
                val roi = Mat(copy, boundingBox)
                textRegions.add(roi)
                Imgproc.rectangle(mRgba, boundingBox.br(), boundingBox.tl(), CONTOUR_COLOR, 1)
                continue
            }
            val roi = Mat(mask, boundingBox)
            roi.setTo(zeros)
        }
        Log.i(TAG, "Drawing time:" + (System.currentTimeMillis() - milis))

        bitmap = bitmap.copy(bitmap.config, true)
        Utils.matToBitmap(mRgba, bitmap)
    }

    fun detectTextRegionsVideo() {
        val listOfRegions = ArrayList<MatOfPoint>()
        val boundingBoxes = MatOfRect()
        //val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        var boundingBox: Rect
        val zeros = Scalar(0.0, 0.0, 0.0)

        textRegions.clear()
        mRgba = unprocessedMat

        val mask = Mat.zeros(mRgba.size(), CvType.CV_8UC1)

        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGB2GRAY)

        val detector = MSER.create(
            5, 60, 14400,
            0.2, 0.15, 200,
            1.01, 0.003, 5
        )

        var milis = System.currentTimeMillis()
        detector.detectRegions(mGray, listOfRegions, boundingBoxes)
        Log.i(TAG, "Detection time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        val rectangles = boundingBoxes.toList()
        for (i in 0 until rectangles.size) {
            val width = rectangles[i].width
            val height = rectangles[i].height
            aspectRatio = width.toDouble() / height.toDouble()
            extent = listOfRegions[i].toArray().size.toDouble() / (width * height)
            if (aspectRatio > 0.3 && aspectRatio < 2 && extent > 0.3 && extent < 0.9) {
                val roi = Mat(mask, rectangles[i])
                roi.setTo(CONTOUR_COLOR)
            }
        }
        Log.i(TAG, "Masking time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        Imgproc.morphologyEx(
            mask,
            mask,
            Imgproc.MORPH_DILATE,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 1.0))
        )
        Log.i(TAG, "Morpho time:" + (System.currentTimeMillis() - milis))

        var newContours = ArrayList<MatOfPoint>()
        milis = System.currentTimeMillis()
        Imgproc.findContours(
            mask,
            newContours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        Log.i(TAG, "Finding contours time:" + (System.currentTimeMillis() - milis))
        contours = newContours

        val copy = mRgba.clone()
        milis = System.currentTimeMillis()
        for (i in 0 until contours.size) {
            boundingBox = Imgproc.boundingRect(contours[i])
            if (boundingBox.area() > 100 && boundingBox.width / boundingBox.height > 1.5) {
                val roi = Mat(copy, boundingBox)
                textRegions.add(roi)
                //Imgproc.rectangle(mRgba, boundingBox.br(), boundingBox.tl(), CONTOUR_COLOR, 1)
                continue
            }
            if (boundingBox.area() > 100 && boundingBox.height / boundingBox.width > 1.5) {
                val roi = Mat(copy, boundingBox)
                textRegions.add(roi)
                //Imgproc.rectangle(mRgba, boundingBox.br(), boundingBox.tl(), CONTOUR_COLOR, 1)
                continue
            }
            val roi = Mat(mask, boundingBox)
            roi.setTo(zeros)
        }
        Log.i(TAG, "Drawing time:" + (System.currentTimeMillis() - milis))
    }

    fun showContours(inputMat: Mat): Mat {
        var actualContours = contours
        var boundingBox: Rect
        val milis = System.currentTimeMillis()
        for (i in 0 until actualContours.size) {
            boundingBox = Imgproc.boundingRect(actualContours[i])
            if (boundingBox.area() > 100 && boundingBox.width / boundingBox.height > 1.5) {
                Imgproc.rectangle(inputMat, boundingBox.br(), boundingBox.tl(), CONTOUR_COLOR, 1)
                continue
            }
            if (boundingBox.area() > 100 && boundingBox.height / boundingBox.width > 1.5) {
                Imgproc.rectangle(inputMat, boundingBox.br(), boundingBox.tl(), CONTOUR_COLOR, 1)
                continue
            }
            //val roi = Mat(mask, boundingBox)
            //roi.setTo(zeros)
        }
        Log.i(TAG, "Drawing time:" + (System.currentTimeMillis() - milis))
        return inputMat
    }
}