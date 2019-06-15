package com.example.opencvrecognition.utilities

import android.content.Context
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import org.opencv.utils.Converters

class EASTOperations(var context: Context) {
    lateinit var mRgba: Mat
    private lateinit var net: Net

    fun initialize() {
        val inputStream = context.assets.open("frozen_east_text_detection.pb")
        net = Dnn.readNetFromTensorflow(MatOfByte(*inputStream.readBytes()))
    }

    fun doEAST() {

        val scoreThresh = 0.5f
        val nmsThresh = 0.4f

        val size = Size(320.0, 320.0)
        var W = (size.width / 4).toInt()
        val H = (size.height / 4).toInt()
        val blob = Dnn.blobFromImage(mRgba, 1.0, size, Scalar(123.68, 116.78, 103.94), true, false)
        net.setInput(blob)
        val outs = ArrayList<Mat>(2)
        val outNames = ArrayList<String>()
        outNames.add("feature_fusion/Conv_7/Sigmoid")
        outNames.add("feature_fusion/concat_3")
        net.forward(outs, outNames)

        val scores = outs[0].reshape(1, H)
        val geometry = outs[1].reshape(1, 5 * H)
        val confidencesList = ArrayList<Float>()
        val boxesList = decode(scores, geometry, confidencesList, scoreThresh)

        if (confidencesList.size > 0) {
            val confidences = MatOfFloat(Converters.vector_float_to_Mat(confidencesList))
            val boxesArray = boxesList.toTypedArray()
            val boxes = MatOfRotatedRect(*boxesArray)
            val indices = MatOfInt()
            Dnn.NMSBoxesRotated(boxes, confidences, scoreThresh, nmsThresh, indices)

            val ratio = Point(mRgba.cols().toFloat() / size.width, mRgba.rows().toFloat() / size.height)
            val indexes = indices.toArray()

            for (i in 0 until indexes.size) {
                val rot = boxesArray[indexes[i]]
                val vertices = arrayOfNulls<Point>(4)
                rot.points(vertices)
                for (j in 0 until 4) {
                    vertices[j]!!.x *= ratio.x
                    vertices[j]!!.y *= ratio.y
                }
                for (j in 0 until 4)
                    Imgproc.line(mRgba, vertices[j], vertices[(j + 1) % 4], Scalar(255.0, 0.0, 255.0), 2)
            }
        }
    }

    private fun decode(
        scores: Mat,
        geometry: Mat,
        confidences: ArrayList<Float>,
        scoreThresh: Float
    ): ArrayList<RotatedRect> {
        val W = geometry.cols()
        val H = geometry.rows() / 5

        val detections = ArrayList<RotatedRect>()
        for (y in 0 until H) {
            val scoresData = scores.row(y)
            val x0Data = geometry.submat(0, H, 0, W).row(y)
            val x1Data = geometry.submat(H, 2 * H, 0, W).row(y)
            val x2Data = geometry.submat(2 * H, 3 * H, 0, W).row(y)
            val x3Data = geometry.submat(3 * H, 4 * H, 0, W).row(y)
            val anglesData = geometry.submat(4 * H, 5 * H, 0, W).row(y)

            for (x in 0 until W) {
                val score = scoresData.get(0, x)[0]
                if (score >= scoreThresh) {
                    val offsetX = x * 4.0
                    val offsetY = y * 4.0
                    val angle = anglesData.get(0, x)[0]
                    val cosA = Math.cos(angle)
                    val sinA = Math.sin(angle)
                    val x0 = x0Data.get(0, x)[0]
                    val x1 = x1Data.get(0, x)[0]
                    val x2 = x2Data.get(0, x)[0]
                    val x3 = x3Data.get(0, x)[0]
                    val h = x0 + x2
                    val w = x1 + x3

                    val offset = Point(offsetX + cosA * x1 + sinA * x2, offsetY - sinA * x1 + cosA * x2)
                    val p1 = Point(-1 * sinA * h + offset.x, -1 * cosA * h + offset.y)
                    val p3 = Point(-1 * cosA * w + offset.x, sinA * w + offset.y)
                    val r = RotatedRect(
                        Point(0.5 * (p1.x + p3.x), 0.5 * (p1.y + p3.y)),
                        Size(w, h),
                        -1 * angle * 180 / Math.PI
                    )
                    detections.add(r)
                    confidences.add(score.toFloat())
                }
            }
        }
        return detections
    }
}