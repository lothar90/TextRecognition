package com.example.opencvrecognition.utilities

import android.app.Activity
import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel.MapMode.READ_ONLY
import kotlin.math.max


class CNNOCR {
    private val TAG = "CNNOCR"
    lateinit var context: Context
    private lateinit var net: Net
    lateinit var mRgba: Mat
    private lateinit var contentResolver: ContentResolver
    private lateinit var tfliteInterpreter: Interpreter
    var textRegions = ArrayList<Mat>()

    val LABELS = arrayListOf(
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'I',
        'J',
        'K',
        'L',
        'M',
        'N',
        'O',
        'P',
        'Q',
        'R',
        'S',
        'T',
        'U',
        'V',
        'W',
        'X',
        'Y',
        'Z',
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'g',
        'h',
        'i',
        'j',
        'k',
        'l',
        'm',
        'n',
        'o',
        'p',
        'q',
        'r',
        's',
        't',
        'u',
        'v',
        'w',
        'x',
        'y',
        'z'
    )

    fun initialize(context: Context, contentResolver: ContentResolver) {
        this.context = context
        this.contentResolver = contentResolver
        val inputStream = context.assets.open("tensorflow_cnn.pb")
        net = Dnn.readNetFromTensorflow(MatOfByte(*inputStream.readBytes()))
        tfliteInterpreter = Interpreter(loadModelFile(context))
    }

    @Throws(IOException::class)
    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd("tflite_model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(READ_ONLY, startOffset, declaredLength)
    }

    fun doRecognition() {
        val milis = System.currentTimeMillis()
        var wholeText = ""
        for (region in textRegions) {
            val h = region.height()
            val scale = h / 128.0
            Imgproc.resize(region, region, Size(region.width() / scale, 128.0))
            val lines = getTextLines(region)
            var textLines = ""
            for (line in lines) {
                val words = getWords(line)
                var wordText = ""
                for (word in words) {
                    val characters = getCharacters(word)
                    var chars = ""
                    for (char in characters) {
                        val gray = Mat()
                        Imgproc.resize(char, gray, Size(32.0, 32.0))
//                        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGBA2GRAY, 1)
                        Core.normalize(gray, gray, 0.0, 255.0, Core.NORM_MINMAX)
                        var bitmap =
                            Bitmap.createBitmap(gray.cols(), gray.rows(), Bitmap.Config.ARGB_8888)
                        Utils.matToBitmap(gray, bitmap)
                        var input = convertBitmap(bitmap)
                        val result = Array(1) { FloatArray(LABELS.size) }
                        tfliteInterpreter.run(input, result)
                        chars += LABELS[result[0].indices.maxBy { result[0][it] }!!]
                    }
                    wordText += chars + " "
                }
                textLines += wordText
            }
            wholeText += "\n" + textLines
        }
        (context as Activity).runOnUiThread {
            Toast.makeText(context, wholeText, Toast.LENGTH_LONG).show()
        }
        Log.i(TAG, "Recognition time:" + (System.currentTimeMillis() - milis))
//        val gray = Mat()
//        Imgproc.resize(mRgba, gray, Size(32.0, 32.0))
//        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGBA2GRAY, 1)
//        Core.normalize(gray, gray, 0.0, 255.0, Core.NORM_MINMAX)
//        //gray.assignTo(gray, CvType.CV_32FC1)
//        var bitmap = Bitmap.createBitmap(gray.cols(), gray.rows(), Bitmap.Config.ARGB_8888)
//        Utils.matToBitmap(gray, bitmap)
//        var input = convertBitmap(bitmap)
//        val result = Array(1) { FloatArray(LABELS.size) }
//        tfliteInterpreter.run(input, result)
//        val blob = Dnn.blobFromImage(gray, 1.0, Size(32.0, 32.0))
//        net.setInput(blob)
//        val out = net.forward()
//        val result = Core.minMaxLoc(out)
//        val probabilities = ArrayList<Double>()
//        for (i in 0 until out.rows()) {
//            for (j in 0 until out.cols()) {
//                probabilities.add(out.get(i,j)[0])
//            }
//        }
//        Toast.makeText(context, "Results: ${LABELS[result.maxLoc.x.toInt()]}", Toast.LENGTH_LONG).show()
//        Toast.makeText(
//            context,
//            "Results: ${LABELS[result[0].indices.maxBy { result[0][it] }!!]}",
//            Toast.LENGTH_LONG
//        ).show()
    }

    fun getTextLines(mat: Mat): ArrayList<Mat> {
        var milis = System.currentTimeMillis()
        val th = 2
        var image = Mat()
        Imgproc.cvtColor(mat, image, Imgproc.COLOR_RGB2GRAY, 1)
        val hist = Mat()
        val ranges = MatOfFloat(0f, 256f)
        val histSize = MatOfInt(256)
        Imgproc.calcHist(listOf(image), MatOfInt(0), Mat(), hist, histSize, ranges)
        val maxValue = Core.minMaxLoc(hist)
        if (maxValue.maxLoc.y < 180) {
            Core.bitwise_not(image, image)
        }
        Imgproc.GaussianBlur(image, image, Size(5.0, 5.0), 0.0)
        Imgproc.threshold(image, image, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        val invertedBinary = Mat()
        val horizontal = Mat()
        Core.bitwise_not(image, invertedBinary)
        Core.reduce(invertedBinary, horizontal, 1, Core.REDUCE_AVG)
        val height = image.height()
        val uppers = ArrayList<Int>()
        val lowers = ArrayList<Int>()
        for (y in 0 until height - 1) {
            if (horizontal.get(y, 0)[0] <= th && horizontal.get(y + 1, 0)[0] > th)
                uppers.add(y)
            if (horizontal.get(y, 0)[0] > th && horizontal.get(y + 1, 0)[0] <= th)
                lowers.add(y)
        }
        if (lowers.size > uppers.size)
            uppers.add(0, 0)
        if (lowers.size < uppers.size)
            lowers.add(height)
        val texts = ArrayList<Mat>()
        for (i in 0 until uppers.size) {
            val text = image.submat(uppers[i], lowers[i], 0, mat.width())
            Core.copyMakeBorder(text, text, 10, 10, 0, 0, Core.BORDER_CONSTANT, Scalar(255.0))
            texts.add(text)
        }
        Log.i(TAG, "Text lines time:" + (System.currentTimeMillis() - milis))
        return texts
    }

    fun selector(rect: MatOfPoint): Double = Imgproc.boundingRect(rect).tl().x

    fun getWords(mat: Mat): ArrayList<Mat> {
        var milis = System.currentTimeMillis()
        val image = Mat()
        val contours = ArrayList<MatOfPoint>()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(10.0, 10.0))
        val erodeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE, Size(3.0, 3.0))
        Imgproc.dilate(mat, image, erodeKernel)
        Imgproc.erode(image, image, kernel)
        Core.bitwise_not(image, image)
        Imgproc.findContours(
            image,
            contours,
            Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        contours.sortBy { selector(it) }
        val words = ArrayList<Mat>()
        for (i in 0 until contours.size) {
            val rect = Imgproc.boundingRect(contours[i])
            val word = Mat(mat, rect)
            words.add(word)
        }
        Log.i(TAG, "Words time:" + (System.currentTimeMillis() - milis))
        return words
    }

    fun getCharacters(mat: Mat): ArrayList<Mat> {
        var milis = System.currentTimeMillis()
        val image = Mat()
        val contours = ArrayList<MatOfPoint>()
        val erodeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE, Size(1.0, 1.0))
        Imgproc.dilate(mat, image, erodeKernel)
//        saveMat(image, "title")
        Core.bitwise_not(image, image)
//        saveMat(image, "title")
        Imgproc.findContours(
            image,
            contours,
            Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        contours.sortBy { selector(it) }
        val characters = ArrayList<Mat>()
        for (i in 0 until contours.size) {
            val rect = Imgproc.boundingRect(contours[i])
            val height = image.height()
            if (rect.width.toDouble() / rect.height.toDouble() < 4 && height.toDouble() / 2 < rect.y + rect.height && height > rect.y + rect.height) {
                val character = Mat(mat, rect)
//                saveMat(character, "title")
                val newSize = max(character.width(), character.height())
                val temp = Mat(newSize, newSize, CvType.CV_8U, Scalar.all(255.0))
                val ax = (newSize - character.width()) / 2
                val ay = (newSize - character.height()) / 2
                character.copyTo(
                    temp.rowRange(ay, character.height() + ay).colRange(
                        ax,
                        character.width() + ax
                    )
                )
                Core.copyMakeBorder(
                    temp,
                    character,
                    5,
                    5,
                    5,
                    5,
                    Core.BORDER_CONSTANT,
                    Scalar(255.0)
                )
                characters.add(character)
//                saveMat(character, "title")
            }
        }
        Log.i(TAG, "Characters time:" + (System.currentTimeMillis() - milis))
        return characters
    }

    fun convertBitmap(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(bitmap.width * bitmap.height * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixel in pixels) {
            val rChannel = (pixel shr 16 and 0xFF).toFloat()
            val gChannel = (pixel shr 8 and 0xFF).toFloat()
            val bChannel = (pixel and 0xFF).toFloat()
            val pixelValue = (rChannel + gChannel + bChannel) / 3f / 255f
            byteBuffer.putFloat(pixelValue)
        }
        return byteBuffer
    }

    private fun saveMat(mat: Mat, title: String) {
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bitmap)
        CapturePhotoUtils.insertImage(contentResolver, bitmap, title, title)
    }
}