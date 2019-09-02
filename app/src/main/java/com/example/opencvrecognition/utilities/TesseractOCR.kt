package com.example.opencvrecognition.utilities

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import com.googlecode.tesseract.android.TessBaseAPI
import com.googlecode.tesseract.android.TessBaseAPI.PageSegMode.PSM_AUTO
import java.io.File
import java.io.FileOutputStream


class TesseractOCR {
    private val TAG = "TesseractOCR"
    private val TESS_DATA = "/tessdata"
    private lateinit var datapath: String
    private val mTess = TessBaseAPI()
    lateinit var context: Context

    fun initialize(context: Context) {
        this.context = context
        prepareTessData()
        datapath = context.getExternalFilesDir("/")?.path + "/"
        mTess.setDebug(true)
        mTess.init(datapath, "eng+pol")
        mTess.pageSegMode = PSM_AUTO
    }

    private fun prepareTessData() {
        try {
            val dir = context.getExternalFilesDir(TESS_DATA)
            if (!dir!!.exists()) {
                if (!dir.mkdir()) {
                    Toast.makeText(
                        context.applicationContext,
                        "The folder " + dir.path + "was not created",
                        Toast.LENGTH_SHORT
                    )
                        .show()
                }
            }
            val fileList = context.assets.list("tessdata")

            for (fileName in fileList!!) {
                val pathToDataFile = "$dir/$fileName"
                if (!File(pathToDataFile).exists()) {
                    val input = context.assets.open("tessdata/$fileName")
                    val output = FileOutputStream(pathToDataFile)
                    val buff = ByteArray(1024)
                    while (true) {
                        val len = input.read(buff)
                        if (len <= 0) break
                        output.write(buff, 0, len)
                    }
                    input.close()
                    output.close()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, e.message)
        }

    }

    fun getOCRResult(bitmap: Bitmap): String {
        val whitelist = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.-!?/"
        mTess.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, whitelist)
        mTess.setImage(bitmap)
        return mTess.utF8Text
    }

    fun destroy() {
        mTess.end()
    }

}