package com.example.opencvrecognition.activities

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.support.design.widget.NavigationView
import android.support.v4.view.GravityCompat
import android.support.v4.widget.DrawerLayout
import android.support.v7.app.ActionBar
import android.support.v7.widget.Toolbar
import android.util.Log
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import com.example.opencvrecognition.R
import com.squareup.picasso.Picasso
import kotlinx.android.synthetic.main.activity_east.*
import kotlinx.android.synthetic.main.activity_swt.*
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import org.opencv.utils.Converters


class EASTActivity : BaseActivity() {

    private val TAG = "EASTOperations"
    private val SELECT_PHOTO = 100
    private lateinit var drawerLayout: DrawerLayout
    private var photoLoaded = false
    private lateinit var originalBitmap: Bitmap
    private lateinit var mRgba: Mat
    private lateinit var net: Net

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_east)

        net = Dnn.readNetFromTensorflow("./additions/frozen_east_text_detection.pb")

        drawerLayout = findViewById(R.id.drawer_layout)
        val navigationView: NavigationView = findViewById(R.id.navigationViewEAST)
        navigationView.setNavigationItemSelectedListener { menuItem ->
            // set item as selected to persist highlight
            menuItem.isChecked = true
            // close drawer when item is tapped
            drawerLayout.closeDrawers()

            when {
                menuItem.itemId == R.id.mser_activity -> {
                    val intent = Intent(this, MSERActivity::class.java)
                    startActivity(intent)
                }
                menuItem.itemId == R.id.swt_activity -> {
                    val intent = Intent(this, SWTActivity::class.java)
                    startActivity(intent)
                }
                menuItem.itemId == R.id.east_activity -> {
                    val intent = Intent(this, EASTActivity::class.java)
                    startActivity(intent)
                }
            }
            true
        }

        val toolbar: Toolbar = findViewById(R.id.toolbarEAST)
        setSupportActionBar(toolbar)
        val actionbar: ActionBar? = supportActionBar
        actionbar?.apply {
            setDisplayHomeAsUpEnabled(true)
            setHomeAsUpIndicator(R.drawable.ic_menu)
        }
    }

    fun onGalleryActionButtonClick(view: View) {
        if (checkPermission()) {
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            startActivityForResult(Intent.createChooser(intent, "Select Picture"), SELECT_PHOTO)
        } else
            Toast.makeText(this, "No permission to show gallery", Toast.LENGTH_SHORT).show()
    }

    fun onPlayActionButtonClick(view: View) {
        if (photoLoaded) {
            var bitmap = originalBitmap.copy(originalBitmap.config, true)
            if (bitmap.height > 2000 || bitmap.width > 1000)
                bitmap = Bitmap.createScaledBitmap(
                    bitmap,
                    (bitmap.width * 0.25).toInt(),
                    (bitmap.height * 0.25).toInt(),
                    false
                )
            mRgba = Mat()
            Utils.bitmapToMat(bitmap, mRgba)
            Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2RGB)
            val milis = System.currentTimeMillis()
            //doEAST()
            Log.i(TAG, "Detection time:" + (System.currentTimeMillis() - milis))
            Utils.matToBitmap(mRgba, bitmap)
            galleryImageViewEAST.setImageBitmap(bitmap)
            Toast.makeText(this, "Detection completed", Toast.LENGTH_SHORT).show()
        } else
            Toast.makeText(this, "First load a photo", Toast.LENGTH_SHORT).show()
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

        val confidences = MatOfFloat(Converters.vector_float_to_Mat(confidencesList))
        val boxesArray = boxesList.toArray() as Array<RotatedRect>
        val boxes = MatOfRotatedRect(*boxesArray)
        val indices = MatOfInt()
        //Dnn.NMSBoxesRotated(boxes, confidences, scoreThresh, nmsThresh, indices)

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
                Imgproc.line(mRgba, vertices[j], vertices[(j + 1) % 4], Scalar(255.0, 255.0, 0.0), 1)
        }
    }

    fun decode(
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

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK)
            if (requestCode == SELECT_PHOTO) {
                Picasso.get().load(data?.data).noPlaceholder().into(galleryImageViewSWT)
                originalBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, data?.data)
                photoLoaded = true
            }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                drawerLayout.openDrawer(GravityCompat.START)
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}
