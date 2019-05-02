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
import com.example.opencvrecognition.utilities.Ray
import com.example.opencvrecognition.utilities.SWTPoint2d
import com.squareup.picasso.Picasso
import kotlinx.android.synthetic.main.activity_swt.*
import org.opencv.android.Utils
import org.opencv.core.CvType.*
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.Canny
import org.opencv.imgproc.Imgproc.GaussianBlur
import kotlin.math.*


class SWTActivity : BaseActivity() {

    private val TAG = "SWTOperations"
    private val SELECT_PHOTO = 100
    private var photoLoaded = false
    private lateinit var drawerLayout: DrawerLayout
    private lateinit var mRgba: Mat
    private lateinit var mGray: Mat
    private lateinit var originalBitmap: Bitmap
    var lowThreshold = 1.0
    var highThreshold = 1.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_swt)

        drawerLayout = findViewById(R.id.drawer_layout)
        val navigationView: NavigationView = findViewById(R.id.navigationViewSWT)
        navigationView.setNavigationItemSelectedListener { menuItem ->
            // set item as selected to persist highlight
            menuItem.isChecked = true
            // close drawer when item is tapped
            drawerLayout.closeDrawers()

            if (menuItem.itemId == R.id.mser_activity) {
                val intent = Intent(this, MSERActivity::class.java)
                startActivity(intent)
            } else if (menuItem.itemId == R.id.swt_activity) {
                val intent = Intent(this, SWTActivity::class.java)
                startActivity(intent)
            }
            true
        }

        val toolbar: Toolbar = findViewById(R.id.toolbarSWT)
        setSupportActionBar(toolbar)
        val actionbar: ActionBar? = supportActionBar
        actionbar?.apply {
            setDisplayHomeAsUpEnabled(true)
            setHomeAsUpIndicator(R.drawable.ic_menu)
        }

        lowThreshValue.text = lowThreshold.toString()
        highThreshValue.text = highThreshold.toString()
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
        var milis = System.currentTimeMillis()
        doCanny()
        Log.i(TAG, "Detection time:" + (System.currentTimeMillis() - milis))

        milis = System.currentTimeMillis()
        doSWT()
        Log.i(TAG, "SWT time:" + (System.currentTimeMillis() - milis))
    }

    fun doCanny() {
        val tempMat = Mat()
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
            mGray = Mat()
            Utils.bitmapToMat(bitmap, mRgba)
            Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_RGB2GRAY)
            lowThreshold = Imgproc.threshold(mGray, tempMat, 0.0, 255.0, Imgproc.THRESH_OTSU or Imgproc.THRESH_BINARY)
            highThreshold = if (lowThreshold * 3 < 255.0) lowThreshold * 3 else 255.0
            val detectedEdges = Mat()
            Imgproc.blur(mGray, detectedEdges, Size(3.0, 3.0))
            Canny(detectedEdges, detectedEdges, lowThreshold, highThreshold, 3)
            mGray = detectedEdges
            Utils.matToBitmap(mGray, bitmap)
            galleryImageViewSWT.setImageBitmap(bitmap)
            lowThreshValue.text = lowThreshold.toString()
            highThreshValue.text = highThreshold.toString()
        } else
            Toast.makeText(this, "Nie wczytano żadnego zdjęcia", Toast.LENGTH_SHORT).show()
    }

    fun doSWT() {
        var SWTMat = Mat(mGray.size(), CV_8SC1, Scalar.all(-1.0))
        var gradX = Mat()
        var gradY = Mat()
        var gaussian = Mat()
        var gray = Mat()
        Imgproc.cvtColor(mRgba, gray, Imgproc.COLOR_RGB2GRAY)
        gray.convertTo(gaussian, CV_32FC1, 1.0 / 255.0)
        GaussianBlur(gaussian, gaussian, Size(5.0, 5.0), 0.0)
        Imgproc.Scharr(gaussian, gradX, CV_64F, 1, 0)
        Imgproc.Scharr(gaussian, gradY, CV_64F, 0, 1)
        GaussianBlur(gradX, gradX, Size(3.0, 3.0), 0.0)
        GaussianBlur(gradY, gradY, Size(3.0, 3.0), 0.0)
        var prec = .05
        for (row in 0 until mGray.rows()) {
            for (col in 0 until mGray.cols()) {
                var r = Ray()

                var p = SWTPoint2d()
                p.x = col
                p.y = row
                r.p = p
                var points = ArrayList<SWTPoint2d>()
                points.add(p)

                var curX = col.toDouble() + 0.5
                var curY = row.toDouble() + 0.5
                var curPixX = col
                var curPixY = row
                var G_x = gradX.get(row, col)[0]
                var G_y = gradY.get(row, col)[0]

                var mag = sqrt((G_x * G_x) + (G_y * G_y))
                G_x = -G_x / mag
                G_y = -G_y / mag

                while (true) {
                    curX += G_x * prec
                    curY += G_y * prec
                    if ((floor(curX).toInt()) != curPixX || (floor(curY).toInt()) != curPixY) {
                        curPixX = (floor(curX).toInt())
                        curPixY = (floor(curY).toInt())
                        if (curPixX < 0 || (curPixX >= mGray.cols()) || curPixY < 0 || (curPixY >= mGray.rows()))
                            break
                        var pnew = SWTPoint2d()
                        pnew.x = curPixX
                        pnew.y = curPixY
                        points.add(pnew)

                        if (mGray.get(curPixY, curPixX)[0] > 0) {
                            r.q = pnew
                            var G_xt = gradX.get(curPixY, curPixX)[0]
                            var G_yt = gradY.get(curPixY, curPixX)[0]
                            mag = sqrt((G_xt * G_xt) + (G_yt * G_yt))
                            G_x = -G_x / mag
                            G_y = -G_y / mag
                            if (acos(G_x * -G_xt + G_y * -G_yt) < PI / 2.0) {
                                var length = sqrt(((r.q.x - r.p.x) * (r.q.x - r.p.x) + (r.q.y - r.p.y) * (r.q.y - r.p.y)).toDouble())
                                for (pit in points) {
                                    if (SWTMat.get(pit.y, pit.x)[0] < 0)
                                        SWTMat.get(pit.y, pit.x)[0] = length
                                    else
                                        SWTMat.get(pit.y, pit.x)[0] = min(length, SWTMat.get(pit.y, pit.x)[0])
                                }
                                r.points = points
                            }
                            break
                        }
                    }
                }
            }
        }
        mGray = SWTMat
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
