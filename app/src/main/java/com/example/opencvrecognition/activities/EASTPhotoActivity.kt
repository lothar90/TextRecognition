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
import com.example.opencvrecognition.utilities.EASTOperations
import com.squareup.picasso.Picasso
import kotlinx.android.synthetic.main.activity_east_photo.*
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.concurrent.thread


class EASTPhotoActivity : BaseActivity() {

    private val TAG = "EASTOperations"
    private val SELECT_PHOTO = 100
    private lateinit var drawerLayout: DrawerLayout
    private var photoLoaded = false
    private lateinit var originalBitmap: Bitmap
    private lateinit var mRgba: Mat
    private var loadedNetwork = false
    private val eastOperations = EASTOperations(this)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_east_photo)

        drawerLayout = findViewById(R.id.drawer_layout)
        val navigationView: NavigationView = findViewById(R.id.navigationViewEAST)
        navigationView.setNavigationItemSelectedListener { menuItem ->
            // set item as selected to persist highlight
            menuItem.isChecked = true
            // close drawer when item is tapped
            drawerLayout.closeDrawers()

            when {
                menuItem.itemId == R.id.main_activity -> {
                    val intent = Intent(this, MainActivity::class.java)
                    startActivity(intent)
                }
                menuItem.itemId == R.id.mser_activity -> {
                    val intent = Intent(this, MSERPhotoActivity::class.java)
                    startActivity(intent)
                }
                menuItem.itemId == R.id.swt_activity -> {
                    val intent = Intent(this, SWTActivity::class.java)
                    startActivity(intent)
                }
                menuItem.itemId == R.id.east_activity -> {
                    val intent = Intent(this, EASTPhotoActivity::class.java)
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

        thread(start = true) {
            eastOperations.initialize()
            loadedNetwork = true
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
        if (photoLoaded && loadedNetwork) {
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
            eastOperations.mRgba = mRgba
            eastOperations.doEAST()
            Log.i(TAG, "Detection time:" + (System.currentTimeMillis() - milis))
            Utils.matToBitmap(mRgba, bitmap)
            galleryImageViewEAST.setImageBitmap(bitmap)
            Toast.makeText(this, "Detection completed", Toast.LENGTH_SHORT).show()
        } else
            Toast.makeText(this, "First load a photo", Toast.LENGTH_SHORT).show()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK)
            if (requestCode == SELECT_PHOTO) {
                Picasso.get().load(data?.data).noPlaceholder().into(galleryImageViewEAST)
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
