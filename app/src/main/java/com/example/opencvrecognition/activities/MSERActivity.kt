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
import com.example.opencvrecognition.utilities.MSEROperations
import com.squareup.picasso.Picasso
import kotlinx.android.synthetic.main.activity_mser.*


class MSERActivity : BaseActivity() {

    private val SELECT_PHOTO = 100
    private val TAG = "MSERActivity"
    private var photoLoaded = false
    private lateinit var drawerLayout: DrawerLayout
    private lateinit var mserOperations: MSEROperations
    private lateinit var originalBitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_mser)
        setTitle(R.string.MSER)
        mserOperations = MSEROperations()
        Log.d(TAG, "Initialized MSEROperations")

        if (!checkPermission())
            requestPermission()

        drawerLayout = findViewById(R.id.drawer_layout)
        val navigationView: NavigationView = findViewById(R.id.navigationView)
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

        val toolbar: Toolbar = findViewById(R.id.toolbarMSER)
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

    fun onMserRegionsActionButtonClick(view: View) {
        if (photoLoaded) {
            var bitmap = originalBitmap.copy(originalBitmap.config, true)
            if (bitmap.height > 2000 || bitmap.width > 1000)
                bitmap = Bitmap.createScaledBitmap(
                    bitmap,
                    (bitmap.width * 0.25).toInt(),
                    (bitmap.height * 0.25).toInt(),
                    false
                )

            mserOperations.initialize()
            mserOperations.bitmap = bitmap
            mserOperations.detectMSERRegions()
            galleryImageViewMSER.setImageBitmap(mserOperations.bitmap)
            Toast.makeText(this, "Detection completed", Toast.LENGTH_SHORT).show()
        } else
            Toast.makeText(this, "First load a photo", Toast.LENGTH_SHORT).show()
    }

    fun onMserMaskActionButtonClick(view: View) {
        if (photoLoaded) {
            var bitmap = originalBitmap.copy(originalBitmap.config, true)
            if (bitmap.height > 2000 || bitmap.width > 1000)
                bitmap = Bitmap.createScaledBitmap(
                    bitmap,
                    (bitmap.width * 0.25).toInt(),
                    (bitmap.height * 0.25).toInt(),
                    false
                )

            mserOperations.initialize()
            mserOperations.bitmap = bitmap
            mserOperations.createMSERMask()
            galleryImageViewMSER.setImageBitmap(mserOperations.bitmap)
            Toast.makeText(this, "Detection completed", Toast.LENGTH_SHORT).show()
        } else
            Toast.makeText(this, "First load a photo", Toast.LENGTH_SHORT).show()
    }

    fun onMserMorphoActionButtonClick(view: View) {
        if (photoLoaded) {
            var bitmap = originalBitmap.copy(originalBitmap.config, true)
            if (bitmap.height > 2000 || bitmap.width > 1000)
                bitmap = Bitmap.createScaledBitmap(
                    bitmap,
                    (bitmap.width * 0.25).toInt(),
                    (bitmap.height * 0.25).toInt(),
                    false
                )

            mserOperations.initialize()
            mserOperations.bitmap = bitmap
            mserOperations.maskAfterMorphology()
            galleryImageViewMSER.setImageBitmap(mserOperations.bitmap)
            Toast.makeText(this, "Detection completed", Toast.LENGTH_SHORT).show()
        } else
            Toast.makeText(this, "First load a photo", Toast.LENGTH_SHORT).show()
    }

    fun onMserDetectedContoursActionButtonClick(view: View) {
        if (photoLoaded) {
            var bitmap = originalBitmap.copy(originalBitmap.config, true)
            if (bitmap.height > 2000 || bitmap.width > 1000)
                bitmap = Bitmap.createScaledBitmap(
                    bitmap,
                    (bitmap.width * 0.25).toInt(),
                    (bitmap.height * 0.25).toInt(),
                    false
                )

            mserOperations.initialize()
            mserOperations.bitmap = bitmap
            mserOperations.drawAllFoundContours()
            galleryImageViewMSER.setImageBitmap(mserOperations.bitmap)
            Toast.makeText(this, "Detection completed", Toast.LENGTH_SHORT).show()
        } else
            Toast.makeText(this, "First load a photo", Toast.LENGTH_SHORT).show()
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

            mserOperations.initialize()
            mserOperations.bitmap = bitmap
            mserOperations.detectTextRegions()
            galleryImageViewMSER.setImageBitmap(mserOperations.bitmap)
            Toast.makeText(this, "Detection completed", Toast.LENGTH_SHORT).show()
        } else
            Toast.makeText(this, "First load a photo", Toast.LENGTH_SHORT).show()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK)
            if (requestCode == SELECT_PHOTO) {
                Picasso.get().load(data?.data).noPlaceholder().into(galleryImageViewMSER)
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
