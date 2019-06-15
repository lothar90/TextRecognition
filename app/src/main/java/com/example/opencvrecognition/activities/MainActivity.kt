package com.example.opencvrecognition.activities

import android.content.Intent
import android.os.Bundle
import android.support.design.widget.NavigationView
import android.support.v4.view.GravityCompat
import android.support.v4.widget.DrawerLayout
import android.support.v7.app.ActionBar
import android.support.v7.widget.Toolbar
import android.view.MenuItem
import android.view.View
import android.widget.RadioButton
import android.widget.Toast
import com.example.opencvrecognition.R
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : BaseActivity() {

    private val TAG = "MainActivity"
    private lateinit var drawerLayout: DrawerLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!checkPermission())
            requestPermission()

        drawerLayout = findViewById(R.id.drawer_layout)
        val navigationView: NavigationView = findViewById(R.id.navigationViewMain)
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
                    if (!checkPermission()) {
                        Toast.makeText(
                            this,
                            "To start working you need to give all required permissions",
                            Toast.LENGTH_SHORT
                        ).show()
                        requestPermission()
                    } else {
                        val intent = Intent(this, MSERPhotoActivity::class.java)
                        startActivity(intent)
                    }
                }
                menuItem.itemId == R.id.swt_activity -> {
                    if (!checkPermission()) {
                        Toast.makeText(
                            this,
                            "To start working you need to give all required permissions",
                            Toast.LENGTH_SHORT
                        ).show()
                        requestPermission()
                    } else {
                        val intent = Intent(this, SWTActivity::class.java)
                        startActivity(intent)
                    }
                }
                menuItem.itemId == R.id.east_activity -> {
                    if (!checkPermission()) {
                        Toast.makeText(
                            this,
                            "To start working you need to give all required permissions",
                            Toast.LENGTH_SHORT
                        ).show()
                        requestPermission()
                    } else {
                        val intent = Intent(this, EASTPhotoActivity::class.java)
                        startActivity(intent)
                    }
                }
            }
            true
        }

        val toolbar: Toolbar = findViewById(R.id.toolbarMain)
        setSupportActionBar(toolbar)
        val actionbar: ActionBar? = supportActionBar
        actionbar?.apply {
            setDisplayHomeAsUpEnabled(true)
            setHomeAsUpIndicator(R.drawable.ic_menu)
        }
    }

    fun onTypeRadioButtonClick(v: View) {
        if (v is RadioButton) {
            // Check which radio button was clicked
            when (v.getId()) {
                R.id.photoRadioButton -> {
                    v.isChecked = true
                }
                R.id.cameraRadioButton -> {
                    v.isChecked = true
                }
            }
        }
    }

    fun onMethodRadioButtonClick(v: View) {
        if (v is RadioButton) {
            // Check which radio button was clicked
            when (v.getId()) {
                R.id.MSERRadioButton -> {
                    v.isChecked = true
                }
                R.id.EASTRadioButton -> {
                    v.isChecked = true
                }
                R.id.SWTRadioButton -> {
                    v.isChecked = false
                    Toast.makeText(this, "SWT method is not working properly", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    fun onConfirmButtonClick(v: View) {
        if (!checkPermission()) {
            Toast.makeText(this, "To start working you need to give all required permissions", Toast.LENGTH_SHORT)
                .show()
            requestPermission()
        } else {
            if (photoRadioButton.isChecked) {
                if (MSERRadioButton.isChecked) {
                    val intent = Intent(this, MSERPhotoActivity::class.java)
                    startActivity(intent)
                }
                if (EASTRadioButton.isChecked) {
                    val intent = Intent(this, EASTPhotoActivity::class.java)
                    startActivity(intent)
                }
            }
            if (cameraRadioButton.isChecked) {
                if (MSERRadioButton.isChecked) {
                    val intent = Intent(this, MSERCameraActivity::class.java)
                    startActivity(intent)
                }
                if (EASTRadioButton.isChecked) {
                    val intent = Intent(this, EASTCameraActivity::class.java)
                    startActivity(intent)
                }
            }
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
