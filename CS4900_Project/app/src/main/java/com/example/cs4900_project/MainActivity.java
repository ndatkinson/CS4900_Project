package com.example.cs4900_project;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Build;
import android.os.Bundle;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]
                    {android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
}