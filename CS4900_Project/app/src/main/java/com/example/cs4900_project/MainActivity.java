package com.example.cs4900_project;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.media.Image;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import org.pytorch.LiteModuleLoader;
import java.io.BufferedReader;
import java.io.Console;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.PyTorchAndroid;

import java.io.IOException;
public class MainActivity extends AppCompatActivity {

    Bitmap bitmap = null;
    Module module = null;
    int DIM_X = bitmap.getWidth();
    int DIM_Y = bitmap.getHeight();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]
                    {android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);



        Button buttonLoad = findViewById(R.id.btnLoad);
        //button to load image into the ImageView
        buttonLoad.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, 3);
            }
        });

        Button buttonSegment = findViewById(R.id.btnSegment);
        //button to start the process of segmenting the image
        buttonSegment.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
              segment();

            }
        });


    }

    //this method handles the selection of the image from the emulator storage and displays it in the imageView
    //also creates a bitmap from the selected image
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            ImageView imageView = findViewById(R.id.imageView);
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getApplicationContext().getContentResolver(), selectedImage);
                imageView.setImageBitmap(bitmap);

            } catch (IOException e) {
                e.printStackTrace();
            }
        }


    }

    //this method handles the segmentation of the image
    protected void segment(){
        TextView tv= findViewById(R.id.textView);

        ImageView iv = findViewById(R.id.imageView);
        normalize();

        //when loading in the actual model, replace the one in the assetFilePath asset name
        // with the custom trained model name.
        try {
            module = LiteModuleLoader.load(assetFilePath(this, "scripted_resnet18_optimized.py"));
            if(module != null){
                tv.setText("Model found");
            }
            else{
                tv.setText("Model not found");
            }

        } catch (IOException e) {
            e.printStackTrace();

        }
    }

    private void normalize() {
        float[][][][] input = new float[1][DIM_X][DIM_Y][3];
        for (int x = 0; x < DIM_X; x++) {
            for (int y = 0; y < DIM_Y; y++) {
                int pixel = bitmap.getPixel(x, y);
                // Normalize channel values to [0.0, 1.0]
                input[0][x][y][0] = Color.red(pixel) / 255.0f;
                input[0][x][y][1] = Color.green(pixel) / 255.0f;
                input[0][x][y][2] = Color.blue(pixel) / 255.0f;
            }
        }
    }


    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }


    }


}