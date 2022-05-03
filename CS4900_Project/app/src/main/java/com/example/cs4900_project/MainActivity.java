package com.example.cs4900_project;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;


import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;

import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;

import java.io.File;

import java.io.FileOutputStream;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;


import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;


public class MainActivity extends AppCompatActivity {

    Bitmap bitmap;
    Module module;
    //0 background
    //1 aeroplane
    //2 bicycle
    //3 bird
    //4 boat
    //5 bottle
    //6 bus
    //7 car
    //8 cat
    //9 chair
    //10 cow
    //11 diningtable
    //12 dog
    //13 horse
    //14 motorbike
    //15 person
    //16 pottedplant
    //17 sheep
    //18 sofa
    //19 train
    //20 tvmonitor
    private static final int CLASSNUM = 20;
    private static final int BACKGROUND = 0;
    private static final int AEROPLANE = 1;
    private static final int BICYCLE = 2;
    private static final int BIRD = 3;
    private static final int BOAT = 4;
    private static final int BOTTLE = 5;
    private static final int BUS = 6;
    private static final int CAR = 7;
    private static final int CAT = 8;
    private static final int CHAIR = 9;
    private static final int COW = 10;
    private static final int DININGTABLE = 11;
    private static final int DOG = 12;
    private static final int HORSE = 13;
    private static final int MOTORBIKE = 14;
    private static final int PERSON = 15;
    private static final int POTTEDPLANT = 16;
    private static final int SHEEP = 17;
    private static final int SOFA = 18;
    private static final int TRAIN = 19;
    private static final int TVMONITOR = 20;


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
        //button to start the process of segmenting the selected image
        buttonSegment.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.O)
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
                bitmap = Bitmap.createScaledBitmap(bitmap, 400, 300, false);
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


        //when loading in the actual model, replace the one in the assetFilePath asset name
        // with the custom trained model name.

        // much of the segmentation code below was gotten from the pytorch website
        // on image segmentation: https://pytorch.org/tutorials/beginner/deeplabv3_on_android.html

        try {
            module = LiteModuleLoader.load(assetFilePath(this, "newest_unet_5_epochs.py"));
            tv.setText("Model found");

            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB);

            final float[] inputs = inputTensor.getDataAsFloatArray();

            final Tensor outTensor = module.forward(IValue.from(inputTensor)).toTensor();
            final float[] outputs = outTensor.getDataAsFloatArray();




            int width = bitmap.getWidth();
            int height = bitmap.getHeight();
            int[] intValues = new int[outputs.length];
            for (int j = 0; j < width; j++) {

                    // maxi: the index of the 21 CLASSNUM with the max probability
                    int maxi = 0;
                    double maxnum = -100000.0;
                    for (int i = 0; i < CLASSNUM; i++) {
                        if (outputs[i] > maxnum) {
                            maxnum = outputs[i];
                            maxi = i;

                        }
                    }
                    // color coding for person (red), dog (green), sheep (blue)
                    // black color for background and other classes
                    if (maxi == PERSON)
                        intValues[j] = Color.RED; // red
                    else if (maxi == DOG)
                        intValues[j] = Color.GREEN; // green
                    else if (maxi == AEROPLANE)
                        intValues[j] = Color.YELLOW; // blue
                    else if (maxi == BICYCLE)
                        intValues[j] = Color.BLUE; // blue
                    else if (maxi == BIRD)
                        intValues[j] = Color.GRAY; // blue
                    else if (maxi == BOAT)
                        intValues[j] = Color.WHITE; // blue
                    else if (maxi == BOTTLE)
                        intValues[j] = Color.RED; // blue
                    else if (maxi == BUS)
                        intValues[j] = Color.GREEN; // blue
                    else if (maxi == CAR)
                        intValues[j] = Color.YELLOW; // blue
                    else if (maxi == CAT)
                        intValues[j] = Color.BLUE; // blue
                    else if (maxi == CHAIR)
                        intValues[j] = Color.GRAY; // blue
                    else if (maxi == COW)
                        intValues[j] = Color.WHITE; // blue
                    else if (maxi == DININGTABLE)
                        intValues[j] = Color.RED; // blue
                    else if (maxi == HORSE)
                        intValues[j] = Color.GREEN; // blue
                    else if (maxi == MOTORBIKE)
                        intValues[j] = Color.YELLOW; // blue
                    else if (maxi == POTTEDPLANT)
                        intValues[j] = Color.BLUE; // blue
                    else if (maxi == SOFA)
                        intValues[j] = Color.GRAY; // blue
                    else if (maxi == TRAIN)
                        intValues[j] = Color.WHITE; // blue
                    else if (maxi == TVMONITOR)
                        intValues[j] = Color.RED; // blue
                    else if (maxi == SHEEP)
                        intValues[j] = Color.BLUE; // blue
                    else
                        intValues[j] = Color.BLACK; // black

            }

            Bitmap bmpSegmentation = Bitmap.createScaledBitmap(bitmap, width, height, true);
            Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
            outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0,
                    outputBitmap.getWidth(), outputBitmap.getHeight());
            iv.setImageBitmap(outputBitmap);

        } catch (IOException e) {
            e.printStackTrace();
            tv.setText("Model not found");
        }
    }






    //method that gets the model from the assets folder
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
