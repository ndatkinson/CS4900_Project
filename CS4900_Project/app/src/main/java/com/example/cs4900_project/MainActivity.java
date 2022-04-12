package com.example.cs4900_project;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;

import android.media.Image;
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
import java.util.Map;


import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class MainActivity extends AppCompatActivity {

    Bitmap bitmap;
    Module module = null;

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


        //when loading in the actual model, replace the one in the assetFilePath asset name
        // with the custom trained model name.
        // I got most of the segmentation code below from the pytorch website:
        // https://pytorch.org/tutorials/beginner/deeplabv3_on_android.html
        try {
            module = LiteModuleLoader.load(assetFilePath(this, "model3.py"));
            tv.setText("Model found");


            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB);
            final float[] inputs = inputTensor.getDataAsFloatArray();

            final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
            //Map<String, IValue> outTensors =
            //        module.forward(IValue.from(inputTensor)).toDictStringKey();


// the key "out" of the output tensor contains the semantic masks
// see https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101
            //final Tensor outputTensor = outTensor.get("out").toTensor();
            final float[] outputs = outputTensor.getDataAsFloatArray();

            int width = bitmap.getWidth();
            int height = bitmap.getHeight();
            int[] intValues = new int[width * height];
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

    /*private void normalize() {
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
    }*/


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
