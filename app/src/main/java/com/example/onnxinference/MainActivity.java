package com.example.onnxinference;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;

import com.example.onnxinference.detectionPipeline.ImageProcessor;
import com.example.onnxinference.detectionPipeline.Yolov8Inference;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private List<String> classNames;
    private ImageView imageView;
    private ImageProcessor imageProcessor;
    private Yolov8Inference yolo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        classNames = loadClassNames();
        imageProcessor = new ImageProcessor();
        yolo = new Yolov8Inference(this);

        imageView = findViewById(R.id.imageView);

        Bitmap bitmap = loadImageFromAssets("test.jpg");
        float[][][][] inputTensor = imageProcessor.preprocessImage(bitmap);

        float[][][] result = yolo.runInference(inputTensor);

        float confidenceThreshold = 0.5f;
        float iouThreshold = 0.5f;
        Bitmap resultBitmap = imageProcessor.processOutput(result, bitmap, classNames, confidenceThreshold, iouThreshold);
        imageView.setImageBitmap(resultBitmap);
    }

    private List<String> loadClassNames() {
        List<String> classNames = new ArrayList<>();
        try {
            InputStream inputStream = getResources().openRawResource(R.raw.labels);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = reader.readLine()) != null) {
                classNames.add(line.trim());
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return classNames;
    }

    private Bitmap loadImageFromAssets(String fileName) {
        try {
            InputStream inputStream = getAssets().open(fileName);
            return BitmapFactory.decodeStream(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
